import os
import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import json

def _np_bf16_dtype_or_fp16():
    try:
        return np.dtype('bfloat16') 
    except Exception:
        return np.float16

def _load_embeddings(model_path: str, adapter_path: str = None, torch_dtype: torch.dtype = torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, low_cpu_mem_usage=True,
            torch_dtype=torch_dtype, device_map=None,
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, low_cpu_mem_usage=True,
            torch_dtype=torch_dtype, device_map=None,
        )
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

    with torch.no_grad():
        emb = model.get_input_embeddings().weight.detach().to(torch.float16).cpu().numpy()
    vocab_dict = tokenizer.get_vocab()
    id_to_token = [None] * len(vocab_dict)
    for tok, idx in vocab_dict.items():
        if idx < len(id_to_token):
            id_to_token[idx] = tok
    for i, v in enumerate(id_to_token):
        if v is None:
            id_to_token[i] = f"<unk_{i}>"
    return emb, id_to_token

def _whiten(X: np.ndarray, eps: float = 1e-6):
    device = 'cuda'
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mean = Xt.mean(dim=0, keepdim=True)
    Xc = Xt - mean
    n = max(1, Xc.shape[0] - 1)
    cov = (Xc.transpose(0, 1) @ Xc) / float(n)
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=eps)
    inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.transpose(0, 1)
    X_hat = Xc @ inv_sqrt
    return (
        X_hat.detach().to('cpu', dtype=torch.float16).numpy(),
        mean.detach().to('cpu', dtype=torch.float16).numpy(),
        inv_sqrt.detach().to('cpu', dtype=torch.float16).numpy(),
    )

def _ridge_t2s(X_t: np.ndarray, Y_s: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    device = 'cuda'
    Xt = torch.as_tensor(X_t, dtype=torch.float32, device=device)
    Ys = torch.as_tensor(Y_s, dtype=torch.float32, device=device)
    XtX = Xt.transpose(0, 1) @ Xt
    A = XtX + lam * torch.eye(XtX.shape[0], dtype=torch.float32, device=device)
    XtY = Xt.transpose(0, 1) @ Ys
    W_T = torch.linalg.solve(A, XtY)
    return W_T.transpose(0, 1).to('cpu', dtype=torch.float16).numpy()

def _project_embeddings(teacher_emb: np.ndarray, W: np.ndarray, row_bs: int = 8192) -> np.ndarray:
    V_T, d_t = teacher_emb.shape
    d_s, d_t_w = W.shape
    assert d_t == d_t_w, "Dimension mismatch between teacher_emb and W"
    result = np.empty((V_T, d_s), dtype=np.float16)
    device = 'cuda'
    W_tT = torch.as_tensor(W, dtype=torch.float32, device=device).transpose(0, 1)
    with torch.no_grad():
        for i in tqdm(range(0, V_T, row_bs), desc="Project T->S", unit="blk"):
            ie = min(i + row_bs, V_T)
            block = torch.as_tensor(teacher_emb[i:ie], dtype=torch.float32, device=device)
            result[i:ie] = (block @ W_tT).to('cpu', dtype=torch.float16).numpy()
    return result

def _normalize_rows_gpu(A: torch.Tensor) -> torch.Tensor:
    return A / (A.norm(dim=1, keepdim=True) + 1e-8)

@torch.no_grad()
def _sinkhorn_block_gpu_and_save(An: torch.Tensor,
                                 Bn: torch.Tensor,
                                 reg: float,
                                 topk: int,
                                 out_path: str,
                                 save_dtype: np.dtype,
                                 row_bs: int,
                                 col_bs: int,
                                 max_iter: int,
                                 tol: float,
                                 save_transport_optimized: bool = True):
    """Blockwise GPU Sinkhorn without materializing K or C on host.
    - If topk>0: directly saves sparse top-k per row as .npz (indices int32, values save_dtype).
    - If topk<=0: stream-writes dense matrix to .npy using memmap with save_dtype.
    """
    device = An.device
    n, m = An.size(0), Bn.size(0)
    a = torch.full((n,), 1.0 / float(n), device=device)
    b = torch.full((m,), 1.0 / float(m), device=device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Helper: K^T @ u and K @ v via blocks
    def KT_times_u(u_vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(m, device=device)
        for j0 in range(0, m, col_bs):
            je = min(j0 + col_bs, m)
            acc = torch.zeros(je - j0, device=device)
            for i0 in range(0, n, row_bs):
                ie = min(i0 + row_bs, n)
                sim = An[i0:ie] @ Bn[j0:je].T               # [rb, cb]
                Kblk = torch.exp(-(1.0 - sim) / reg)        # [rb, cb]
                acc += (Kblk.T @ u_vec[i0:ie])              # [cb]
            out[j0:je] = acc
        return out

    def K_times_v(v_vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n, device=device)
        for i0 in range(0, n, row_bs):
            ie = min(i0 + row_bs, n)
            acc = torch.zeros(ie - i0, device=device)
            for j0 in range(0, m, col_bs):
                je = min(j0 + col_bs, m)
                sim = An[i0:ie] @ Bn[j0:je].T
                Kblk = torch.exp(-(1.0 - sim) / reg)
                acc += (Kblk @ v_vec[j0:je])
            out[i0:ie] = acc
        return out

    # Sinkhorn iterations
    for it in tqdm(range(max_iter), desc="Sinkhorn (GPU blk)", unit="it"):
        # v <- b / (K^T u)
        KT_u = KT_times_u(u) + 1e-12
        v_new = b / KT_u
        # u <- a / (K v_new)
        K_v = K_times_v(v_new) + 1e-12
        u_new = a / K_v
        # check convergence
        if torch.max(torch.abs(u_new - u)).item() < tol and torch.max(torch.abs(v_new - v)).item() < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new

    Ktop = min(topk, m)
    inds = torch.empty((n, Ktop), dtype=torch.int32, device='cpu')
    vals = torch.empty((n, Ktop), dtype=torch.float32, device='cpu')
    for i0 in tqdm(range(0, n, row_bs), desc="Save topk", unit="blk"):
        ie = min(i0 + row_bs, n)
        # init per-row buffers
        best_vals = torch.full((ie - i0, 0), -float('inf'), device=device)
        best_inds = torch.empty((ie - i0, 0), dtype=torch.int64, device=device)
        for j0 in range(0, m, col_bs):
            je = min(j0 + col_bs, m)
            sim = An[i0:ie] @ Bn[j0:je].T                      # [rb, cb]
            Kblk = torch.exp(-(1.0 - sim) / reg)               # [rb, cb]
            Pblk = (u[i0:ie].unsqueeze(1) * Kblk) * v[j0:je].unsqueeze(0)
            # merge candidates
            merged_vals = torch.cat([best_vals, Pblk], dim=1)  # [rb, Kprev+cb]
            merged_inds = torch.cat([best_inds, torch.arange(j0, je, device=device).repeat(ie - i0, 1)], dim=1)
            top_vals, top_pos = torch.topk(merged_vals, k=min(Ktop, merged_vals.size(1)), dim=1)
            top_inds = torch.gather(merged_inds, 1, top_pos)
            best_vals, best_inds = top_vals, top_inds
            # free
            del sim, Kblk, Pblk, merged_vals, merged_inds, top_vals, top_pos, top_inds
        inds[i0:ie] = best_inds.to('cpu', dtype=torch.int32)
        vals[i0:ie] = best_vals.to('cpu', dtype=torch.float32)
        del best_vals, best_inds
    
    # Save with optimization metadata for faster Wasserstein computation
    # save_data = {
    #     'format': np.array(["topk"], dtype=object),
    #     'indices': inds_sorted,
    #     'values': vals_sorted,
    #     'k': np.array([Ktop], dtype=np.int32),
    #     'shape': np.array([n, m], dtype=np.int32)
    # }
    
    
    # np.savez(out_path, **save_data)
    save_data = {}
    for i in range(n):
        for j in range(Ktop):
            save_data[(i, j)] = vals[i, j]
    with open(out_path, "w") as f:
        json.dump(save_data, f)


def main():
    parser = argparse.ArgumentParser(description="Generate global alignment and projection for FKD_H")
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--teacher-adapter-path", type=str, default=None)
    parser.add_argument("--student-adapter-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True, help="Dense: .npy (topk=-1), Sparse: .npz (topk>0)")
    parser.add_argument("--save-projection-path", type=str, required=True, help="Output .pt for W_q init")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1)
    parser.add_argument("--teacher-vocab-max", type=int, default=None)
    parser.add_argument("--student-vocab-max", type=int, default=None)
    parser.add_argument("--topk", type=int, default=-1, help="TopK=-1: dense full matrix; TopK>0: per-row top-K sparse")
    # GPU block sizes and iterations
    parser.add_argument("--row-batch", type=int, default=4096)
    parser.add_argument("--col-batch", type=int, default=8192)
    parser.add_argument("--sinkhorn-iters", type=int, default=200)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-5)
    args = parser.parse_args()

    print("Loading teacher embeddings...")
    teacher_emb, teacher_tokens = _load_embeddings(args.teacher_model, args.teacher_adapter_path)
    print("Loading student embeddings...")
    student_emb, student_tokens = _load_embeddings(args.student_model, args.student_adapter_path)

    if args.teacher_vocab_max and teacher_emb.shape[0] > args.teacher_vocab_max:
        teacher_emb = teacher_emb[:args.teacher_vocab_max]; teacher_tokens = teacher_tokens[:args.teacher_vocab_max]
    if args.student_vocab_max and student_emb.shape[0] > args.student_vocab_max:
        student_emb = student_emb[:args.student_vocab_max]; student_tokens = student_tokens[:args.student_vocab_max]

    print(f"Teacher embedding shape: {teacher_emb.shape}")
    print(f"Student embedding shape: {student_emb.shape}")

    # Overlap pairs (token-string exact match)
    st_dict = {tok: idx for idx, tok in enumerate(student_tokens)}
    t_idx, s_idx = [], []
    for i, tok in enumerate(teacher_tokens):
        j = st_dict.get(tok, None)
        if j is not None:
            t_idx.append(i); s_idx.append(j)
    print(f"Found {len(t_idx)} overlapping tokens")
    if len(t_idx) < 100:
        print("Warning: Very few overlapping tokens found. Results may be poor.")

    teacher_overlap = teacher_emb[t_idx]
    student_overlap = student_emb[s_idx]

    print("Whitening embeddings...")
    teacher_white, _, _ = _whiten(teacher_overlap)
    student_white, _, _ = _whiten(student_overlap)

    print("Learning teacher->student projection W via ridge...")
    W = _ridge_t2s(teacher_white, student_white, args.ridge_lambda) # [d_s, d_t]

    print("Projecting full teacher embedding table...")
    teacher_proj = _project_embeddings(teacher_emb, W)

    # Normalize on GPU
    device = 'cuda'
    An = _normalize_rows_gpu(torch.as_tensor(teacher_proj, dtype=torch.float32, device=device))  # [V_T, d_s]
    Bn = _normalize_rows_gpu(torch.as_tensor(student_emb, dtype=torch.float32, device=device))   # [V_S, d_s]

    print("Running blockwise GPU Sinkhorn and saving...")
    save_dtype = _np_bf16_dtype_or_fp16()
    _sinkhorn_block_gpu_and_save(
        An=An,
        Bn=Bn,
        reg=float(args.sinkhorn_reg),
        topk=int(args.topk) if int(args.topk) > 0 else -1,
        out_path=args.output_path,
        save_dtype=save_dtype,
        row_bs=int(args.row_batch),
        col_bs=int(args.col_batch),
        max_iter=int(args.sinkhorn_iters),
        tol=float(args.sinkhorn_tol),
        save_transport_optimized=True, 
    )

    print(f"Saving projection to {args.save_projection_path}")
    embedding_projection_state = {
        'weight': torch.from_numpy(W), 
        'bias': torch.zeros(W.shape[0]),
    }
    torch.save(embedding_projection_state, args.save_projection_path)
    print("Done!")
    print(f"Projection matrix shape: {W.shape}")

if __name__ == "__main__":
    main()

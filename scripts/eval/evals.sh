# bash run_eval.sh /workspace/DSKD/models/sft_gpt2-120m 16 gpt2
# bash run_eval.sh /workspace/DSKD/models/uld_gpt2-120m 16 gpt2
# bash run_eval.sh /workspace/DSKD/models/dskd_gpt2-120m 16 gpt2
# bash run_eval.sh /workspace/DSKD/models/mined_gpt2-120m 16 gpt2
# bash run_eval.sh /workspace/DSKD/models/multiot_gpt2-120m 16 gpt2
# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__epoch=20__bsz=4x2x1=8__lr=0.0005/epoch20_step28580_loss6.3126_rougel24.6371

# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/hoang_sft/MCW_KD_GPT2_SFT-1

# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/uld/MCW_KD_GPT2_ULD
# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/dskd/MCW_KD_GPT2_DSKD
# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/mined/MCW_KD_GPT2_MinED
# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/multiot/MCW_KD_GPT2_MultiOT

# bash /workspace/DSKD/scripts/eval/run_eval.sh outputs/gpt2/gpt2-base/dual_space_kd_with_cma/criterion=dual_space_kd_with_cma__forward_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=0.5__kd^temp=2.0__epoch=20__bsz=4x2x1=8__lr=0.0005__proj^lr=0.001/epoch18_step25722_loss8.0516_rougel26.1376
# bash /workspace/DSKD/scripts/eval/run_eval.sh /workspace/DSKD/outputs/gpt2/gpt2-base/wctkd/criterion=wctkd__forward_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=0.5__kd^temp=2.0__wctkd^alpha=0.5__wctkd^beta=0.2__wctkd^gamma=0.3__wctkd^hidden_gamma=0.5__wctkd^top_k=8__epoch=20__bsz=4x2x1=8__lr=0.0005__proj^lr=0.001/epoch20_step28580_loss7.7107_rougel25.4203

# bash /workspace/DSKD/scripts/eval/run_eval_lora.sh /workspace/DSKD/outputs/tinyllama/tinyllama-1.1b-3T/wctkd/criterion=wctkd__forward_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=mistral__kd^rate=0.5__kd^temp=2.0__wctkd^alpha=0.5__wctkd^beta=0.2__wctkd^gamma=0.3__wctkd^hidden_gamma=0.5__wctkd^top_k=8__epoch=15__bsz=4x2x1=8__lr=0.001/epoch8_step11432_loss2.5627_rougel29.7835 /workspace/DSKD/model_hub/tinyllama/tinyllama-1.1b-3T
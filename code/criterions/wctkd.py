import math
import torch
import torch.nn.functional as F
from .various_divergence import VariousDivergence

import json


class WCTKD(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.alpha = args.wctkd_alpha
        self.beta = args.wctkd_beta
        self.gamma = args.wctkd_gamma
        self.hidden_gamma = args.wctkd_hidden_gamma
        self.top_k = args.wctkd_top_k
        self.input_max_length = args.max_length
        self.M_global_path = args.M_global_path
        self.M_global: dict[tuple[int, int], float] = self.load_M_global()

    def load_M_global(self):
        with open(self.M_global_path, "r") as f:
            M_global = json.load(f)
        return M_global

    def forward(
        self,
        distiller,
        input_data,
        output_data,
        logging_output,
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[
                    f"teacher_{distiller.teacher_model_type}_attention_mask"
                ],
                position_ids=input_data.get(
                    f"teacher_{distiller.teacher_model_type}_position_ids", None
                ),
                output_hidden_states=True,
            )

        wctkd_loss, log = self.compute_wctkd_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        loss = self.alpha * ce_loss + self.beta * wctkd_loss + self.gamma * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits,
            output_data["label"],
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    def compute_wctkd_loss(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # compute BI
        teacher_hidden_states = teacher_outputs.hidden_states
        num_teacher_layer = len(teacher_hidden_states) - 1
        batch_size = teacher_hidden_states.shape[0]
        batch_bi_scores = torch.zeros((num_teacher_layer, batch_size))
        for layer_idx in range(num_teacher_layer):
            X_l = teacher_hidden_states[layer_idx]
            Y_l = teacher_hidden_states[layer_idx + 1]
            # Normalize
            X_l = F.normalize(X_l, dim=-1)
            Y_l = F.normalize(Y_l, dim=-1)
            cosine_score = F.cosine_similarity(X_l, Y_l, dim=-1)
            scores = 1 - cosine_score.mean().item()
            batch_bi_scores[layer_idx, :] = scores

        avg_bi_scores = batch_bi_scores.mean(dim=-1)

        top_indices = sorted(
            range(len(avg_bi_scores)), key=lambda x: avg_bi_scores[x], reverse=True
        )[: self.top_k]

        top_bi_scores = avg_bi_scores[top_indices]
        # softmax top_bi_scores
        log["top_k_indices"] = top_indices
        log["top_k_bi_scores"] = top_bi_scores

        layer_weights = torch.softmax(top_bi_scores, dim=-1) # [K]

        # calculate overlap position between student and teacher
        overlaps = torch.zeros(
            (batch_size, self.input_max_length, self.input_max_length)
        )
        student_tokenizer = distiller.student_tokenizer
        teacher_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]
        for b in range(batch_size):
            student_input_ids = input_data["input_ids"][b]
            teacher_input_ids = input_data[
                f"teacher_{distiller.teacher_model_type}_input_ids"
            ][b]
            student_text = student_tokenizer.decode(student_input_ids)
            teacher_text = teacher_tokenizer.decode(teacher_input_ids)
            student_special_mask = student_tokenizer.get_special_tokens_mask(
                student_input_ids, already_has_special_tokens=True
            )
            teacher_special_mask = teacher_tokenizer.get_special_tokens_mask(
                teacher_input_ids, already_has_special_tokens=True
            )
            student_encoded = student_tokenizer(
                student_text, return_offsets_mapping=True
            )
            teacher_encoded = teacher_tokenizer(
                teacher_text, return_offsets_mapping=True
            )
            student_offsets = student_encoded.offset_mapping
            teacher_offsets = teacher_encoded.offset_mapping
            for i in range(len(student_offsets)):
                if student_special_mask[i]:
                    continue
                s_start, s_end = student_offsets[i]
                if s_start == s_end:
                    continue
                for j in range(len(teacher_offsets)):
                    if teacher_special_mask[j]:
                        continue
                    t_start, t_end = teacher_offsets[j]
                    if t_start == t_end:
                        continue
                    if s_start >= t_end or s_end <= t_start:
                        continue
                    overlaps[b, i, j] = 1

        global_scores = torch.zeros(
            (batch_size, self.input_max_length, self.input_max_length)
        )
        for b in range(batch_size):
            for i in range(self.input_max_length):
                for j in range(self.input_max_length):
                    if overlaps[b, i, j] == 1:
                        global_scores[b, i, j] = self.M_global[(i, j)]

        student_hidden_states = (
            outputs.hidden_states
        )  # [num_layer, batch_size, seq_len, hidden_size]
        teacher_hidden_states = (
            teacher_outputs.hidden_states
        )  # [num_layer, batch_size, seq_len, hidden_size]

        A = torch.zeros(
            (self.top_k, batch_size, self.input_max_length, self.input_max_length)
        )
        H_T = teacher_hidden_states.shape[-1]
        H_S = student_hidden_states.shape[-1]
        teacher_hidden_states_projected = torch.zeros(
            (self.top_k, batch_size, self.input_max_length, H_S)
            )
        student_hidden_states_stacked = torch.zeros(
            (self.top_k, batch_size, self.input_max_length, H_S)
        )
        for k in range(self.top_k):
            teacher_layer_idx = top_indices[k]
            student_layer_idx = teacher_layer_idx // num_teacher_layer

            teacher_hidden_state = teacher_hidden_states[teacher_layer_idx] # [B, T, H]
            student_hidden_state = student_hidden_states[student_layer_idx]  # [B, S, H]

            teacher_hidden_states_projected[k] = distiller.hidden_states_projectors[
                f"teacher_{teacher_layer_idx}"
            ](teacher_hidden_state)  # [B, T, H]
            student_hidden_states_stacked[k] = student_hidden_state

            contextual = torch.bmm(
                student_hidden_state, teacher_hidden_states_projected[k].transpose(-1, -2)
            )  # [B, S, T]

            hybrid = (
                1.0 - self.hidden_gamma
            ) * contextual + self.hidden_gamma * global_scores
            A[k] = hybrid.masked_fill(~overlaps[b], 0.0)

        A_softmax = torch.softmax(A, dim=-1)
        H_tilte = torch.einsum("kbst,kbth->kbsh", A_softmax, teacher_hidden_states_projected)
        H_tilte_norm = F.normalize(H_tilte, dim=-1)
        student_hidden_states_stacked_norm = F.normalize(student_hidden_states_stacked, dim=-1)
        cosines = 1.0 - F.cosine_similarity(H_tilte_norm, student_hidden_states_stacked_norm, dim=-1) # [K, B, S]
        weighted_cosines = torch.sum((cosines * layer_weights.unsqueeze(-1)), dim=0) # [B, S]
        pad_mask = output_data["label"].ne(self.padding_id)
        wctkd_loss = (weighted_cosines * pad_mask).sum()
        log["wctkd_loss"] = wctkd_loss
        return wctkd_loss, log



    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]

        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") and hasattr(
            distiller.student_model.model, "embed_tokens"
        ):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif (
            hasattr(distiller.student_model, "model")
            and hasattr(distiller.student_model.model, "model")
            and hasattr(distiller.student_model.model.model, "embed_tokens")
        ):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") and hasattr(
            distiller.student_model.transformer, "wte"
        ):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") and hasattr(
            distiller.teacher_model.model, "embed_tokens"
        ):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif (
            hasattr(distiller.teacher_model, "model")
            and hasattr(distiller.teacher_model.model, "model")
            and hasattr(distiller.teacher_model.model.model, "embed_tokens")
        ):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") and hasattr(
            distiller.teacher_model.model, "wte"
        ):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(
            pad_mask, input_data["input_ids"], torch.zeros_like(target)
        )
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(
            teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target)
        )
        formal_teacher_input = torch.where(
            teacher_pad_mask,
            input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
            torch.zeros_like(teacher_target),
        )
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()

        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(
            -1
        ) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs

        if (
            not self.args.only_save_projector
        ):  # skip if only train projectors (pre-train projectors)
            t2s_kd_loss = self.dist_func(
                outputs.logits,
                t2s_logits.detach(),
                target,
                reduction="none",
                use_tea_temp=True,
            )
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()

            s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            s2t_logits = s2t_hiddens.matmul(
                distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            )

            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            )
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (
                (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum()
                * pad_mask.sum()
                / teacher_pad_mask.sum()
            )

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log


import torch
from contextlib import nullcontext
from torch import autocast
from trainer.dpo_base import (
    DLDPOConfig,
    DLDPOTrainer,
    DataCollatorAndTokenizerForPreference,
)

__all__ = ["DLDPOConfig", "DPOLLaDATrainer", "DataCollatorAndTokenizerForPreference"]


class DPOLLaDATrainer(DLDPOTrainer):
    def _prepare_inputs_for_mc(self, input_ids, attention_mask, loss_mask):
        """LLaDA does not require any sequence-length alignment — return as-is."""
        return input_ids, attention_mask, loss_mask

    @torch.no_grad()
    def _sample_mask_for_completions(
        self, loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select completion tokens to replace with [MASK].

        Args:
            loss_mask: [N, L] bool tensor where True marks completion token positions.

        Returns:
            mask_positions: [N, L] bool — True at positions replaced with [MASK].
            p_per_row:      [N]    float — masking rate p = k / completion_len per row.
        """
        device = loss_mask.device
        N, L   = loss_mask.shape

        # Number of completion tokens per row
        completion_len = loss_mask.sum(dim=1)  # [N]

        mask_positions = torch.zeros((N, L), dtype=torch.bool,    device=device)
        p_per_row      = torch.zeros(N,      dtype=torch.float32, device=device)

        for i in range(N):
            tlen = int(completion_len[i].item())
            if tlen == 0:
                continue
            # Draw k_i uniformly from [1, tlen]; p_i = k_i / tlen is the masking rate
            k_i = torch.randint(1, tlen + 1, (), device=device).item()
            p_per_row[i] = float(k_i) / float(tlen)

            # Positions of completion tokens in this row
            comp_idx = torch.nonzero(loss_mask[i], as_tuple=True)[0]  # [tlen]

            # Select which positions to mask
            if self.args.block_wise:
                # Block-wise masking: pick a random block of `block_size` completion tokens,
                # then randomly mask k_i of them (k_i is already proportional to tlen).
                block_size = self.args.block_size
                if tlen <= block_size:
                    # Completion fits in one block — treat the whole completion as the block
                    block_start, block_end = 0, tlen
                else:
                    # Randomly pick a block start within the completion
                    max_block_start = tlen - block_size
                    block_start = torch.randint(0, max_block_start + 1, (), device=device).item()
                    block_end   = block_start + block_size

                block_indices = comp_idx[block_start:block_end]  # completion positions in the block
                block_len     = len(block_indices)
                k_i_block     = min(k_i, block_len)              # mask at most block_len tokens
                if k_i_block > 0:
                    perm   = torch.randperm(block_len, device=device)
                    chosen = block_indices[perm[:k_i_block]]
                    mask_positions[i, chosen] = True
            else:
                # Uniform masking: randomly pick k_i positions from all completion tokens
                perm   = torch.randperm(tlen, device=device)
                chosen = comp_idx[perm[:k_i]]
                mask_positions[i, chosen] = True

        return mask_positions, p_per_row

    def _mc_forward_step(
        self,
        model,
        ref_model,
        input_ids,       # [r*2B, L]
        attention_mask,  # [r*2B, L]
        targets,         # [r*2B, L]  original token ids (ground truth for scoring)
        loss_mask,       # [r*2B, L]  True at completion token positions
        mask_token_id,   # int  token id used to replace masked positions
    ):
        """
        LLaDA MC forward step.

        Randomly masks a subset of completion tokens (via ``_sample_mask_for_completions``),
        runs the policy and reference models on the masked input, and returns per-sequence
        log-probability estimates for both.

        Optionally applies Classifier-Free Guidance (CFG): when ``args.cfg_scale > 0``,
        an unconditional copy (with the prompt also masked) is concatenated along the
        batch dimension.  The guided logits are computed for both policy and ref models.
        """
        # Sample which completion positions to replace with [MASK]
        mask_positions, p_per_row = self._sample_mask_for_completions(loss_mask)

        # Build the masked input
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_positions] = mask_token_id  # [r*2B, L]

        model_kwargs = {
            "use_cache":           False,
            "output_hidden_states": True,
            "attention_mask":      attention_mask,
        }

        if self.args.cfg_scale > 0.0:
            # Unconditional: additionally mask the prompt tokens
            # (positions that are attended to but not in the completion)
            is_prompt    = (~loss_mask) & attention_mask.bool()
            ids_uncond   = masked_input_ids.clone()
            ids_uncond[is_prompt] = mask_token_id
            # Concatenate conditional and unconditional along the batch dim
            model_input_ids = torch.cat([masked_input_ids, ids_uncond], dim=0)  # [2*r*2B, L]
            model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask], dim=0)
        else:
            model_input_ids = masked_input_ids  # [r*2B, L]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            outputs = model(model_input_ids, **model_kwargs)

        logits = outputs.logits  # [r*2B, L, V]  (or [2*r*2B, L, V] with CFG)
        if self.args.cfg_scale > 0.0:
            logits_cond, logits_uncond = torch.chunk(logits, 2, dim=0)
            logits = logits_uncond + (self.args.cfg_scale + 1.0) * (logits_cond - logits_uncond)

        # Reference model forward (no grad, optional autocast for PEFT bf16)
        ref_context_manager = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with torch.no_grad(), ref_context_manager:
            ref_outputs = ref_model(model_input_ids, **model_kwargs)

        ref_logits = ref_outputs.logits  # [r*2B, L, V]  (or [2*r*2B, L, V] with CFG)
        if self.args.cfg_scale > 0.0:
            ref_logits_cond, ref_logits_uncond = torch.chunk(ref_logits, 2, dim=0)
            ref_logits = ref_logits_uncond + (self.args.cfg_scale + 1.0) * (ref_logits_cond - ref_logits_uncond)

        logpb,     log_dict     = self.compute_diffusion_logps(logits,     targets, mask_positions, p_per_row)
        ref_logpb, ref_log_dict = self.compute_diffusion_logps(ref_logits, targets, mask_positions, p_per_row)
        return logpb, ref_logpb, log_dict, ref_log_dict


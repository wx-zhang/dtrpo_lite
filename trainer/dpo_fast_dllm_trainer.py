"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""


import torch
from contextlib import nullcontext
from torch import autocast
from trainer.dpo_base import DLDPOTrainer
from trl.trainer.utils import pad_to_length


class FastDLLMDPOTrainer(DLDPOTrainer):

    def _prepare_inputs_for_mc(self, input_ids, attention_mask, loss_mask):
        """Pad input_ids, attention_mask, and loss_mask to a multiple of block_size.

        Fast-dLLM uses block-diagonal attention, so the sequence length must be
        a multiple of the model's block size (``model.model.bd_size``).
        """
        block_size = self.model.model.bd_size
        seq_len    = input_ids.shape[1]
        pad_len    = (block_size - (seq_len % block_size)) % block_size
        if pad_len > 0:
            input_ids      = pad_to_length(input_ids,      seq_len + pad_len, self.padding_value)
            attention_mask = pad_to_length(attention_mask, seq_len + pad_len, 0)
            loss_mask      = pad_to_length(loss_mask,      seq_len + pad_len, False)
        return input_ids, attention_mask, loss_mask

    def _mc_forward_step(
        self,
        model,
        ref_model,
        input_ids,       # [r*2B, L]  repeated input token ids
        attention_mask,  # [r*2B, L]
        targets,         # [r*2B, L]  original token ids (unused directly; targets derived from labels)
        loss_mask,       # [r*2B, L]  True at completion token positions
        mask_token_id,   # not used in Fast-dLLM (no random masking); kept for interface compat
    ):
        """
        Fast-dLLM MC forward step.

        Fast-dLLM uses a shifted-label scheme instead of random token masking:
          - ``labels[t]`` = ``input_ids[t+1]`` — each position predicts the next token.
          - Completion positions use the real next token; all other positions are set to -100.
          - The reference model receives ``outputs.noisy_input_ids``, the internally
            diffusion-noised sequence produced by the policy model's forward pass.

        Sequences are first padded to a multiple of ``block_size`` by
        ``_prepare_inputs_for_mc``, required by Fast-dLLM's block-diagonal attention.
        """
        # Pad to multiple of block_size (Fast-dLLM requirement)
        input_ids, attention_mask, loss_mask = self._prepare_inputs_for_mc(
            input_ids, attention_mask, loss_mask
        )

        # Build shifted labels: label[t] = next token = input_ids[t+1]
        labels    = torch.roll(input_ids.clone(), shifts=-1, dims=1)  # [r*2B, L]
        loss_mask = torch.roll(loss_mask,         shifts=-1, dims=1)  # shift mask to align
        labels[~loss_mask] = -100  # ignore prompt and padding positions

        model_kwargs = {
            "attention_mask": attention_mask,
            "labels":         labels,
        }

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            outputs = model(input_ids, **model_kwargs)

        # Ref model runs on the diffusion-noised sequence from the policy model's forward pass
        ref_context_manager = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        noisy_ids = outputs.noisy_input_ids.to(device)
        ref_model_kwargs = {
            "attention_mask": model_kwargs["attention_mask"].to(device),
            "labels":         model_kwargs["labels"].to(device),
        }
        with torch.no_grad(), ref_context_manager:
            ref_outputs = ref_model(noisy_ids, **ref_model_kwargs)

        logits     = outputs.logits      # [r*2B, L, V]
        ref_logits = ref_outputs.logits  # [r*2B, L, V]

        # Scoring mask: positions where the model predicts a real completion token
        mask_positions = labels != -100  # [r*2B, L]

        # Replace -100 with 0 so gather ops are safe; non-scored positions are
        # masked out inside compute_diffusion_logps and never affect the result.
        targets_clean = labels.clone()
        targets_clean[labels == -100] = 0

        # Masking rate: fraction of scored tokens per sequence (used by vrpo estimator)
        p_per_row = mask_positions.sum(dim=1).float() / mask_positions.shape[1]  # [r*2B]
        p_per_row = torch.clamp(p_per_row, min=1e-6)

        # Ensure scoring tensors match the device of each model's logits (FSDP may place tensors on CPU)
        logpb, log_dict = self.compute_diffusion_logps(
            logits,
            targets_clean.to(logits.device),
            mask_positions.to(logits.device),
            p_per_row.to(logits.device),
        )
        ref_logpb, ref_log_dict = self.compute_diffusion_logps(
            ref_logits,
            targets_clean.to(ref_logits.device),
            mask_positions.to(ref_logits.device),
            p_per_row.to(ref_logits.device),
        )
        # Align return tensors to the same device (policy logits device) for the caller's accumulators
        ref_logpb = ref_logpb.to(logpb.device)
        return logpb, ref_logpb, log_dict, ref_log_dict

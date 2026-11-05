"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""


from dataclasses import dataclass

import torch
from trainer.dpo_base import DLDPOTrainer
from trainer.trl_compat import pad_to_length


@dataclass
class PreparedFastDLLMBatch:
    """Clean inputs and labels reused across diffusion trajectories."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    num_examples: int


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
        return input_ids.to(dtype=torch.long), attention_mask, loss_mask.bool()

    def _prepare_forward_inputs(self, input_ids, attention_mask, loss_mask):
        """Apply Fast-dLLM block padding and construct shifted completion labels."""
        # Pad to multiple of block_size (Fast-dLLM requirement)
        input_ids, attention_mask, loss_mask = self._prepare_inputs_for_mc(
            input_ids, attention_mask, loss_mask
        )

        # Build shifted labels: label[t] = next token = input_ids[t+1]
        input_ids = input_ids.to(dtype=torch.long)
        labels    = torch.roll(input_ids.clone(), shifts=-1, dims=1)  # [r*2B, L]
        loss_mask = torch.roll(loss_mask,         shifts=-1, dims=1)  # shift mask to align
        labels[~loss_mask] = -100  # ignore prompt and padding positions
        labels = labels.to(device=input_ids.device, dtype=torch.long)

        return input_ids, attention_mask, labels

    def _prepare_preference_batch(self, batch) -> PreparedFastDLLMBatch:
        """Prepare one clean batch for policy/reference scoring and replay."""
        input_ids, attention_mask, loss_mask, num_examples = (
            self._prepare_preference_inputs(batch)
        )
        input_ids, attention_mask, labels = self._prepare_forward_inputs(
            input_ids, attention_mask, loss_mask
        )
        return PreparedFastDLLMBatch(input_ids, attention_mask, labels, num_examples)

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
        input_ids, attention_mask, labels = self._prepare_forward_inputs(
            input_ids, attention_mask, loss_mask
        )

        model_kwargs = {
            "attention_mask": attention_mask,
            "labels":         labels,
            "compute_loss":   False,
        }

        # with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        outputs = model(input_ids, **model_kwargs)

        device = input_ids.device
        noisy_ids = outputs.noisy_input_ids.to(device=device, dtype=torch.long)
        ref_model_kwargs = {
            "attention_mask": model_kwargs["attention_mask"].to(device),
            "labels":         labels.clone().to(device=device, dtype=torch.long),
            "compute_loss":   False,
        }

        # TRL prepares the frozen FSDP reference model in eval mode. Fast-dLLM
        # only consumes precomputed_noisy_input_ids in its training forward, so
        # restore that mode immediately before every reference-model call.
        ref_model.train()
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids,
                precomputed_noisy_input_ids=noisy_ids,
                **ref_model_kwargs,
            )

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
            unmask_positions=(
                log_dict["unmask_positions"].to(ref_logits.device)
                if log_dict["unmask_positions"] is not None else None
            ),
        )

        # Align return tensors to the same device (policy logits device) for the caller's accumulators
        ref_logpb = ref_logpb.to(logpb.device)
        return logpb, ref_logpb, log_dict, ref_log_dict

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

DiffuLLM-DPO Trainer
--------------------
A lightweight drop-in trainer that mirrors the public interface of TRL's
`DPOTrainer`, but swaps the log-prob computation with a diffusion-LLM
log-likelihood estimator

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import pad
from trainer.trl_compat import flush_left, flush_right, maybe_apply_chat_template, maybe_extract_prompt


def _repeat_for_mc(x: torch.Tensor, repeats: int) -> torch.Tensor:
    # Repeat rows r times: [B, ...] -> [B*r, ...]
    # repeats are contiguous: x0,x0,...,x0, x1,x1,...,x1, ...
    return x.repeat_interleave(repeats, dim=0)


# =========================
# Config
# =========================
@dataclass
class DLDPOConfig(DPOConfig):
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "Loss type for DPO training."},
    )

    mc_samples: int = 2
    mc_batch_size: int = 1
    mask_token_id: int = 0
    cfg_scale: float = 0.0
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    logp_type: str = "dtrpo"
    unmask_size: int = 4
    block_size: int = 32
    block_wise: bool = False


# =========================
# Dataset collator
# Only tokenize dataset when collect the dataset, so that each rank only tokenize their own dataset
# =========================


class DataCollatorAndTokenizerForPreference(DataCollatorMixin):
    """
    Data collator for preference datasets with dynamic tokenization.

    This collator reproduces the functionality of the original
    `DataCollatorForPreference` but performs tokenization inside
    the collator instead of requiring pre-tokenized datasets.

    It supports:
    - prompt/chosen/rejected fields (string inputs)
    - dynamic tokenization per batch
    - padding of all sequences
    - optional fields such as pixel_values, image_sizes,
      ref_chosen_logps/ref_rejected_logps
    """

    def __init__(
        self,
        process_fcn_kwargs,
        process_fcn,
        pad_token_id: int,
        return_tensors: str = "pt",
    ):
        self.process_fcn_kwargs = process_fcn_kwargs
        self.process_fcn = process_fcn
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Expect dataset examples in string format:
        # {"prompt": str, "chosen": str, "rejected": str, ...}

        # Tokenize on-the-fly
        tokenized = []
        for ex in examples:
            process_kwargs = dict(self.process_fcn_kwargs)
            if ex.get("_is_chat_template_applied") is not None:
                process_kwargs["is_chat"] = bool(ex["_is_chat_template_applied"])
            tokenized.append(self.process_fcn(ex, **process_kwargs))

        # Convert to tensors
        prompt_input_ids = [torch.tensor(
            e["prompt_input_ids"], dtype=torch.long) for e in tokenized]
        chosen_input_ids = [torch.tensor(
            e["chosen_input_ids"], dtype=torch.long) for e in tokenized]
        rejected_input_ids = [torch.tensor(
            e["rejected_input_ids"], dtype=torch.long) for e in tokenized]

        prompt_attention_mask = [torch.ones_like(x) for x in prompt_input_ids]
        chosen_attention_mask = [torch.ones_like(x) for x in chosen_input_ids]
        rejected_attention_mask = [
            torch.ones_like(x) for x in rejected_input_ids]

        # Optional additional fields (kept identical to the original collator)
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(ex["pixel_values"])
                            for ex in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [
                torch.tensor(ex["pixel_attention_mask"]) for ex in examples
            ]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor(
                [ex["ref_chosen_logps"] for ex in examples])
            ref_rejected_logps = torch.tensor(
                [ex["ref_rejected_logps"] for ex in examples]
            )

        # Pad sequences
        output = {}
        output["prompt_input_ids"] = pad(
            prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        output["prompt_attention_mask"] = pad(
            prompt_attention_mask, padding_value=0, padding_side="left"
        )
        output["chosen_input_ids"] = pad(
            chosen_input_ids, padding_value=self.pad_token_id
        )
        output["chosen_attention_mask"] = pad(
            chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(
            rejected_input_ids, padding_value=self.pad_token_id
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask, padding_value=0
        )

        # Attach optional fields if present
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(
                pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor(
                [ex["image_sizes"] for ex in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output


# =========================
# Trainer
# =========================
class DLDPOTrainer(DPOTrainer):
    """
    DLDPOTrainer that disables tokenization during dataset
    preparation and instead relies on `DataCollatorAndTokenizerForPreference`
    to perform dynamic tokenization at collator time.
    """

    def __init__(
        self,
        model,
        ref_model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        **kwargs,
    ):
        self.is_vision_model = False
        model_id = model.config._name_or_path
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model_id, fix_mistral_regex=True,
            )
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token

        process_fcn_kwargs = {
            "processing_class": processing_class,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_length,
            "add_special_tokens": False,
        }
        data_collator = DataCollatorAndTokenizerForPreference(
            process_fcn_kwargs=process_fcn_kwargs,
            process_fcn=self.tokenize_row
            if not self.is_vision_model
            else self.process_row,
            pad_token_id=processing_class.pad_token_id,
        )

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,  # Keep raw string data
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )
        self.padding_value = processing_class.pad_token_id

    # tokenize_row is inherited from DPOTrainer and called by the custom collator

    def log(self, logs, *args, **kwargs):
        """Shim: transformers >=4.48 passes start_time as a second arg."""
        return super().log(logs)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[
            PreTrainedTokenizerBase,
            BaseImageProcessor,
            FeatureExtractionMixin,
            ProcessorMixin,
        ],
        args: DPOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """
        Override the base method: do not tokenize the dataset here.
        Simply return the raw dataset and let the collator handle tokenization.
        """
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(
            dataset, Dataset
        ):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        state = PartialState()
        if isinstance(dataset, Dataset):
            # Mapped caches can be shared by checkouts with different preprocessing
            # implementations. Rebuild on rank 0, then let other ranks load its output.
            map_kwargs["load_from_cache_file"] = not state.is_main_process

        with state.main_process_first():
            # Extract prompt if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                **map_kwargs,
            )

        return dataset

    def _prepare_inputs_for_mc(self, input_ids, attention_mask, loss_mask):
        raise NotImplementedError("Subclasses must implement this method")

    def _mc_forward_step(
        self,
        model,
        ref_model,
        input_ids,
        attention_mask,
        targets,
        loss_mask,
        mask_token_id,
    ):
        raise NotImplementedError("Subclasses must implement this method")

    def _prepare_preference_inputs(self, batch):
        """Concatenate preference pairs, truncate and align their loss masks."""
        assert (
            not self.is_encoder_decoder
        ), "Encoder–decoder path is not implemented in this trainer."
        assert (
            getattr(self, "padding_free", False) is False
        ), "Disable padding_free for masking variant to keep shapes simple."
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.padding_value
        )

        # Build concatenated decoder-only inputs
        prompt_input_ids = concatenated_batch["prompt_input_ids"]  # [2B, Lp]
        # [2B, Lp]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        # [2B, Lc]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch[
            "completion_attention_mask"
        ]  # [2B, Lc]

        input_ids = torch.cat(
            (prompt_input_ids, completion_input_ids), dim=1
        )  # [2B, L]
        input_ids = input_ids.to(dtype=torch.long)
        attention_mask = torch.cat(
            (prompt_attention_mask, completion_attention_mask), dim=1
        )  # [2B, L]
        # completion loss mask: prompt 0s then completion mask
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        ).bool()

        # Truncation path (same as original)
        if self.max_length is not None and self.max_length < attention_mask.size(1):
            if self.truncation_mode == "keep_start":
                attention_mask, input_ids, loss_mask = flush_left(
                    attention_mask, input_ids, loss_mask
                )
                attention_mask = attention_mask[:, : self.max_length]
                input_ids = input_ids[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]
            elif self.truncation_mode == "keep_end":
                attention_mask, input_ids, loss_mask = flush_right(
                    attention_mask, input_ids, loss_mask
                )
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                loss_mask = loss_mask[:, -self.max_length:]
                attention_mask, input_ids, loss_mask = flush_left(
                    attention_mask, input_ids, loss_mask
                )
            else:
                raise ValueError(
                    f"Unknown truncation mode: {self.truncation_mode}")
        else:
            attention_mask, input_ids, loss_mask = flush_left(
                attention_mask, input_ids, loss_mask
            )

        return input_ids, attention_mask, loss_mask, num_examples

    def concatenated_forward_with_random_masking(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        mc_samples: int = None,
        mc_batch_size: int = None,
        mask_token_id: int = None,
        cfg_scale: float = None,
    ) -> dict[str, torch.Tensor]:
        """
        Decoder-only path: compute chosen/rejected scores via MC masked-token estimator.
        - Randomly mask some completion tokens in each pass.
        - Predict masked tokens and debias by 1/p (p = masked_fraction).
        - Average across MC samples to get sequence scores.
        """
        input_ids, attention_mask, loss_mask, num_examples = (
            self._prepare_preference_inputs(batch)
        )

        # prepare ref model as it shares the same mc_samples as the model
        ref_model = self.ref_model if self.ref_model is not None else self.model

        # ---- MC estimation ----
        B2, L = input_ids.shape
        device = input_ids.device

        original_ids = input_ids.clone()
        target_ids = original_ids  # [2B, L]

        logpbs_sum = torch.zeros(B2, device=device, dtype=torch.float32)
        ref_logpbs_sum = torch.zeros(B2, device=device, dtype=torch.float32)

        repeats_done = 0
        while repeats_done < mc_samples:
            r = min(mc_batch_size, mc_samples - repeats_done)
            # Repeat each sequence r times so one forward pass covers r MC samples
            input_ids_rep    = _repeat_for_mc(original_ids,    r)  # [r*2B, L]
            attn_mask_rep    = _repeat_for_mc(attention_mask,  r)  # [r*2B, L]
            loss_mask_rep    = _repeat_for_mc(loss_mask,       r)  # [r*2B, L]
            targets_rep      = _repeat_for_mc(target_ids,      r)  # [r*2B, L]

            logpb_full, ref_logpb_full, log_dict, ref_log_dict = self._mc_forward_step(
                model, ref_model,
                input_ids_rep, attn_mask_rep, targets_rep, loss_mask_rep, mask_token_id,
            )

            logpb_full = logpb_full.view(B2, r)
            logpbs_sum += logpb_full.sum(dim=1)
            ref_logpb_full = ref_logpb_full.view(B2, r)
            ref_logpbs_sum += ref_logpb_full.sum(dim=1)

            repeats_done += r

        # Final per-sequence estimates
        all_logps_est = logpbs_sum / mc_samples  # [2B]
        all_logps_est_ref = ref_logpbs_sum / mc_samples  # [2B]

        output = {
            "chosen_logps": all_logps_est[:num_examples],
            "rejected_logps": all_logps_est[num_examples:],
            "chosen_logps_ref": all_logps_est_ref[:num_examples],
            "rejected_logps_ref": all_logps_est_ref[num_examples:],
        }

        output["mean_chosen_logits"] = output["chosen_logps"]
        output["mean_rejected_logits"] = output["rejected_logps"]

        output.update(log_dict)
        for key in ref_log_dict:
            output[key + "_ref"] = ref_log_dict[key]
        return output

    def compute_diffusion_logps(
        self,
        logits: torch.Tensor,          # [N_seq, L, V]
        targets: torch.Tensor,         # [N_seq, L]  ground-truth token ids
        mask_positions: torch.Tensor,  # [N_seq, L]  bool — positions replaced with [MASK]
        p_per_row: torch.Tensor,       # [N_seq]     masking rate per sequence
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Estimate the diffusion sequence log-probability for a batch of sequences.

        The batch spans chosen and rejected responses concatenated: the first
        N_seq//2 rows are chosen, the rest are rejected.

        Two estimators (``args.logp_type``):
          - ``"vrpo"``:  VRPO importance-weighted estimator.
                Per-token CE is debiased by dividing by the masking rate p,
                giving an unbiased estimate of the full sum of token log-probs.
          - ``"dtrpo"``: Trajectory estimator.
                Selects the ``args.unmask_size`` highest-confidence masked
                positions and scales their log-probs by block_size to estimate
                the per-step contribution to log p(y).

        If ``args.block_size > 0``, sequences are first split into
        non-overlapping blocks of length ``block_size``.  Each block is scored
        independently and the block-level estimates are summed to a seq-level
        score at the end.
        """
        p_per_row  = torch.clamp(p_per_row, min=1e-6)
        N_seq      = logits.shape[0]        # original batch size, preserved for final reshape
        block_size = self.args.block_size

        # ── Optional: reshape sequence → non-overlapping blocks ──────────────
        # After reshape every "row" represents one block of `block_size` tokens.
        # Block-level log-prob estimates are summed back to seq level at the end.
        if block_size > 0:
            logits         = logits.reshape(-1, block_size, logits.shape[-1])  # [N_blocks, B, V]
            targets        = targets.reshape(-1, block_size)                   # [N_blocks, B]
            mask_positions = mask_positions.reshape(-1, block_size)            # [N_blocks, B]
            # Recompute masking rate per block
            p_per_row = mask_positions.sum(dim=1).float() / block_size         # [N_blocks]
            p_per_row = torch.clamp(p_per_row, min=1e-6)

        N = logits.shape[0]  # N_blocks when block_size > 0, else N_seq

        # ── Log-probability estimator ─────────────────────────────────────────
        if self.args.logp_type == "vrpo":
            # Importance-weighted CE estimator.
            # For each masked position, CE / p is an unbiased estimator of
            # −log p(target_t | context) summed over all completion tokens.
            p_per_position = p_per_row.unsqueeze(1).expand(-1, logits.shape[1])  # [N, B]
            masked_p  = p_per_position[mask_positions]  # [M]  masking rate per masked token

            # Cross-entropy at masked positions only
            masked_ce = F.cross_entropy(
                logits[mask_positions].float(),   # [M, V]
                targets[mask_positions],  # [M]
                reduction="none",
            )  # [M]

            # Debias: CE / p  →  unbiased estimate of −log p(target)
            per_token_neglogp = masked_ce / masked_p  # [M]

            # Scatter back: sum debiased values per row
            row_idx = mask_positions.nonzero(as_tuple=True)[0]  # [M] row index per masked token
            per_row_neglogp = torch.zeros(N, device=logits.device, dtype=per_token_neglogp.dtype)
            per_row_neglogp.index_add_(0, row_idx, per_token_neglogp)

            logp = -per_row_neglogp  # [N]  negate: neg-log-prob → log-prob

        elif self.args.logp_type == "dtrpo":
            # Trajectory-based estimator.
            # At each diffusion step the model unmaskes `unmask_size` tokens
            # (the most confident ones).  The per-step log-prob contribution is
            # approximated as:  block_size * Σ log p(y*_t)  over unmasked positions.
            # Summing over steps gives an estimate of log p(y).
            # Upcast for scoring: bf16 log_softmax can quantize target log-probs
            # enough to change DPO log-ratios and near-tie top-k selection.
            logits_f = logits.float()
            target_token_logp = logits_f.gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1) - torch.logsumexp(logits_f, dim=-1)  # [N, B]

            k = min(max(1, self.args.unmask_size), target_token_logp.size(1))
            if kwargs.get("unmask_positions", None) is not None:
                # we have computed the unmask positions in policy prob
                # in the ref prob, we will use the same unmask positions to compute the logp
                unmask_positions = kwargs["unmask_positions"]  # [N, B]
            else:
                # Restrict scoring to masked (unknown) positions
                conf = target_token_logp.masked_fill(~mask_positions, float("-inf"))  # [N, B]

                _, topk_indices = torch.topk(conf, k, dim=1)  # [N, k]

                # Build boolean unmask mask; rows with no masked tokens (all -inf) are skipped
                valid_rows      = ~torch.all(torch.isinf(conf), dim=1)          # [N]
                topk_indices    = topk_indices * valid_rows.unsqueeze(1)
                unmask_positions = torch.zeros(N, logits.shape[1], dtype=torch.bool, device=logits.device)
                unmask_positions.scatter_(1, topk_indices, True)
                unmask_positions = unmask_positions & mask_positions  # stay within masked region

            # Per-step DTRPO contribution: block_size times the selected top-k log-probs.
            per_pos_logp = target_token_logp * unmask_positions  # [N, B]
            selected_logp = per_pos_logp.sum(dim=1)  # [N]
            block_logp = block_size * selected_logp

            if self.args.loss_type == "ipo":
                # IPO normalises by the number of unmasked tokens instead of block_size
                block_logp = selected_logp / max(k, 1)

            logp = block_logp  # [N]

        else:
            raise ValueError(
                f"Unknown logp_type: {self.args.logp_type!r}. Expected 'vrpo' or 'dtrpo'."
            )

        # ── Block → sequence aggregation ─────────────────────────────────────
        # logp is [N_blocks]; reshape to [N_seq, n_blocks] and sum along block dim.
        if block_size > 0:
            logp = logp.reshape(N_seq, -1).sum(dim=-1)  # [N_seq]

        log_dict = {
            "sum_logp_chosen":   logp[:N_seq // 2],
            "sum_logp_rejected": logp[N_seq // 2:],
            "unmask_positions": unmask_positions.detach() if self.args.logp_type == "dtrpo" else None,
        }
        return logp, log_dict

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics = {}

        model_output = self.concatenated_forward_with_random_masking(
            model,
            batch,
            mc_samples=self.args.mc_samples,
            mc_batch_size=self.args.mc_batch_size,
            mask_token_id=self.args.mask_token_id,
            cfg_scale=self.args.cfg_scale,
        )

        # Initialize combined losses
        losses = 0
        chosen_rewards = 0
        rejected_rewards = 0

        # Ensure loss_type is a list (may be a single string from DPOConfig)
        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        loss_weights = getattr(self, "loss_weights", None)

        for idx, loss_type in enumerate(loss_types):
            _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                model_output["chosen_logps_ref"],
                model_output["rejected_logps_ref"],
                loss_type,
                model_output,
            )

            # Add weighted contributions
            weight = loss_weights[idx] if loss_weights else 1.0
            losses = losses + _losses * weight
            chosen_rewards = chosen_rewards + _chosen_rewards * weight
            rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        chosen_logratios = model_output["chosen_logps"] - model_output["chosen_logps_ref"]
        rejected_logratios = model_output["rejected_logps"] - model_output["rejected_logps_ref"]

        if self.args.rpo_alpha is not None:
            losses = (
                losses + self.args.rpo_alpha * model_output["nll_loss"]
            )  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss"] = losses.detach().mean().item()
        metrics[f"{prefix}logps/chosen_ref"] = (
            self.accelerator.gather_for_metrics(
                model_output["chosen_logps_ref"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected_ref"] = (
            self.accelerator.gather_for_metrics(
                model_output["rejected_logps_ref"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(
                reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(
                chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics[f"{prefix}logratios/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_logratios)
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logratios/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_logratios)
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logratios/margins"] = (
            self.accelerator.gather_for_metrics(chosen_logratios - rejected_logratios)
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(
                model_output["mean_chosen_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(
                model_output["mean_rejected_logits"])
            .detach()
            .mean()
            .item()
        )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"])
                .detach()
                .mean()
                .item()
            )
        if "sum_logp_chosen" in model_output.keys():
            metrics[f"{prefix}sum_logp_chosen"] = (
                self.accelerator.gather_for_metrics(
                    model_output["sum_logp_chosen"])
                .detach()
                .mean()
                .item()
            )
            metrics[f"{prefix}sum_logp_rejected"] = (
                self.accelerator.gather_for_metrics(
                    model_output["sum_logp_rejected"])
                .detach()
                .mean()
                .item()
            )
        if "keep_coeff_chosen" in model_output.keys():
            metrics[f"{prefix}keep_coeff_chosen"] = (
                self.accelerator.gather_for_metrics(
                    model_output["keep_coeff_chosen"])
                .detach()
                .mean()
                .item()
            )
            metrics[f"{prefix}keep_coeff_rejected"] = (
                self.accelerator.gather_for_metrics(
                    model_output["keep_coeff_rejected"])
                .detach()
                .mean()
                .item()
            )
            metrics[f"{prefix}unmask_coeff_chosen"] = (
                self.accelerator.gather_for_metrics(
                    model_output["unmask_coeff_chosen"])
                .detach()
                .mean()
                .item()
            )
            metrics[f"{prefix}unmask_coeff_rejected"] = (
                self.accelerator.gather_for_metrics(
                    model_output["unmask_coeff_rejected"]
                )
                .detach()
                .mean()
                .item()
            )

        return losses.mean(), metrics

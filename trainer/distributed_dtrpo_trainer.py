"""Distributed trajectory-reduction preference optimization for Fast-dLLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from trainer.dpo_fast_dllm_trainer import (
    FastDLLMDPOTrainer,
    PreparedFastDLLMBatch,
)


log = logging.getLogger(__name__)


@dataclass
class DTRPOReplayRecord:
    """Detached trajectories for one MC microbatch.

    Log-probabilities have shape [2 * B, R], with chosen rows followed by
    rejected rows. Noisy inputs flatten [2 * B, R, L] to [2 * B * R, L];
    selected positions additionally flatten the sequence's block dimension.
    """

    noisy_input_ids: torch.Tensor
    unmask_positions: torch.Tensor
    policy_logps: torch.Tensor
    reference_logps: torch.Tensor


class FastDLLMDistributedDTRPOTrainer(FastDLLMDPOTrainer):
    """Distributed dTRPO through a surrogate with the same gradient.

    For preference pair i=(x_i, y_i^+, y_i^-), let s_theta[i,m,+/-] be
    the reduced trajectory log-probability estimate for MC sample m:

        s_theta[i,m,+/-] = K * sum_b sum_{t in S[i,m,b,+/-]}
            log p_theta(target_t^+/- | x_i, noisy_y[i,m,+/-])

    Here K is block_size and S contains the policy-selected unmasking
    positions in each block. The frozen reference computes s_ref using the
    same noisy inputs and positions. With B pairs and M=mc_samples, dTRPO is

        delta[i] = (1/M) * sum_m [(s_theta[i,m,+] - s_ref[i,m,+])
                               - (s_theta[i,m,-] - s_ref[i,m,-])]
        L_dtrpo = -(1/B) * sum_i log sigmoid(beta * delta[i])

    At the current parameters, define the detached coefficient

        w[i] = stop_gradient(beta * sigmoid(-beta * delta[i]))

    and the surrogate

        L_sur = (1/(B*M)) * sum_i sum_m
                    w[i] * (s_theta[i,m,-] - s_theta[i,m,+]).

    For fixed sampled states/selected positions and a frozen reference,

        grad_theta L_dtrpo
            = -(1/B) * sum_i w[i] * grad_theta delta[i]
            = grad_theta L_sur.

    The differentiable MC contributions in L_sur accumulate OUTSIDE the
    sigmoid; its nonlinearity is contained in the detached weight w[i].
    Consequently, each MC microbatch can backpropagate its additive share
    using ordinary gradient accumulation. Weights must be computed from all
    M samples of each pair first. Every microbatch, including a smaller final
    one, uses the total normalization B*M, not its own sample count.

    This class distributes preference examples through the inherited trainer;
    each rank computes all MC samples of its local pairs. It applies the
    outer gradient-accumulation normalization and calls accelerator.backward.
    Existing DDP/FSDP gradient synchronization and sharding then apply to the
    surrogate unchanged, because summing/averaging its gradients gives the
    dTRPO gradient. Training metrics report L_dtrpo, not L_sur's value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_distributed_dtrpo_config()

    def compare_trainer_and_checkpoint_args(
        self,
        training_args,
        trainer_state,
    ) -> None:
        """Keep the current logging/eval/save cadence when resuming."""
        super().compare_trainer_and_checkpoint_args(training_args, trainer_state)
        max_steps = trainer_state.max_steps or training_args.max_steps
        trainer_state.compute_steps(training_args, max_steps)
        log.info(
            "Using current resumed cadence: logging_steps=%s, eval_steps=%s, save_steps=%s",
            trainer_state.logging_steps,
            trainer_state.eval_steps,
            trainer_state.save_steps,
        )

    def _validate_distributed_dtrpo_config(self) -> None:
        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        f_divergence = getattr(self.args.f_divergence_type, "value", self.args.f_divergence_type)
        unsupported = []

        if loss_types != ["sigmoid"]:
            unsupported.append(f"loss_type={loss_types!r}")
        if self.args.logp_type != "dtrpo":
            unsupported.append(f"logp_type={self.args.logp_type!r}")
        if self.args.label_smoothing != 0.0:
            unsupported.append(f"label_smoothing={self.args.label_smoothing}")
        if f_divergence != "reverse_kl":
            unsupported.append(f"f_divergence_type={f_divergence!r}")
        if self.args.mc_batch_size < 1:
            unsupported.append(f"mc_batch_size={self.args.mc_batch_size}")
        if self.args.mc_samples < 1:
            unsupported.append(f"mc_samples={self.args.mc_samples}")
        if self.args.rpo_alpha is not None:
            unsupported.append(f"rpo_alpha={self.args.rpo_alpha}")
        if self.use_weighting:
            unsupported.append("use_weighting=True")
        if self.aux_loss_enabled:
            unsupported.append("aux_loss_enabled=True")
        if self.reference_free:
            unsupported.append("reference_free=True")
        if getattr(self.args, "sync_ref_model", False):
            unsupported.append("sync_ref_model=True")
        if not getattr(self.args, "disable_dropout", True):
            unsupported.append("disable_dropout=False")
        if self.ref_model is None:
            unsupported.append("ref_model=None")
        elif any(parameter.requires_grad for parameter in self.ref_model.parameters()):
            unsupported.append("reference model has trainable parameters")

        if unsupported:
            raise ValueError(
                "FastDLLMDistributedDTRPOTrainer only supports the exact two-pass "
                "sigmoid DTRPO configuration; unsupported settings: "
                + ", ".join(unsupported)
            )

    def _score_policy(
        self,
        model: nn.Module,
        prepared: PreparedFastDLLMBatch,
        *,
        noisy_input_ids: torch.Tensor | None = None,
        unmask_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        replay_noisy_ids = None
        if noisy_input_ids is not None:
            replay_noisy_ids = noisy_input_ids.to(
                device=prepared.input_ids.device,
                dtype=torch.long,
            )
            if unmask_positions is None:
                raise RuntimeError(
                    "DTRPO replay requires unmask_positions with precomputed noisy IDs."
                )

        # Fast-dLLM masks labels in place, so every forward must receive a fresh copy.
        labels = prepared.labels.clone()
        outputs = model(
            prepared.input_ids,
            attention_mask=prepared.attention_mask,
            labels=labels,
            compute_loss=False,
            precomputed_noisy_input_ids=replay_noisy_ids,
        )
        output_noisy_ids = outputs.noisy_input_ids
        if output_noisy_ids is None:
            raise RuntimeError(
                "Fast-dLLM DTRPO policy forward did not return noisy_input_ids; "
                f"model.training={model.training}."
            )
        output_noisy_ids = output_noisy_ids.to(
            device=prepared.input_ids.device,
            dtype=torch.long,
        )
        if replay_noisy_ids is not None and not torch.equal(
            output_noisy_ids,
            replay_noisy_ids,
        ):
            raise RuntimeError("Policy replay did not reuse noisy_input_ids exactly.")

        mask_positions = labels != -100
        targets = labels.clone()
        targets[~mask_positions] = 0
        p_per_row = mask_positions.sum(dim=1).float() / mask_positions.shape[1]
        p_per_row = torch.clamp(p_per_row, min=1e-6)

        score_kwargs = {}
        if unmask_positions is not None:
            replay_positions = unmask_positions.to(
                device=outputs.logits.device,
                dtype=torch.bool,
            )
            score_kwargs["unmask_positions"] = replay_positions

        sequence_logps, log_dict = self.compute_diffusion_logps(
            outputs.logits,
            targets.to(outputs.logits.device),
            mask_positions.to(outputs.logits.device),
            p_per_row.to(outputs.logits.device),
            **score_kwargs,
        )
        selected_positions = log_dict["unmask_positions"]
        if selected_positions is None:
            raise RuntimeError("DTRPO policy scoring did not return selected positions.")
        if unmask_positions is not None and not torch.equal(
            selected_positions,
            replay_positions,
        ):
            raise RuntimeError("Policy replay did not reuse selected positions exactly.")

        return (
            sequence_logps,
            output_noisy_ids.detach(),
            selected_positions.detach(),
        )

    def _score_reference(
        self,
        prepared: PreparedFastDLLMBatch,
        noisy_input_ids: torch.Tensor,
        unmask_positions: torch.Tensor,
    ) -> torch.Tensor:
        self.ref_model.train()
        labels = prepared.labels.clone()
        ref_outputs = self.ref_model(
            prepared.input_ids,
            attention_mask=prepared.attention_mask,
            labels=labels,
            compute_loss=False,
            precomputed_noisy_input_ids=noisy_input_ids.to(
                device=prepared.input_ids.device,
                dtype=torch.long,
            ),
        )
        if ref_outputs.noisy_input_ids is None or not torch.equal(
            ref_outputs.noisy_input_ids.to(
                device=prepared.input_ids.device,
                dtype=torch.long,
            ),
            noisy_input_ids.to(
                device=prepared.input_ids.device,
                dtype=torch.long,
            ),
        ):
            raise RuntimeError("Reference forward did not reuse policy noisy IDs exactly.")

        mask_positions = labels != -100
        targets = labels.clone()
        targets[~mask_positions] = 0
        p_per_row = mask_positions.sum(dim=1).float() / mask_positions.shape[1]
        p_per_row = torch.clamp(p_per_row, min=1e-6)

        reference_logps, _ = self.compute_diffusion_logps(
            ref_outputs.logits,
            targets.to(ref_outputs.logits.device),
            mask_positions.to(ref_outputs.logits.device),
            p_per_row.to(ref_outputs.logits.device),
            unmask_positions=unmask_positions.to(
                device=ref_outputs.logits.device,
                dtype=torch.bool,
            ),
        )
        return reference_logps

    @staticmethod
    def _repeat_prepared_batch(
        prepared: PreparedFastDLLMBatch, num_samples: int,
    ) -> PreparedFastDLLMBatch:
        """Repeat each sequence contiguously, preserving chosen/rejected order."""
        if num_samples == 1:
            return prepared
        return PreparedFastDLLMBatch(
            input_ids=prepared.input_ids.repeat_interleave(num_samples, dim=0),
            attention_mask=prepared.attention_mask.repeat_interleave(num_samples, dim=0),
            labels=prepared.labels.repeat_interleave(num_samples, dim=0),
            num_examples=prepared.num_examples * num_samples,
        )

    @torch.no_grad()
    def _collect_replay_records(
        self,
        model: nn.Module,
        prepared: PreparedFastDLLMBatch,
    ) -> list[DTRPOReplayRecord]:
        records = []
        num_sequences = prepared.input_ids.shape[0]
        for start in range(0, self.args.mc_samples, self.args.mc_batch_size):
            count = min(self.args.mc_batch_size, self.args.mc_samples - start)
            expanded = self._repeat_prepared_batch(prepared, count)
            policy_logps, noisy_ids, selected_positions = self._score_policy(
                model, expanded,
            )
            reference_logps = self._score_reference(
                expanded, noisy_ids, selected_positions,
            )
            records.append(
                DTRPOReplayRecord(
                    noisy_input_ids=noisy_ids,
                    unmask_positions=selected_positions,
                    policy_logps=policy_logps.detach().reshape(num_sequences, count),
                    reference_logps=(
                        reference_logps.detach().to(policy_logps.device)
                        .reshape(num_sequences, count)
                    ),
                )
            )
        return records

    def _compute_true_dpo_state(
        self,
        records: list[DTRPOReplayRecord],
        num_examples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        # Concatenate samples before averaging so the tail group has its true weight.
        policy_logps = torch.cat(
            [record.policy_logps for record in records], dim=1,
        ).mean(dim=1)
        reference_logps = torch.cat(
            [record.reference_logps for record in records], dim=1,
        ).mean(dim=1)

        chosen_logps = policy_logps[:num_examples]
        rejected_logps = policy_logps[num_examples:]
        chosen_ref_logps = reference_logps[:num_examples]
        rejected_ref_logps = reference_logps[num_examples:]
        model_output = {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logps_ref": chosen_ref_logps,
            "rejected_logps_ref": rejected_ref_logps,
            "sum_logp_chosen": chosen_logps,
            "sum_logp_rejected": rejected_logps,
        }
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_logps,
            rejected_logps,
            chosen_ref_logps,
            rejected_ref_logps,
            "sigmoid",
            model_output,
        )

        chosen_logratios = chosen_logps - chosen_ref_logps
        rejected_logratios = rejected_logps - rejected_ref_logps
        dpo_logits = self.beta * (chosen_logratios - rejected_logratios)
        dpo_weight = self.beta * torch.sigmoid(-dpo_logits)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        def gathered_mean(value: torch.Tensor) -> float:
            return (
                self.accelerator.gather_for_metrics(value)
                .detach()
                .float()
                .mean()
                .item()
            )

        metrics = {
            "loss": gathered_mean(losses),
            "logps/chosen_ref": gathered_mean(chosen_ref_logps),
            "logps/rejected_ref": gathered_mean(rejected_ref_logps),
            "rewards/chosen": gathered_mean(chosen_rewards),
            "rewards/rejected": gathered_mean(rejected_rewards),
            "rewards/accuracies": gathered_mean(reward_accuracies),
            "rewards/margins": gathered_mean(chosen_rewards - rejected_rewards),
            "logratios/chosen": gathered_mean(chosen_logratios),
            "logratios/rejected": gathered_mean(rejected_logratios),
            "logratios/margins": gathered_mean(
                chosen_logratios - rejected_logratios
            ),
            "logps/chosen": gathered_mean(chosen_logps),
            "logps/rejected": gathered_mean(rejected_logps),
            "logits/chosen": gathered_mean(chosen_logps),
            "logits/rejected": gathered_mean(rejected_logps),
            "sum_logp_chosen": gathered_mean(chosen_logps),
            "sum_logp_rejected": gathered_mean(rejected_logps),
            "dpo/logit": gathered_mean(dpo_logits),
            "dpo/weight": gathered_mean(dpo_weight),
        }
        return losses.mean(), dpo_weight.detach(), metrics

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del num_items_in_batch
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                prepared = self._prepare_preference_batch(inputs)
                with torch.no_grad():
                    records = self._collect_replay_records(model, prepared)
                    true_loss, dpo_weight, metrics = self._compute_true_dpo_state(
                        records,
                        prepared.num_examples,
                    )

                self.store_metrics(metrics, train_eval="train")
                gradient_accumulation = self.current_gradient_accumulation_steps

                for record in records:
                    count = record.policy_logps.shape[1]
                    expanded = self._repeat_prepared_batch(prepared, count)
                    replay_logps, _, _ = self._score_policy(
                        model,
                        expanded,
                        noisy_input_ids=record.noisy_input_ids,
                        unmask_positions=record.unmask_positions,
                    )
                    replay_logps = replay_logps.reshape(-1, count)
                    chosen_logps = replay_logps[: prepared.num_examples]
                    rejected_logps = replay_logps[prepared.num_examples :]
                    surrogate_loss = (
                        dpo_weight
                        * (rejected_logps - chosen_logps).sum(dim=1)
                        / self.args.mc_samples
                    ).mean()
                    self.accelerator.backward(
                        surrogate_loss / gradient_accumulation
                    )
                    del replay_logps, chosen_logps, rejected_logps, surrogate_loss

        return true_loss.detach() / gradient_accumulation

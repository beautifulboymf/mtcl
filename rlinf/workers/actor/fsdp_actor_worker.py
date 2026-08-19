# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from torch.utils import _pytree

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].to(Worker.torch_device_type)
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        # VLA-OPD: load a frozen teacher (separate bf16 model, no LoRA, no grad) that
        # scores the student's on-policy rollouts. Kept resident on GPU alongside the
        # (LoRA) student; only used for no_grad forward -> teacher logprobs.
        self.teacher_model = None
        if self.cfg.actor.get("use_teacher_distill", False):
            self._load_teacher_model()

        # dual-KL BASE anchor: frozen copy of the base/generalist (= student init) whose
        # broad behavior we preserve via a mode-covering forward-KL term (data-free).
        # Only loaded when the anchor is active (anchor_lambda>0), so default runs unchanged.
        self.base_model = None
        if (
            float(self.cfg.algorithm.get("anchor_lambda", 0.0)) > 0.0
            or float(self.cfg.algorithm.get("visual_anchor_lambda", 0.0)) > 0.0
            or float(self.cfg.algorithm.get("shift_beta", 0.0)) > 0.0
        ):
            self._load_base_model()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

        self._setup_rollout_weight_dst_ranks()

    def _load_teacher_model(self) -> None:
        """VLA-OPD frozen teacher(s). Two forms:

          * single  : ``actor.teacher_model_path`` (one teacher for everything).
          * routed  : ``actor.teacher_map`` = {suite: ckpt_path} (sequential CL). Teachers
            are DE-DUPLICATED by path, so mapping every suite to the one 130 generalist
            (the current setup) loads exactly ONE model. Point a suite at its own expert
            ckpt later to go true multi-teacher -- only the map changes.

        Each teacher is full (non-LoRA), eval + requires_grad_(False), resident on the
        training device, and only SCORES the student's rollouts (never acts)."""
        from copy import deepcopy

        from omegaconf import OmegaConf, open_dict

        def _load_one(path: str):
            # A teacher may be given either as a full HF dir, or as "<base_dir>::<adapter_dir>"
            # (base + PEFT LoRA adapter). The latter lets N per-suite expert teachers that all
            # sit on the SAME base be expressed as N small adapters (our spatial/goal/object
            # teachers are exactly this); get_model applies the adapter via is_lora/lora_path.
            _adapter = None
            if "::" in str(path):
                path, _adapter = str(path).split("::", 1)
            tcfg = deepcopy(self.cfg.actor.model)
            with open_dict(tcfg):
                tcfg.model_path = path
                tcfg.is_lora = _adapter is not None
                tcfg.lora_path = _adapter
                # teacher may have DIFFERENT native norm_stats than the student. Teacher
                # only SCORES (never acts), so its unnorm_key is a load-time validation
                # only. Optional override; default (None) = inherit student's key.
                _tuk = self.cfg.actor.get("teacher_unnorm_key", None)
                if _tuk:
                    tcfg.unnorm_key = _tuk
            m = get_model(tcfg)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            return m

        _tmap = self.cfg.actor.get("teacher_map", None)
        if _tmap:
            if OmegaConf.is_config(_tmap):
                _tmap = OmegaConf.to_container(_tmap, resolve=True)
            _tmap = dict(_tmap)
            # Only keep suites we actually roll out on. The stock config maps ALL FOUR suites to
            # the 130 generalist; a run over a subset would otherwise load teachers for suites
            # that never appear in a batch (wasting a full 7B each -> OOM) and would also break
            # the shared-base fast path below (one stale non-adapter entry disables it).
            try:
                _act = self.cfg.env.train.get("active_suites", None)
                if _act:
                    _act = set(OmegaConf.to_container(_act, resolve=True)
                               if OmegaConf.is_config(_act) else _act)
                    _drop = [s for s in _tmap if s not in _act]
                    if _drop and len(_act & set(_tmap)) > 0:
                        for s in _drop:
                            _tmap.pop(s)
                        self.log_info(
                            f"[VLA-OPD] teacher_map restricted to active suites "
                            f"{sorted(_act)}; dropped {sorted(_drop)}"
                        )
            except Exception:
                pass
            self.teacher_models = {}  # unique ckpt path -> model (loaded once)
            self.teacher_suite_to_path = {}  # suite name -> ckpt path (routing table)
            # SHARED-BASE FAST PATH: when every teacher is "<same base>::<adapter>", load the
            # 7B base ONCE and attach the adapters to it (PEFT multi-adapter). N experts then
            # cost 1 base + N small adapters instead of N full models -- without this, 3
            # teachers + the student = 4x7B and the run OOMs on 2 GPUs.
            _paths = list(dict.fromkeys(_tmap.values()))
            _all_adapters = all("::" in str(p) for p in _paths)
            self.teacher_adapter_of_path = None
            if _all_adapters and len(_paths) > 1:
                # GROUP BY BASE, rather than demanding a single base for all teachers. The 4-teacher
                # set has two lineages -- spatial/object/goal sit on base_stats130 while the long
                # teacher sits on the CL student's own merged model -- and the old "len(bases)==1"
                # test failed that outright, silently falling back to loading FOUR full 7B teachers
                # (the 78.5 GiB OOM). Per base we pay one 7B and attach that base's adapters, so
                # here it is 2 bases + 4 adapters instead of 4 full models.
                # Adapter names only have to be unique WITHIN one model, so each group may reuse
                # "default"; the routing code looks up teacher_models[path] first, then switches to
                # teacher_adapter_of_path[path] on THAT model.
                _by_base = {}
                for _p in _paths:
                    _by_base.setdefault(_p.split("::", 1)[0], []).append(_p)
                self.teacher_adapter_of_path = {}
                for _gi, (_b, _ps) in enumerate(_by_base.items()):
                    _shared = _load_one(_ps[0])         # base + its first adapter (name "default")
                    _pm = _shared if hasattr(_shared, "load_adapter") else getattr(_shared, "model", None)
                    self.teacher_adapter_of_path[_ps[0]] = "default"
                    for _i, _p in enumerate(_ps[1:], start=1):
                        _name = f"g{_gi}t{_i}"
                        _pm.load_adapter(_p.split("::", 1)[1], adapter_name=_name)
                        self.teacher_adapter_of_path[_p] = _name
                    for _p in _ps:
                        self.teacher_models[_p] = _shared  # one object per BASE; adapter per suite
                self.log_info(
                    f"[VLA-OPD] SHARED-BASE teachers: {len(_by_base)} base(s) + {len(_paths)} adapters "
                    f"{list(self.teacher_adapter_of_path.values())}"
                )
                for suite, path in _tmap.items():
                    self.teacher_suite_to_path[suite] = path
            else:
                for suite, path in _tmap.items():
                    self.teacher_suite_to_path[suite] = path
                    if path not in self.teacher_models:
                        self.teacher_models[path] = _load_one(path)
            # Single handle used by the OPD loss. With one generalist teacher for all
            # suites this IS that teacher. For true multi-teacher, select per suite via
            # self.teacher_suite_to_path at the teacher-forward site (OPD block).
            self.teacher_model = next(iter(self.teacher_models.values()))
            self.log_info(
                f"[VLA-OPD] teacher_map: loaded {len(self.teacher_models)} unique "
                f"teacher(s) for {len(self.teacher_suite_to_path)} suite(s): "
                f"{sorted(set(self.teacher_suite_to_path.values()))}"
            )
            # TRUE multi-teacher routing table. The batch reaching the actor carries the
            # tokenized task prompt (input_ids); each LIBERO task's language instruction is
            # unique, so prompt -> task -> suite -> teacher is an exact lookup. We key on the
            # LOWERCASED instruction text (matching how the prompt is built) and resolve at
            # forward time by decoding input_ids once per micro-batch.
            self.teacher_prompt_to_suite = None
            if len(self.teacher_models) > 1:
                try:
                    # Build prompt->suite the SAME way get_libero130_task_id_to_suite()
                    # builds task_id->suite (iterate benchmark.libero_suites -> task_maps,
                    # de-dup by task name) so the two are guaranteed consistent.
                    from libero.libero import benchmark as _lb

                    # Only the suites we actually route (the teacher_map keys). LIBERO's 130
                    # tasks have 112 unique instructions; the 2 cross-suite duplicates both
                    # involve libero_90, which is never a teacher_map key -> restricting to
                    # the mapped suites makes the prompt key EXACT for our routing.
                    _want = set(self.teacher_suite_to_path.keys())
                    _p2s, _seen, _dupe = {}, set(), 0
                    for _suite_name in getattr(_lb, "libero_suites", []):
                        if _suite_name not in _want:
                            continue
                        for _tname, _task in _lb.task_maps.get(_suite_name, {}).items():
                            if _tname in _seen:
                                continue
                            _seen.add(_tname)
                            _lang = getattr(_task, "language", None)
                            if not _lang:
                                continue
                            _k = _lang.strip().lower()
                            if _k in _p2s and _p2s[_k] != _suite_name:
                                _dupe += 1  # ambiguous across ROUTED suites -> would misroute
                            _p2s[_k] = _suite_name
                    if not _p2s:
                        raise RuntimeError("empty prompt->suite map")
                    if _dupe:
                        self.log_warning(
                            f"[VLA-OPD] {_dupe} instruction(s) are ambiguous across routed "
                            "suites; those samples may be routed to the wrong expert."
                        )
                    self.teacher_prompt_to_suite = _p2s
                    self.log_info(
                        f"[VLA-OPD] multi-teacher routing ON: {len(_p2s)} task prompts -> "
                        f"suites {sorted(set(_p2s.values()))}"
                    )
                except Exception as e:  # routing table optional; fall back to single teacher
                    self.log_warning(
                        f"[VLA-OPD] could not build prompt->suite routing table ({e}); "
                        "falling back to the FIRST teacher for all suites."
                    )
        else:
            self.teacher_models = None
            self.teacher_suite_to_path = None
            self.teacher_model = _load_one(self.cfg.actor.teacher_model_path)
            self.log_info(
                f"[VLA-OPD] loaded frozen teacher from "
                f"{self.cfg.actor.teacher_model_path}"
            )

    def _teacher_forward(self, forward_inputs, kwargs):
        """Score the student's rollout with the teacher(s).

        Single teacher (or no routing table) -> one forward, unchanged behaviour.
        TRUE multi-teacher -> split the micro-batch by the sample's SUITE (derived from its
        task instruction in ``forward_inputs['input_ids']``) and run each group through its
        own expert, then scatter the per-group outputs back into full-batch tensors. This is
        what makes "one student, N per-suite expert teachers" work: every sample is scored by
        the expert for ITS suite, never by another suite's expert.
        """
        # OPD_DUMP_STATES=<path>: save ONE micro-batch of the student's OWN on-policy rollout
        # states, then exit. Offline teacher-comparison studies otherwise have to use expert DEMO
        # states, which is the wrong distribution -- OPD's whole point is that the teacher scores
        # the states the STUDENT actually visits (and the measured teacher-student gap there was
        # far larger than on demo states). Env-gated, off by default, writes once.
        _dump = os.environ.get("OPD_DUMP_STATES", "")
        if _dump and not getattr(self, "_states_dumped", False):
            self._states_dumped = True
            try:
                torch.save(
                    {k: v.detach().cpu() for k, v in forward_inputs.items() if torch.is_tensor(v)},
                    _dump,
                )
                self.log_info(f"[VLA-OPD] dumped on-policy states -> {_dump}")
                print(f"OPD_STATES_DUMPED={_dump}", flush=True)
            except Exception as e:
                print(f"OPD_STATES_DUMP_FAILED {e}", flush=True)

        _route = getattr(self, "teacher_prompt_to_suite", None)
        _models = getattr(self, "teacher_models", None)
        if not _route or not _models or len(_models) <= 1:
            return self.teacher_model(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )

        ids = forward_inputs["input_ids"]
        bsz = ids.shape[0]
        # decode prompts once -> suite -> teacher path (unknown prompt falls back to default).
        # The embodied actor has no self.tokenizer; the (frozen) teacher model carries the
        # OFT input_processor, whose .tokenizer decodes the rollout prompts.
        _tok = getattr(self, "_route_tokenizer", None)
        if _tok is None:
            _proc = getattr(self.teacher_model, "input_processor", None)
            _tok = getattr(_proc, "tokenizer", None) if _proc is not None else None
            self._route_tokenizer = _tok
        if _tok is None:
            return self.teacher_model(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )
        texts = _tok.batch_decode(ids, skip_special_tokens=True)
        _default_path = next(iter(_models))
        groups: dict = {}
        for i, t in enumerate(texts):
            tl = t.strip().lower()
            suite = None
            for k, v in _route.items():   # instruction is a substring of the full prompt
                if k in tl:
                    suite = v
                    break
            path = self.teacher_suite_to_path.get(suite, _default_path) if suite else _default_path
            groups.setdefault(path, []).append(i)

        # Expose the routing so the OPD loss can (a) build per-suite masks and (b) CROSS-SCORE:
        # re-score suite i's states with suite j's expert. Cross-scoring is the functional-space
        # test of "do the teachers fight" -- if KL(expert_j || student) RISES on suite i's states
        # while KL(expert_i || student) falls, the student is paying for one teacher with another.
        # Note the teachers never meet in the loss itself (each scores only its own suite), so any
        # conflict has to be parameter-level interference; this measures its behavioural shadow.
        self._last_groups = groups

        if len(groups) == 1:  # whole micro-batch is one suite -> single forward
            only_path = next(iter(groups))
            _m1 = _models[only_path]
            _ad1 = getattr(self, "teacher_adapter_of_path", None)
            if _ad1:  # shared base -> must still select THIS suite's adapter
                _pm1 = _m1 if hasattr(_m1, "set_adapter") else getattr(_m1, "model", None)
                _pm1.set_adapter(_ad1[only_path])
            return _m1(
                forward_inputs=forward_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )

        out: dict = {}
        _ad_of = getattr(self, "teacher_adapter_of_path", None)
        for path, idxs in groups.items():
            sel = torch.as_tensor(idxs, device=ids.device, dtype=torch.long)
            sub_inputs = {
                k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (bsz,) else v)
                for k, v in forward_inputs.items()
            }
            _m = _models[path]
            if _ad_of:  # shared base: all paths map to ONE model -> switch the adapter
                _pm = _m if hasattr(_m, "set_adapter") else getattr(_m, "model", None)
                _pm.set_adapter(_ad_of[path])
            sub_out = _m(
                forward_inputs=sub_inputs, compute_logprobs=True,
                use_cache=False, **kwargs,
            )
            for k, v in sub_out.items():
                if not torch.is_tensor(v) or v.shape[:1] != (len(idxs),):
                    continue
                if k not in out:
                    out[k] = v.new_zeros((bsz,) + tuple(v.shape[1:]))
                out[k][sel] = v
        if not hasattr(self, "_mt_logged"):
            self._mt_logged = True
            self.log_info(
                f"[VLA-OPD] multi-teacher forward: micro-batch split across "
                f"{len(groups)} expert(s)"
            )
        return out

    @torch.no_grad()
    def _cross_score(self, forward_inputs, kwargs, ls, loss_mask, sad):
        """DIAGNOSTIC ONLY (no grad, no effect on the loss): on each suite's own rollout states,
        measure forward-KL(expert_j || student) for EVERY expert j, not just that suite's own.

        Returns {"actor/xkl_<states_suite>_by_<expert_suite>": float}. The diagonal
        (states_suite == expert_suite) is the quantity training actually minimises; the
        off-diagonal is the one nobody optimises. Reading them together over training answers
        "are the teachers fighting" in behaviour space:
          diagonal down + off-diagonal UP   -> the student buys one teacher by selling another
          both down                          -> the experts are compatible, conflict is not the story
          off-diagonal flat                  -> the experts simply live in disjoint state regions

        Requires the shared-base setup (all experts = one base + different LoRA adapters), which is
        what makes this cheap: swapping an adapter costs nothing next to loading another 7B.
        """
        out: dict = {}
        groups = getattr(self, "_last_groups", None)
        ad_of = getattr(self, "teacher_adapter_of_path", None)
        if not groups or not ad_of or len(groups) < 2:
            return out
        suite_of_path = {v: k for k, v in getattr(self, "teacher_suite_to_path", {}).items()}
        models = getattr(self, "teacher_models", None) or {}

        for st_path, idxs in groups.items():
            st_name = str(suite_of_path.get(st_path, "unk")).replace("libero_", "")
            sel = torch.as_tensor(idxs, device=ls.device, dtype=torch.long)
            sub_inputs = {
                k: (v[sel] if torch.is_tensor(v) and v.shape[0] == ls.shape[0] else v)
                for k, v in forward_inputs.items()
            }
            ls_sub = ls[sel]
            if loss_mask is not None:
                m = (
                    loss_mask[sel]
                    .to(ls.dtype)
                    .unsqueeze(-1)
                    .expand(-1, -1, sad)
                    .reshape(ls_sub.shape[0], -1)
                )
            else:
                m = torch.ones(ls_sub.shape[:2], dtype=ls.dtype, device=ls.device)
            for ex_path, ex_ad in ad_of.items():
                ex_name = str(suite_of_path.get(ex_path, "unk")).replace("libero_", "")
                mdl = models.get(ex_path, None)
                if mdl is None:
                    continue
                pm = mdl if hasattr(mdl, "set_adapter") else getattr(mdl, "model", None)
                if pm is None:
                    continue
                pm.set_adapter(ex_ad)
                o = mdl(
                    forward_inputs=sub_inputs, compute_logprobs=True, use_cache=False, **kwargs
                )
                if "action_logits" not in o:
                    continue
                # Skip when this group has NO valid positions: dividing by clamp_min(1.0) would
                # emit a literal 0.0 that _probe_emit then reports as a MEASURED zero (n=1),
                # dragging the cross-rank average down with a value that means "nothing to
                # measure". Observed on 2026-08-17 as xkl_object_by_object=0.0 at n=0.25.
                denom = m.sum()
                if denom.item() <= 0:
                    continue
                lt_x = torch.log_softmax(o["action_logits"].float(), dim=-1)
                kl = (lt_x.exp() * (lt_x - ls_sub)).sum(dim=-1)  # forward KL, per action token
                out[f"actor/xkl_{st_name}_by_{ex_name}"] = ((kl * m).sum() / denom).item()
        return out

    # ---- probe metric plumbing -------------------------------------------------------------
    # all_reduce_dict (rlinf/utils/distributed.py) packs the metric dict into ONE tensor whose
    # length is the NUMBER OF KEYS, then all_reduces it. Every rank must therefore emit the
    # IDENTICAL key set, or the collective is called with mismatched sizes and NCCL hangs until
    # the 30-minute watchdog fires. That is exactly what killed the 2026-08-17 diagnostic run:
    # the cross-scoring probe only fires on micro-batches containing >=2 suites, which is
    # data-dependent and therefore rank-dependent, so rank 1 packed a different-length tensor
    # than ranks 0/2/3 and all four deadlocked.
    #
    # Fix, made structural rather than careful: the key list is computed ONCE from teacher_map
    # (identical on every rank) and EVERY key is emitted on EVERY rank, every step. A probe that
    # did not run contributes 0.0 plus a companion "<key>__n"=0.0. Since the reduction is AVG,
    # the true mean over the ranks that measured is reduced("<key>")/reduced("<key>__n") -- the
    # 1/world_size factor cancels between the two.
    def _probe_key_list(self):
        if getattr(self, "_probe_keys", None) is not None:
            return self._probe_keys
        suites = sorted(
            str(s).replace("libero_", "")
            for s in (getattr(self, "teacher_suite_to_path", None) or {})
        )
        if not suites:
            # called before the teacher map exists -> do NOT cache an empty list, or the probe
            # keys would be permanently missing (silently, which is how the last two bugs hid)
            return []
        keys: list[str] = []
        if self.cfg.algorithm.get("cross_score", False):
            keys += [f"actor/xkl_{a}_by_{b}" for a in suites for b in suites]
        if self.cfg.algorithm.get("signal_stats", False):
            keys += [
                "actor/stu_entropy",
                "actor/tea_entropy",
                "actor/topk5_overlap",
                "actor/tea_top1_rank_in_stu",
            ]
            keys += [
                f"actor/{p}_{q}"
                for p in ("frac", "klmass")
                for q in ("hiH_hiKL", "hiH_loKL", "loH_hiKL", "loH_loKL")
            ]
        if self.cfg.algorithm.get("grad_conflict", False):
            keys += [f"actor/gnorm_{s}" for s in suites]
            keys += [
                f"actor/gcos_{suites[i]}_{suites[j]}"
                for i in range(len(suites))
                for j in range(i + 1, len(suites))
            ]
        self._probe_keys = sorted(keys)
        return self._probe_keys

    def _probe_emit(self):
        """Fixed-shape probe metrics for THIS rank. Always the same keys, on every rank."""
        measured: dict = {}
        for d in (
            getattr(self, "_last_xkl", None),
            getattr(self, "_last_sig", None),
            getattr(self, "_last_gconf", None),
        ):
            if d:
                measured.update(d)
        out: dict = {}
        for k in self._probe_key_list():
            v = measured.get(k, None)
            out[k] = float(v) if v is not None else 0.0
            out[f"{k}__n"] = 1.0 if v is not None else 0.0
        return out

    @torch.no_grad()
    def _signal_stats(self, ls, lt, kl_tok, mtok):
        """DIAGNOSTIC ONLY: is this teacher's signal even ABSORBABLE, and where does it live?

        (a) ABSORBABILITY. If the teacher's preferred action bin sits deep in the student's tail,
            the student cannot move there in reasonable steps and the whole distillation target is
            out of reach -- that would make every reweighting scheme moot, so it must be checked
            BEFORE tuning any of them.
              topk5_overlap        share of the teacher's top-5 bins that are also in the student's
              tea_top1_rank_in_stu rank of the teacher's argmax under the student (0 = same choice;
                                   large = the teacher is pointing somewhere the student ignores)

        (b) WHERE THE SIGNAL IS. Split positions by student entropy and by KL, and report both the
            share of POSITIONS and the share of total KL MASS in each quadrant. The mass share is
            the one that matters: it says which region actually drives the gradient.
              loH_hiKL = "confidently wrong" -- student is sure and disagrees with the teacher
              hiH_*    = "unsure"            -- student has no opinion yet
            Splits are at the batch median, so no threshold needs tuning.

        No extra forward pass: ls/lt/kl_tok are already computed for the loss.
        """
        out: dict = {}
        m = mtok > 0
        if m.sum() < 8:
            return out
        # ls/lt are [B, tokens, V]; kl_tok/mtok are [B, tokens]
        ps, pt = ls.exp(), lt.exp()
        H_s = -(ps * ls).sum(-1)
        H_t = -(pt * lt).sum(-1)
        out["actor/stu_entropy"] = H_s[m].mean().item()
        out["actor/tea_entropy"] = H_t[m].mean().item()

        k = 5
        t_top = lt.topk(k, dim=-1).indices
        s_top = ls.topk(k, dim=-1).indices
        inboth = (t_top.unsqueeze(-1) == s_top.unsqueeze(-2)).any(-1).float().mean(-1)
        out[f"actor/topk{k}_overlap"] = inboth[m].mean().item()

        t_arg = lt.argmax(-1, keepdim=True)
        # rank of the teacher's argmax under the student = #bins the student prefers over it
        rank = (ls > ls.gather(-1, t_arg)).sum(-1).float()
        out["actor/tea_top1_rank_in_stu"] = rank[m].mean().item()

        h, kv = H_s[m], kl_tok[m]
        hm, km = h.median(), kv.median()
        tot = kv.sum().clamp_min(1e-12)
        n = float(h.numel())
        for tag, sel in (
            ("hiH_hiKL", (h > hm) & (kv > km)),
            ("hiH_loKL", (h > hm) & (kv <= km)),
            ("loH_hiKL", (h <= hm) & (kv > km)),
            ("loH_loKL", (h <= hm) & (kv <= km)),
        ):
            out[f"actor/frac_{tag}"] = (sel.sum().item() / n) if n > 0 else 0.0
            out[f"actor/klmass_{tag}"] = (kv[sel].sum() / tot).item()
        return out

    def _grad_conflict(self, kl_tok, mtok):
        """DIAGNOSTIC ONLY: per-suite gradients of the distill loss, then pairwise cosine + norms.

        Answers two DIFFERENT questions that "the teachers fight" conflates:
          cos < 0            -> genuine directional conflict; gradient surgery (PCGrad) is justified
          cos ~ 0            -> the experts are simply orthogonal; conflict is NOT the mechanism
          |g_i| >> |g_j|     -> not conflict but DOMINANCE; a per-suite weight fixes it, and no
                                amount of gradient surgery would
        Cosine is a local first-order quantity and does NOT by itself explain the final SR gap --
        it is used here to RULE OUT mechanisms, not to prove one.

        Cost: one extra backward per suite on the probe micro-batch (retain_graph). Intended for
        the small single-GPU debug run: at world_size=1 FSDP does no gradient sharding, so the
        numbers are exact without any cross-rank reduction. On a sharded multi-GPU run these are
        LOCAL-SHARD cosines and would need an all_reduce of the dot products to be meaningful.
        """
        out: dict = {}
        groups = getattr(self, "_last_groups", None)
        if not groups or len(groups) < 2:
            return out
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return out
        suite_of_path = {v: k for k, v in getattr(self, "teacher_suite_to_path", {}).items()}
        grads: dict = {}
        for path, idxs in groups.items():
            name = str(suite_of_path.get(path, "unk")).replace("libero_", "")
            rows = torch.zeros(kl_tok.shape[0], device=kl_tok.device, dtype=kl_tok.dtype)
            rows[torch.as_tensor(idxs, device=kl_tok.device, dtype=torch.long)] = 1.0
            m_s = mtok * rows.unsqueeze(-1)
            den = m_s.sum()
            if den.item() <= 0:
                continue
            loss_s = (kl_tok * m_s).sum() / den
            g = torch.autograd.grad(
                loss_s, params, retain_graph=True, allow_unused=True
            )
            # keep per-parameter (no torch.cat) -- concatenating would add a full extra copy
            grads[name] = [None if gi is None else gi.detach().float() for gi in g]

        names = sorted(grads)
        norms = {}
        for n in names:
            sq = sum(float((gi * gi).sum()) for gi in grads[n] if gi is not None)
            norms[n] = sq**0.5
            out[f"actor/gnorm_{n}"] = norms[n]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = grads[names[i]], grads[names[j]]
                dot = sum(
                    float((x * y).sum())
                    for x, y in zip(a, b)
                    if x is not None and y is not None
                )
                den = norms[names[i]] * norms[names[j]]
                out[f"actor/gcos_{names[i]}_{names[j]}"] = dot / den if den > 0 else 0.0
        del grads
        return out

    def _load_base_model(self) -> None:
        """Dual-KL anchor: frozen BASE (= student's init / the generalist) loaded full
        (non-LoRA), eval + requires_grad_(False). Only SCORES anchor states; never acts.
        Defaults to the student's own model_path when actor.base_model_path is unset."""
        from copy import deepcopy

        from omegaconf import open_dict

        bcfg = deepcopy(self.cfg.actor.model)
        with open_dict(bcfg):
            bcfg.model_path = self.cfg.actor.get(
                "base_model_path", self.cfg.actor.model.model_path
            )
            bcfg.is_lora = False
            bcfg.lora_path = None
            _buk = self.cfg.actor.get("base_unnorm_key", None)
            if _buk:
                bcfg.unnorm_key = _buk
        self.base_model = get_model(bcfg)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.log_info(f"[dual-KL] loaded frozen BASE anchor from {bcfg.model_path}")

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    async def sync_model_to_rollout(self) -> None:
        if not self._weight_dst_rank_in_rollout:
            self.log_debug(
                f"Actor rank {self._rank} has no rollout weight-sync destination."
            )
            if self.enable_offload:
                if not self.is_optimizer_offloaded:
                    self.offload_optimizer()
                if not self.is_weight_offloaded:
                    self.offload_param_and_grad(True)
            return

        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            handle = []
            for rank in self._weight_dst_rank_in_rollout:
                handle.append(
                    self.send(
                        data,
                        dst_group_name=self._rollout_group_name,
                        dst_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            await asyncio.gather(*handle)

        async def recv_func():
            handle = []
            for rank in self._weight_dst_rank_in_rollout:
                handle.append(
                    self.recv(
                        src_group_name=self._rollout_group_name,
                        src_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            metadata_list = await asyncio.gather(*handle)
            metadata = metadata_list[0]
            for other_metadata in metadata_list[1:]:
                if other_metadata != metadata:
                    raise ValueError("Patch metadata differs across rollout ranks")
            return metadata

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
            )

        await self.weight_syncer.sync(state_dict, send_func, version=self.version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # FAILURE-ONLY distillation support (algorithm.distill_on_failure).
        # "traj_fail" = 1 where the RETURN-TO-GO of this trajectory is zero, i.e. from this state
        # onward the rollout never succeeded again. Computed HERE because this is the only place
        # the data still carries its trajectory structure -- the shape is documented above as
        # [n_chunk_step, rollout_epoch x bsz, num_action_chunks], so a trajectory is a fixed index
        # in dim 1 running along dim 0. Doing it later (in the OPD loss, on a micro-batch) is what
        # broke the first attempt: there each row is a single 8-action chunk, LIBERO's reward is
        # ~1 only at the success instant, so "this row earned no reward" was true for 98% of rows
        # regardless of whether its episode succeeded -- it just deleted the 2% of chunks that
        # carried the success, the exact opposite of the intent (measured fail_frac=0.977 against
        # success_once=0.736).
        # env.train has auto_reset=False and ignore_terminations=False, so one trajectory slot
        # holds exactly ONE episode and a plain reverse cumsum needs no per-done segment reset.
        # Arm the cross-scoring probe ONCE per step. _process_received_rollout_batch runs exactly
        # once per training step, whereas the loss body runs once per micro-batch (dozens of times
        # a step) -- gating on "first micro-batch" there would fire once per global batch, not once
        # per step. A latch set here and cleared by the first micro-batch is the cheap correct gate.
        if self.cfg.algorithm.get("cross_score", False):
            self._xscore_armed = True
        if self.cfg.algorithm.get("grad_conflict", False):
            self._gconf_armed = True
        if self.cfg.algorithm.get("signal_stats", False):
            self._sig_armed = True

        if self.cfg.algorithm.get("distill_on_failure", False):
            with torch.no_grad():
                rw = rollout_batch["rewards"]  # [T, B, C]
                T, B, C = rw.shape
                # -> [B, T*C] laid out in trajectory time order, reverse-cumsum, back to [T, B, C]
                r = rw.transpose(0, 1).reshape(B, T * C)
                rtg = r.flip(-1).cumsum(-1).flip(-1)
                fail = (rtg <= 0).to(rw.dtype)
                rollout_batch["traj_fail"] = (
                    fail.reshape(B, T, C).transpose(0, 1).contiguous()
                )

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            # VLA-OPD: the advantage is the per-token reverse-KL reward
            # r_t = log pi_teacher(a_t) - log pi_student(a_t), computed PER MICRO-BATCH inside
            # run_training (the teacher forward needs the flattened/processed batch + matching
            # forward_inputs). Here we only place a same-shape placeholder so
            # process_nested_dict_for_train and the training loop find an "advantages" tensor.
            self.rollout_batch["advantages"] = torch.zeros_like(
                self.rollout_batch["prev_logprobs"]
            )
            return compute_rollout_metrics(self.rollout_batch)

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "opd_center": self.cfg.algorithm.get("opd_center", False),
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # NOTE: This must be set before importing openpi.training.data_loader
            if self.cfg.actor.get("sft_data_path", None):
                os.environ["HF_LEROBOT_HOME"] = self.cfg.actor.sft_data_path

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ):
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: x.to(self.device) if x is not None else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        sft_losses = self.model(
            data={"observation": observation, "actions": actions},
            forward_type=ForwardType.SFT,
        )
        # Ensure losses is a tensor and handle different return types
        if isinstance(sft_losses, list | tuple):
            sft_losses = torch.stack(sft_losses)
        elif not isinstance(sft_losses, torch.Tensor):
            sft_losses = torch.tensor(
                sft_losses, device=self.device, dtype=torch.float32
            )

        sft_loss = sft_losses.mean()
        metrics_data["sft_loss"] = sft_loss.clone().detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)
                    opd_kl = None   # KL(student||teacher) diagnostic, filled in the opd block
                    opd_gap = None  # mean(teacher_lp - student_rollout_lp) on executed actions
                    opd_distill_loss = None  # differentiable KL-distill loss (opd_mode=distill)
                    anchor_loss = None  # dual-KL BASE anchor (forward-KL to frozen base)
                    visual_loss = None  # visual-representation anchor (cosine to base mid-layer)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                        # request 256-bin action logits for the OPD student-teacher KL diagnostic
                        kwargs["return_action_logits"] = (
                            self.cfg.algorithm.adv_type == "opd"
                        )
                        # visual-representation anchor: request mid-layer vision+prompt features
                        if float(self.cfg.algorithm.get("visual_anchor_lambda", 0.0)) > 0.0:
                            kwargs["return_mid_features"] = True
                            kwargs["mid_layer"] = int(
                                self.cfg.algorithm.get("visual_anchor_layer", 16)
                            )
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                            **kwargs,
                        )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        prev_logprobs = output_dict["prev_logprobs"]

                    if self.cfg.algorithm.adv_type == "opd":
                        # VLA-OPD: frozen teacher scores the SAME actions the student executed
                        # (forward_inputs holds the rollout action tokens); the reverse-KL
                        # log-ratio is the advantage (detached -> constant reward).
                        # Sequential CL uses ONE generalist teacher for all suites, so
                        # self.teacher_model is used directly here. For TRUE multi-teacher
                        # (per-suite experts) route with self.teacher_suite_to_path using
                        # the sample's suite -- the rest of this block is unchanged.
                        with torch.no_grad(), self.amp_context:
                            teacher_out = self._teacher_forward(
                                forward_inputs, kwargs
                            )
                        t_lp = teacher_out["logprobs"].detach()
                        # per-token reverse-KL -> aggregate to per-chunk (num_action_chunks) so it
                        # matches RLinf's action-granularity advantages. logprobs are
                        # [B, num_action_chunks * action_dim]; the loss preprocessing reduces
                        # logprobs over action_dim, and expects advantages already at [B, num_action_chunks].
                        rkl = (t_lp - prev_logprobs).detach()  # [B, chunks*action_dim]
                        sad = self.cfg.actor.model.get("action_dim", 7)
                        # MEAN over the action_dim (not sum) -> per-chunk RKL, avoids one outlier
                        # token dominating the whole chunk's advantage.
                        adv = rkl.reshape(rkl.shape[0], -1, sad).mean(dim=-1)  # [B, chunks]
                        # STANDARDIZE the advantage (masked) — the EmbodiedFSDPActor path has no
                        # normalize_advantages, so raw RKL gave grad_norm 200-670 -> divergence.
                        # Bring it to ~O(1) so PPO updates are stable.
                        if self.cfg.algorithm.get("normalize_advantages", False):
                            m = (loss_mask.to(adv.dtype) if loss_mask is not None
                                 else torch.ones_like(adv))
                            cnt = m.sum().clamp_min(1.0)
                            mean = (adv * m).sum() / cnt
                            var = (((adv - mean) ** 2) * m).sum() / cnt
                            adv = ((adv - mean) / (var.sqrt() + 1e-6)) * m
                        advantages = adv
                        opd_gap = rkl.mean().item()  # mean(log pi_tea - log pi_stu) on executed actions
                        # DIFFERENTIABLE on-policy distillation (pi0 op_distill analog): directly
                        # minimize KL(student||teacher) over the 256 action bins on the student's
                        # rollout states. Teacher detached (frozen); student side carries gradient.
                        # This realizes OPD's reverse-KL objective as a DIFFERENTIABLE loss (GKD-style),
                        # NOT the REINFORCE/advantage route (which wouldn't converge here).
                        if "action_logits" in output_dict and "action_logits" in teacher_out:
                            ls = torch.log_softmax(output_dict["action_logits"].float(), dim=-1)
                            lt = torch.log_softmax(
                                teacher_out["action_logits"].float().detach(), dim=-1
                            )
                            # Base-Centered Policy-Shift MOPD (BCF-MOPD): distill toward
                            #   q ∝ π_0 · exp((log π_tea − log π_0)/β)  i.e.
                            #   log q = log_softmax( lb + (lt − lb)/β )
                            # instead of the full teacher lt. Only the teacher's SHIFT relative
                            # to base is transferred, base-anchored: β>1 → q between base and
                            # expert (preserves base generalization); β=1 → q=teacher (vanilla).
                            # ONE coherent target -> no dual-KL anchor conflict.
                            _sbeta = float(self.cfg.algorithm.get("shift_beta", 0.0))
                            if _sbeta > 0.0 and getattr(self, "base_model", None) is not None:
                                with torch.no_grad(), self.amp_context:
                                    _shift_base_out = self.base_model(
                                        forward_inputs=forward_inputs,
                                        compute_logprobs=True,
                                        use_cache=False,
                                        **kwargs,
                                    )
                                if "action_logits" in _shift_base_out:
                                    _lb_s = torch.log_softmax(
                                        _shift_base_out["action_logits"].float().detach(),
                                        dim=-1,
                                    )
                                    lt = torch.log_softmax(
                                        _lb_s + (lt - _lb_s) / _sbeta, dim=-1
                                    )
                            # diagnostic: reverse KL(student||teacher) (comparable across runs)
                            opd_kl = (ls.exp() * (ls - lt)).sum(dim=-1).detach().mean().item()
                            # LOSS direction (distill_kl):
                            #   forward  = KL(teacher||student), teacher-weighted, MODE-COVERING
                            #     (student covers teacher's good actions without deleting its own ->
                            #      FORGETS LESS; matches pi0 velocity-MSE that worked).
                            #   reverse  = KL(student||teacher), MODE-SEEKING (zero-forces student
                            #      onto teacher's OOD flatness -> catastrophic forgetting; what failed).
                            _dkl = self.cfg.algorithm.get("distill_kl", "forward")
                            if _dkl == "reverse":
                                kl_tok = (ls.exp() * (ls - lt)).sum(dim=-1)
                            elif _dkl == "jsd":
                                # Generalized JSD (GKD, verified recommendation): beta->0 = forward
                                # (mode-covering), beta->1 = reverse; bounded by log2 so NO off-support
                                # blow-up, and closed-form so NO dropped state-visitation bias / no
                                # REINFORCE variance. Small beta = the weak-student sweet spot.
                                beta = float(self.cfg.algorithm.get("jsd_beta", 0.3))
                                pt = lt.exp()
                                ps = ls.exp()
                                m = (beta * pt + (1.0 - beta) * ps).clamp_min(1e-8)
                                lm = m.log()
                                kl_tok = (
                                    beta * (pt * (lt - lm)).sum(dim=-1)
                                    + (1.0 - beta) * (ps * (ls - lm)).sum(dim=-1)
                                )
                            else:
                                pt = lt.exp()  # teacher probs (detached)
                                kl_tok = (pt * (lt - ls)).sum(dim=-1)  # forward KL, grad via ls
                            # CONFIDENCE FILTER (kit): down-weight tokens where the TEACHER itself is
                            # uncertain (high entropy = OOD / off-support state) so we don't distill the
                            # teacher's garbage on the weak student's own drifted states.
                            _conf_tau = float(self.cfg.algorithm.get("distill_conf_tau", 0.0))
                            if _conf_tau > 0.0:
                                with torch.no_grad():
                                    pt_d = lt.exp()
                                    t_ent = -(pt_d * lt).sum(dim=-1)  # teacher entropy per token
                                    ent_max = torch.log(
                                        torch.tensor(float(pt_d.shape[-1]), device=pt_d.device)
                                    )
                                    conf_w = (1.0 - t_ent / ent_max).clamp_min(0.0).pow(_conf_tau)
                                kl_tok = kl_tok * conf_w
                            # FAILURE-ONLY distillation (distill_on_failure=True): only distill where
                            # the student's own rollout FAILED. Where it already succeeds, the
                            # teacher's disagreement is style, not substance -- we measured that
                            # three models which ALL solve object still disagree on 25-42% of
                            # actions, i.e. as much as the student-teacher gap itself, so copying it
                            # wastes the shared LoRA capacity and is what makes several per-suite
                            # experts fight each other.
                            # The flag is "traj_fail", precomputed in _process_received_rollout_batch
                            # where the trajectory structure still exists (see the comment there for
                            # why computing it from this micro-batch's rewards is WRONG).
                            _fail_only = bool(
                                self.cfg.algorithm.get("distill_on_failure", False)
                            )
                            fail_m = None
                            if _fail_only:
                                fail_m = batch.get("traj_fail", None)
                                if fail_m is None:
                                    raise RuntimeError(
                                        "distill_on_failure=True but 'traj_fail' is missing from the "
                                        "batch -- it must be built in _process_received_rollout_batch; "
                                        "refusing to silently fall back to distilling everything."
                                    )
                                fail_m = fail_m.to(kl_tok.dtype)
                            if loss_mask is not None:
                                # loss_mask is per-chunk [B, chunks]; expand to per-token to mask kl_tok
                                mtok = (
                                    loss_mask.to(kl_tok.dtype)
                                    .unsqueeze(-1)
                                    .expand(-1, -1, sad)
                                    .reshape(kl_tok.shape[0], -1)
                                )
                            else:
                                mtok = torch.ones_like(kl_tok)
                            if fail_m is not None:
                                # traj_fail is per-chunk [B, chunks] like loss_mask -> expand the same way
                                fm = (
                                    fail_m.unsqueeze(-1)
                                    .expand(-1, -1, sad)
                                    .reshape(kl_tok.shape[0], -1)
                                )
                                mtok = mtok * fm
                                # fraction of the VALID (loss_mask'd) positions we actually distil on.
                                # SELF-CHECK: this must land near (1 - success_rate), NOT ~0.98.
                                with torch.no_grad():
                                    _base = (
                                        loss_mask.to(kl_tok.dtype)
                                        .unsqueeze(-1)
                                        .expand(-1, -1, sad)
                                        .reshape(kl_tok.shape[0], -1)
                                        if loss_mask is not None
                                        else torch.ones_like(kl_tok)
                                    )
                                    self._last_fail_frac = (
                                        mtok.sum() / _base.sum().clamp_min(1.0)
                                    ).item()
                            opd_distill_loss = (kl_tok * mtok).sum() / mtok.sum().clamp_min(1.0)

                            # How many suites this micro-batch actually contains. Both probes are
                            # meaningless on a single-suite micro-batch (there is no other expert
                            # to compare against), and whether the data mixes suites at all is an
                            # empirical question about the rollout/shuffle pipeline -- so MEASURE
                            # it instead of assuming. If this sits at 1.0 the probes are silently
                            # inert and the routing/group_size has to change first.
                            _ngrp = len(getattr(self, "_last_groups", {}) or {})
                            self._last_nsuites = float(_ngrp)

                            # absorbability / signal-location probe. NOT gated on multi-suite:
                            # it asks about ONE teacher-student pair, so a single-suite micro-batch
                            # is perfectly valid input.
                            if getattr(self, "_sig_armed", False):
                                self._sig_armed = False
                                try:
                                    self._last_sig = self._signal_stats(
                                        ls.detach(), lt, kl_tok.detach(), mtok
                                    )
                                except Exception as _e:
                                    self._last_sig = {}
                                    self.log_warning(f"[VLA-OPD] signal_stats failed: {_e}")

                            # Probes run on the first MIXED micro-batch of the step. The latch is
                            # NOT cleared on a single-suite batch: clearing it there would spend
                            # the step's one probe on a batch that can produce nothing.
                            if getattr(self, "_xscore_armed", False) and _ngrp >= 2:
                                try:
                                    self._last_xkl = self._cross_score(
                                        forward_inputs, kwargs, ls.detach(), loss_mask, sad
                                    )
                                    self._xscore_armed = False
                                except Exception as _e:  # never let a probe kill a training run
                                    self._last_xkl = {}
                                    self._xscore_armed = False
                                    self.log_warning(f"[VLA-OPD] cross_score failed: {_e}")

                            # per-suite gradient conflict probe: first mixed micro-batch of the
                            # step. MUST run before the real backward frees the graph.
                            if getattr(self, "_gconf_armed", False) and _ngrp >= 2:
                                try:
                                    self._last_gconf = self._grad_conflict(kl_tok, mtok)
                                    self._gconf_armed = False
                                except Exception as _e:
                                    self._last_gconf = {}
                                    self._gconf_armed = False
                                    self.log_warning(f"[VLA-OPD] grad_conflict failed: {_e}")

                        # ---- data-free BASE anchors (action-KL + visual-representation) ----
                        # (a) action anchor = mode-covering forward-KL to base on rollout states
                        #     (preserve task behavior); (b) visual anchor = cosine of mid-layer
                        #     vision+prompt features to base (preserve BROAD OOD generalization).
                        _alam = float(self.cfg.algorithm.get("anchor_lambda", 0.0))
                        _vlam = float(self.cfg.algorithm.get("visual_anchor_lambda", 0.0))
                        if (_alam > 0.0 or _vlam > 0.0) and getattr(
                            self, "base_model", None
                        ) is not None:
                            with torch.no_grad(), self.amp_context:
                                base_out = self.base_model(
                                    forward_inputs=forward_inputs,
                                    compute_logprobs=True,
                                    use_cache=False,
                                    **kwargs,
                                )
                            # (a) ACTION anchor
                            if (
                                _alam > 0.0
                                and "action_logits" in output_dict
                                and "action_logits" in base_out
                            ):
                                lb = torch.log_softmax(
                                    base_out["action_logits"].float().detach(), dim=-1
                                )
                                ls_a = torch.log_softmax(
                                    output_dict["action_logits"].float(), dim=-1
                                )
                                pb = lb.exp()
                                a_tok = (pb * (lb - ls_a)).sum(dim=-1)  # forward-KL, mode-covering
                                _agate = self.cfg.algorithm.get("anchor_gate", "none")
                                _atau = float(self.cfg.algorithm.get("anchor_gate_tau", 1.0))
                                if _agate in ("low_ent", "high_ent"):
                                    with torch.no_grad():
                                        b_ent = -(pb * lb).sum(dim=-1)
                                        _emax = torch.log(
                                            torch.tensor(float(pb.shape[-1]), device=pb.device)
                                        )
                                        conf = (1.0 - b_ent / _emax).clamp(0.0, 1.0)
                                        gate = (
                                            conf.pow(_atau)
                                            if _agate == "low_ent"
                                            else (1.0 - conf).pow(_atau)
                                        )
                                    a_tok = a_tok * gate
                                if loss_mask is not None:
                                    _amt = (
                                        loss_mask.to(a_tok.dtype)
                                        .unsqueeze(-1)
                                        .expand(-1, -1, sad)
                                        .reshape(a_tok.shape[0], -1)
                                    )
                                    anchor_loss = (a_tok * _amt).sum() / _amt.sum().clamp_min(1.0)
                                else:
                                    anchor_loss = a_tok.mean()
                            # (b) VISUAL anchor: patch-wise cosine of mid-layer features to base
                            if (
                                _vlam > 0.0
                                and "mid_features" in output_dict
                                and "mid_features" in base_out
                            ):
                                fs = output_dict["mid_features"].float()
                                fb = base_out["mid_features"].float().detach()
                                cos = torch.nn.functional.cosine_similarity(fs, fb, dim=-1)
                                visual_loss = (1.0 - cos).mean()

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    if (
                        self.cfg.algorithm.adv_type == "opd"
                        and self.cfg.algorithm.get("opd_mode", "distill") == "distill"
                        and opd_distill_loss is not None
                    ):
                        # differentiable KL-distillation: minimize KL(student||teacher) directly,
                        # skip the PPO/REINFORCE loss entirely (pi0 op_distill style).
                        loss = opd_distill_loss
                        if anchor_loss is not None:
                            loss = loss + float(
                                self.cfg.algorithm.get("anchor_lambda", 0.0)
                            ) * anchor_loss
                        if visual_loss is not None:
                            loss = loss + float(
                                self.cfg.algorithm.get("visual_anchor_lambda", 0.0)
                            ) * visual_loss
                        metrics_data = {
                            "actor/distill_loss": opd_distill_loss.detach().item()
                        }
                        if getattr(self, "_last_fail_frac", None) is not None:
                            # share of rollout samples that never succeeded = what we distill on
                            metrics_data["actor/fail_frac"] = self._last_fail_frac
                        if anchor_loss is not None:
                            metrics_data["actor/anchor_loss"] = anchor_loss.detach().item()
                        if visual_loss is not None:
                            metrics_data["actor/visual_loss"] = visual_loss.detach().item()
                        # cross-suite KL probe (diagonal = what training minimises, off-diagonal =
                        # what nobody optimises); emitted on the probe micro-batch only, so it is
                        # carried on self and re-emitted for the rest of the step's micro-batches.
                        # ALWAYS emit, ALWAYS the same keys -- see _probe_key_list for why a
                        # rank-dependent key set deadlocks the metric all_reduce.
                        # 1.0 => micro-batches are single-suite => the cross-suite probes are inert
                        metrics_data["actor/n_suites_in_batch"] = float(
                            getattr(self, "_last_nsuites", 0.0) or 0.0
                        )
                        metrics_data.update(self._probe_emit())
                    else:
                        loss, metrics_data = policy_loss(**kwargs)

                    if opd_kl is not None:
                        metrics_data["actor/opd_kl_stu_tea"] = opd_kl
                    if opd_gap is not None:
                        metrics_data["actor/opd_gap_raw"] = opd_gap

                    entropy_loss = torch.tensor(
                        0.0, device=Worker.torch_platform.current_device()
                    )
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

                    if self.enable_sft_co_train:
                        self._train_sft_epoch(metrics_data, loss)

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

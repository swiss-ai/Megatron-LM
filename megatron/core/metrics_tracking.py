from typing import Iterator, Optional

import torch

from megatron.core import parallel_state


class Tracker:
    known_metrics = {"mean", "rms", "kurtosis", "underflow", "overflow"}
    expected_names = {"activation": (0, "sp"),
                      "mlp_intermediate": (1, "tp"),
                      "mlp_out": (2, "none"),
                      "qkv": (3, "tp"),
                      "mlp_post_act": (4, "tp"),
                      "qkv_input": (5, "tp"),
                      "q": (6, "tp"),
                      "k": (7, "tp")}
    _inv_expected_names = {idx: name for (name, (idx, _)) in expected_names.items()}

    def __init__(self, args, metrics: Optional[list[str]] = None):
        if metrics is None:
            metrics = []
        assert set(metrics) <= self.known_metrics
        self.enabled = False
        self.metrics = metrics
        self.args = args
        if args is None:
            return
        self.intermediate_metrics_map = {}
        idx = 0
        if "mean" in self.metrics:
            self.intermediate_metrics_map["sum"] = idx
            idx += 1
        if "rms" in self.metrics or "kurtosis" in self.metrics:
            self.intermediate_metrics_map["norm_squared"] = idx
            idx += 1
        if "kurtosis" in self.metrics:
            self.intermediate_metrics_map["sum_of_4th_powers"] = idx
            idx += 1
        if "underflow" in self.metrics:
            self.intermediate_metrics_map["underflow"] = idx
            idx += 1
        if "overflow" in self.metrics:
            self.intermediate_metrics_map["overflow"] = idx
            idx += 1

        self.final_metrics_map = {metric: idx for idx, metric in enumerate(self.metrics)}
        self._inv_final_metrics_map = {idx: metric for metric, idx in self.final_metrics_map.items()}

        self.local_gbs = self.args.global_batch_size//self.args.data_parallel_size
        self.local_layers = self.args.num_layers//self.args.pipeline_model_parallel_size

        self._init_pp()
        self.reset()

    def reset(self):
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        shape = self.local_gbs, self.local_layers, len(self.intermediate_metrics_map), len(self.expected_names)

        self.current_mbs = 0 if vp_size is None else [0 for _ in range(self.local_layers)]
        self.partial_results = torch.full(shape, torch.inf, dtype=torch.float32, device="cuda")
        self._shapes = torch.full((len(self.expected_names),), -1, dtype=torch.int64, device="cuda")
        self.gathered_final_metrics = None

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    @torch.no_grad()
    def update(self, x: torch.Tensor, name: str, layer: int):
        # x.shape = [seq, mbs, ...].
        if len(self.metrics) == 0 or not self.enabled:
            return
        assert self.gathered_final_metrics is None, "Call tracker.reset() before doing tracker.update() after an aggregation"

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        mbs = x.size(1)
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        x = x.transpose(0, 1).reshape(mbs, -1)
        name_idx = self.expected_names[name][0]
        assert self._shapes[name_idx] == -1 or self._shapes[name_idx] == x.size(1)

        if self._vp_map is None:  # i.e. no virtual pp.
            assert 0 <= layer - self.local_first_layer < pp_size*self.local_layers, f"({self.local_first_layer}, {layer}, {self.local_layers})"
            local_layer = layer - self.local_first_layer
            current_mbs = self.current_mbs
        else:
            assert layer + 1 in self._inv_vp_map
            local_layer = self._inv_vp_map[layer + 1]
            current_mbs = self.current_mbs[local_layer]

        assert (current_mbs + 1)*mbs <= self.local_gbs
        self._shapes[name_idx] = x.size(1)
        batch_slice = slice(current_mbs*mbs, (current_mbs + 1)*mbs)

        if "mean" in self.metrics:
            idx = (batch_slice, local_layer, self.intermediate_metrics_map["sum"], name_idx)
            self.partial_results[idx] = torch.sum(x, dim=1).float()
        if "rms" in self.metrics or "kurtosis" in self.metrics:
            idx = (batch_slice, local_layer, self.intermediate_metrics_map["norm_squared"], name_idx)
            x_squared = x**2
            self.partial_results[idx] = torch.sum(x_squared, dim=1).float()
            if "kurtosis" in self.metrics:
                idx = (batch_slice, local_layer, self.intermediate_metrics_map["sum_of_4th_powers"], name_idx)
                self.partial_results[idx] = torch.sum(x_squared**2, dim=1).float()
        if "underflow" in self.metrics or "overflow" in self.metrics:
            absx = torch.abs(x)
            if "underflow" in self.metrics:
                idx = (batch_slice, local_layer, self.intermediate_metrics_map["underflow"], name_idx)
                amin = torch.min(absx)
                self.partial_results[idx] = torch.count_nonzero(absx == amin)
            if "overflow" in self.metrics:
                idx = (batch_slice, local_layer, self.intermediate_metrics_map["overflow"], name_idx)
                amax = torch.amax(absx)
                self.partial_results[idx] = torch.count_nonzero(absx == amax)

        if self._vp_map is None:
            if torch.all(~torch.isinf(self.partial_results[batch_slice, :, :, :])):
                self.current_mbs += 1
        else:
            if torch.all(~torch.isinf(self.partial_results[batch_slice, local_layer, :, :])):
                self.current_mbs[local_layer] += 1

    @torch.no_grad()
    def aggregate(self):
        if len(self.metrics) == 0:
            return
        assert self.enabled, "Can't aggregate metrics disabled"
        assert self.gathered_final_metrics is None, "Tracker has already aggregated metrics"
        assert torch.all(self._shapes > 1)

        # Now we want to aggregate across DP,TP,PP; special attention needed if sequence_parallel is enabled.
        # All metrics can be aggregated simply by taking the sum across TP (optional) and DP.
        # If any metric has shard_style = "sp" and sequence_parallel is not enabled, no TP aggregation is needed.
        # Finally, all ranks send the last rank its metrics, so last rank has all metrics for all layers.
        assert torch.all(~torch.isinf(self.partial_results))
        indices_of_sp = [idx for (idx, parallel_type) in self.expected_names.values() if parallel_type == "sp"]
        indices_of_tp = [idx for (idx, parallel_type) in self.expected_names.values() if parallel_type == "tp"]
        indices_of_np = [idx for (idx, parallel_type) in self.expected_names.values() if parallel_type == "none"]
        sp_metrics = self.partial_results[:, :, :, indices_of_sp]
        tp_metrics = self.partial_results[:, :, :, indices_of_tp]
        name_parallel_factor = torch.full((len(self.expected_names),), -1, dtype=torch.int64, device="cuda")

        # TP all_reduce.
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_size = torch.distributed.get_world_size(tp_group)
        name_parallel_factor[indices_of_tp] = tp_size
        name_parallel_factor[indices_of_np] = 1
        if tp_size > 1:
            torch.distributed.all_reduce(tp_metrics, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            if self.args.sequence_parallel:
                torch.distributed.all_reduce(sp_metrics, op=torch.distributed.ReduceOp.SUM, group=tp_group)
                name_parallel_factor[indices_of_sp] = tp_size
            else:
                name_parallel_factor[indices_of_sp] = 1
        else:
            name_parallel_factor[indices_of_sp] = 1
        self.partial_results[:, :, :, indices_of_sp] = sp_metrics
        self.partial_results[:, :, :, indices_of_tp] = tp_metrics
        assert torch.all(name_parallel_factor >= 1)

        # Now that all possible non-linearities when computing the final metrics are gone, we can start computing them.
        # Reminder of partial_results shape = [local_gbs, local_layers, len(intermediate_metrics_map), len(expected_names)]
        shape = self.local_layers, len(self.final_metrics_map), len(self.expected_names)
        final_metrics = torch.empty(shape, dtype=torch.float32, device="cuda")
        if "mean" in self.metrics:
            sum_id = self.intermediate_metrics_map["sum"]
            mean_id = self.final_metrics_map["mean"]
            mean = torch.mean(self.partial_results[:, :, sum_id, :]/name_parallel_factor/self._shapes, dim=0)
            final_metrics[:, mean_id, :] = mean
        if "rms" in self.metrics or "kurtosis" in self.metrics:
            norm_squared_id = self.intermediate_metrics_map["norm_squared"]
            norm_squared = self.partial_results[:, :, norm_squared_id, :]
            norm_id = self.final_metrics_map["rms"]
            final_metrics[:, norm_id, :] = torch.mean(torch.sqrt(norm_squared/name_parallel_factor/self._shapes), dim=0)
            if "kurtosis" in self.metrics:
                sum_of_4th_power_id = self.intermediate_metrics_map["sum_of_4th_powers"]
                sum_of_4th_power = self.partial_results[:, :, sum_of_4th_power_id, :]
                mean_of_4th_power = sum_of_4th_power/name_parallel_factor/self._shapes
                mean_of_2nd_power = norm_squared/name_parallel_factor/self._shapes
                kurtosis = mean_of_4th_power/mean_of_2nd_power**2
                kurtosis_id = self.final_metrics_map["kurtosis"]
                final_metrics[:, kurtosis_id, :] = torch.mean(kurtosis, dim=0)
        if "underflow" in self.metrics:
            underflow_intermediate_id = self.intermediate_metrics_map["underflow"]
            underflow_final_id = self.final_metrics_map["underflow"]
            underflow = torch.mean(self.partial_results[:, :, underflow_intermediate_id, :]/name_parallel_factor/self._shapes, dim=0)
            final_metrics[:, underflow_final_id, :] = underflow
        if "overflow" in self.metrics:
            overflow_intermediate_id = self.intermediate_metrics_map["overflow"]
            overflow_final_id = self.final_metrics_map["overflow"]
            overflow = torch.mean(self.partial_results[:, :, overflow_intermediate_id, :]/name_parallel_factor/self._shapes, dim=0)
            final_metrics[:, overflow_final_id, :] = overflow

        # DP all_reduce.
        dp_group = parallel_state.get_data_parallel_group()
        dp_size = torch.distributed.get_world_size(dp_group)
        if dp_size > 1:
            torch.distributed.all_reduce(final_metrics, op=torch.distributed.ReduceOp.AVG, group=dp_group)

        # PP all_gather.
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_size = torch.distributed.get_world_size(pp_group)
        if pp_size > 1:
            shape = pp_size*self.local_layers, len(self.final_metrics_map), len(self.expected_names)
            gathered_final_metrics = torch.empty(shape, dtype=torch.float32, device="cuda")
            torch.distributed.all_gather_into_tensor(gathered_final_metrics, final_metrics, group=pp_group)
            if self._vp_map is not None:  # i.e. virtual pp is enabled, let's fix the order.
                vp_map = torch.tensor(list(self._vp_map.values()), dtype=torch.float32, device="cuda")
                gathered_vp_map = torch.empty(pp_size*self.local_layers, dtype=torch.float32, device="cuda")
                torch.distributed.all_gather_into_tensor(gathered_vp_map, vp_map, group=pp_group)
                # Let's fix_vp_ordering.
                gathered_vp_map_list = (gathered_vp_map - 1).long().cpu().tolist()
                assert all(0 <= idx < self.args.num_layers for idx in gathered_vp_map_list)
                assert len(gathered_vp_map_list) == len(set(gathered_vp_map_list))  # i.e. no repeated layers.
                gathered_final_metrics = gathered_final_metrics[gathered_vp_map_list]
        else:
            gathered_final_metrics = final_metrics
        self.gathered_final_metrics = gathered_final_metrics

    def get_final_metrics(self) -> Iterator[tuple[str, float]]:
        if len(self.metrics) == 0:
            return iter([])
        assert self.gathered_final_metrics is not None
        avg_metrics_across_layers = torch.mean(self.gathered_final_metrics, dim=0).tolist()
        values = self.gathered_final_metrics.tolist()
        # Give only the averages first so it looks better in wandb :3
        for name_id in range(self.gathered_final_metrics.size(2)):
            name = self._inv_expected_names[name_id]
            for metric_id in range(self.gathered_final_metrics.size(1)):
                metric = self._inv_final_metrics_map[metric_id]
                yield f"{metric}/{name}_avg", avg_metrics_across_layers[metric_id][name_id]
                for layer in range(self.gathered_final_metrics.size(0)):
                    yield f"{metric}/{name}_layer{layer:03d}", values[layer][metric_id][name_id]

    def _init_pp(self):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        if vp_size is None:  # No Virtual Pipelines.
            self.local_first_layer = self.local_layers*pp_rank
            self._vp_map = None
            return

        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        from megatron.training.arguments import core_transformer_config_from_args
        layers_per_vp = self.args.num_layers_per_virtual_pipeline_stage
        config = core_transformer_config_from_args(self.args)
        self._vp_map = {}
        for local_idx in range(self.local_layers):
            vp_stage = local_idx//layers_per_vp
            layer_in_vp = local_idx % layers_per_vp
            self._vp_map[local_idx] = layer_in_vp + get_transformer_layer_offset(config, vp_stage) + 1
        self._inv_vp_map = {layer: local_idx for local_idx, layer in self._vp_map.items()}


_TRACKER: Optional[Tracker] = None


def init_tracker(args, metrics: list[str]):
    global _TRACKER
    assert _TRACKER is None, "tracker already initialized"
    _TRACKER = Tracker(args, metrics=metrics)


def get_tracker() -> Tracker:
    global _TRACKER
    assert _TRACKER is not None, "tracker not initialized"
    return _TRACKER

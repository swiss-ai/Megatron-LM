# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Object-load correctness and collective coverage for fully parallel checkpoints."""

from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies import fully_parallel
from megatron.core.dist_checkpointing.utils import _sharded_object_id, _sharded_tensor_shard_id
from megatron.core.msc_utils import MultiStorageClientFeature
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class StoredValuesLoadStrategy:
    """A local storage boundary that rejects missing checkpoint entries."""

    def __init__(self, values):
        self.values = values
        self.requests = []

    def load(self, sharded_state_dict, checkpoint_dir):
        self.requests.append(set(sharded_state_dict))
        missing = set(sharded_state_dict) - self.values.keys()
        if missing:
            raise CheckpointingException(f'Missing object in checkpoint: {missing}')
        return {key: self.values[key] for key in sharded_state_dict}


class TestFullyParallelObjectLoad:
    @pytest.fixture
    def planned_load(self, monkeypatch):
        # Keep real object deferral/reassembly; replace distributed planning and
        # tensor transport so the object-load contract also runs without GPUs.
        monkeypatch.setattr(fully_parallel, 'get_pg_size', lambda group: 2)
        monkeypatch.setattr(
            fully_parallel.FullyParallelLoadStrategyWrapper,
            'apply_loading_parallelization',
            lambda self, state: object(),
        )
        monkeypatch.setattr(
            fully_parallel, 'exchange_by_distribution', lambda loaded, *args: loaded
        )
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    @staticmethod
    def state_and_values(with_tensor):
        main = ShardedObject('extra', None, (2,), (0,), replica_id=0)
        remote = ShardedObject('extra', None, (2,), (1,), replica_id=1)
        # Multiple local references to one stored object must restore identically.
        alias = ShardedObject('extra', None, (2,), (0,), replica_id=1)
        state = {'model': {'main': main, 'remote': remote, 'alias': alias}}
        values = {_sharded_object_id(main): {'rank': 0}, _sharded_object_id(remote): {'rank': 1}}
        if with_tensor:
            weight = ShardedTensor.from_rank_offsets('weight', torch.zeros(3))
            state['model']['weight'] = weight
            values[_sharded_tensor_shard_id(weight)] = torch.tensor([2.0, 4.0, 6.0])
        return state, values

    @pytest.mark.parametrize('with_tensor', [False, True])
    def test_per_rank_load_reads_all_objects_without_exchange(self, planned_load, with_tensor):
        state, values = self.state_and_values(with_tensor)
        base = StoredValuesLoadStrategy(values)
        strategy = fully_parallel.FullyParallelLoadStrategyWrapper(base, per_rank_object_load=True)
        with mock.patch.object(
            fully_parallel,
            'exchange_loaded_objects_gather_object',
            side_effect=AssertionError('Per-rank objects must not be exchanged'),
        ):
            loaded = strategy.load(state, None)['model']

        assert loaded['main'] == {'rank': 0}
        assert loaded['remote'] == {'rank': 1}
        assert loaded['alias'] == loaded['main']
        if with_tensor:
            torch.testing.assert_close(loaded['weight'], torch.tensor([2.0, 4.0, 6.0]))
        # All object replicas and local tensors share a single storage plan.
        assert base.requests == [set(values)]

    def test_default_load_exchanges_only_main_replicas(self, planned_load):
        state, values = self.state_and_values(with_tensor=True)
        base = StoredValuesLoadStrategy(values)
        strategy = fully_parallel.FullyParallelLoadStrategyWrapper(base)
        with mock.patch.object(
            fully_parallel,
            'exchange_loaded_objects_gather_object',
            return_value={('extra', (0,), (2,)): {'rank': 0}, ('extra', (1,), (2,)): {'rank': 1}},
        ) as exchange:
            loaded = strategy.load(state, None)['model']

        assert loaded['main'] == {'rank': 0}
        assert loaded['remote'] == {'rank': 1}
        assert loaded['alias'] == loaded['main']
        torch.testing.assert_close(loaded['weight'], torch.tensor([2.0, 4.0, 6.0]))
        assert base.requests == [{('extra', (0,), (2,))}, {('weight', (0,), None)}]
        exchange.assert_called_once_with({('extra', (0,), (2,)): {'rank': 0}})

    def test_default_load_rejects_missing_remote_objects(self, planned_load):
        state, values = self.state_and_values(with_tensor=False)
        strategy = fully_parallel.FullyParallelLoadStrategyWrapper(StoredValuesLoadStrategy(values))
        with mock.patch.object(
            fully_parallel, 'exchange_loaded_objects_gather_object', return_value={}
        ):
            with pytest.raises(CheckpointingException, match='Missing object shards'):
                strategy.load(state, None)

    def test_per_rank_load_propagates_missing_storage_object(self, planned_load):
        state, values = self.state_and_values(with_tensor=False)
        del values[('extra', (1,), (2,))]
        strategy = fully_parallel.FullyParallelLoadStrategyWrapper(
            StoredValuesLoadStrategy(values), per_rank_object_load=True
        )
        with pytest.raises(CheckpointingException, match='Missing object in checkpoint'):
            strategy.load(state, None)


class TestFullyParallelObjectRoundTrip:
    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('per_rank_object_load', [False, True])
    def test_mixed_tensor_object_round_trip(self, tmp_path, per_rank_object_load):
        if Utils.world_size < 2:
            pytest.skip('Requires at least two distributed ranks')
        Utils.initialize_model_parallel(1, 1)
        # Share rank 0's path without depending on the global asset-download
        # fixtures, so this test can also run with --noconftest.
        checkpoint_paths = [str(tmp_path / 'checkpoint') if Utils.rank == 0 else None]
        torch.distributed.broadcast_object_list(checkpoint_paths)
        state = {
            'model': {
                'weight': ShardedTensor.from_rank_offsets(
                    'weight', torch.tensor([2.0, 4.0, 6.0]), replica_id=Utils.rank
                ),
                'objects': [
                    ShardedObject(
                        'objects',
                        {'owner': i},
                        (Utils.world_size,),
                        (i,),
                        replica_id=abs(Utils.rank - i),
                    )
                    for i in range(Utils.world_size)
                ],
            }
        }
        with (
            mock.patch.object(MultiStorageClientFeature, 'is_enabled', return_value=False),
            TempNamedDir(Path(checkpoint_paths[0])) as checkpoint_dir,
        ):
            save(deepcopy(state), checkpoint_dir)
            strategy = fully_parallel.FullyParallelLoadStrategyWrapper(
                get_default_load_sharded_strategy(checkpoint_dir),
                per_rank_object_load=per_rank_object_load,
            )
            with mock.patch.object(
                fully_parallel,
                'exchange_loaded_objects_gather_object',
                wraps=fully_parallel.exchange_loaded_objects_gather_object,
            ) as exchange:
                loaded = load(deepcopy(state), checkpoint_dir, strategy)
            assert exchange.call_count == int(not per_rank_object_load)
            torch.testing.assert_close(loaded['model']['weight'], torch.tensor([2.0, 4.0, 6.0]))
            assert loaded['model']['objects'] == [{'owner': i} for i in range(Utils.world_size)]

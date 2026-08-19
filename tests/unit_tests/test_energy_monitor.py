# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.energy_monitor import EnergyMonitor


def test_energy_monitor_is_noop_until_setup():
    monitor = EnergyMonitor()

    monitor.pause()
    monitor.resume()

    assert monitor.lap() == 0.0
    assert monitor.get_total() == 0.0

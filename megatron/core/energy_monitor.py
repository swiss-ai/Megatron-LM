# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron Energy Monitoring (NVML)"""

import torch
import torch.distributed as dist

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetTotalEnergyConsumption,
        nvmlInit,
        nvmlShutdown,
    )

    has_nvml = True
except ImportError:
    has_nvml = False


class EnergyMonitor:
    """
    Energy monitoring using NVML.

    All ranks in the process group are expected to call functions lap() and get_total().
    Energy is monitored across all ranks and aggregated with an all-reduce.
    """

    def __init__(self) -> None:
        """Initialize EnergyMonitor."""
        self._total_energy = 0
        self._lap_energy = 0
        self._last_energy = 0
        self._handle = None
        self._active = False

    def _is_active(self) -> bool:
        """Return whether NVML has been initialized for this monitor."""
        return has_nvml and self._active and self._handle is not None

    def setup(self) -> None:
        """Setup the NVML Handler."""
        if has_nvml:
            nvmlInit()
            self._handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self._active = True

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._is_active():
            nvmlShutdown()
            self._handle = None
            self._active = False

    def pause(self) -> None:
        """Pause energy monitor (must resume afterward)."""
        if self._is_active():
            energy = self._get_energy()
            self._lap_energy += energy - self._last_energy

    def resume(self) -> None:
        """Resume/start energy monitor."""
        if self._is_active():
            self._last_energy = self._get_energy()

    def _get_energy(self) -> int:
        """Get current energy consumption from NVML."""
        if not self._is_active():
            return self._last_energy
        try:
            return nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except NVMLError:
            return self._last_energy  # return *something* if it errors

    def lap(self) -> float:
        """Returns lap (iteration) energy (J) and updates total energy."""
        if not self._is_active():
            return 0.0

        energy = self._get_energy()
        lap_energy = self._lap_energy + (energy - self._last_energy)

        self._total_energy += lap_energy
        self._lap_energy = 0
        self._last_energy = energy

        lap_tensor = torch.tensor([lap_energy], dtype=torch.int64, device='cuda')
        dist.all_reduce(lap_tensor, op=dist.ReduceOp.SUM)

        return lap_tensor.item() / 1000.0

    def get_total(self) -> float:
        """Get total energy consumption (J) across all GPUs."""
        if not self._is_active():
            return 0.0

        energy_tensor = torch.tensor([self._total_energy], dtype=torch.int64, device='cuda')
        dist.all_reduce(energy_tensor, op=dist.ReduceOp.SUM)

        return energy_tensor.item() / 1000.0

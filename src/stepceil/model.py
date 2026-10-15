from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceOp:
    """One kernel, memcpy, or memset on the GPU timeline."""

    kind: str
    name: str
    start_ns: int
    end_ns: int
    stream_id: int | None
    device_id: int | None
    context_id: int | None = None
    peer_device_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SyncOp:
    """One host synchronization call."""

    name: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class StepWindow:
    """The single NVTX range and the CUDA records that meet it."""

    name: str
    start_ns: int
    end_ns: int
    device_ops: tuple[DeviceOp, ...]
    syncs: tuple[SyncOp, ...]

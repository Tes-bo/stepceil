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
    byte_count: int | None = None
    process_id: int | None = None
    # Full kernel symbol. Template instances can share one short name.
    symbol: str | None = None


@dataclass(frozen=True)
class SyncOp:
    """One host synchronization call."""

    name: str
    start_ns: int
    end_ns: int
    process_id: int | None = None


@dataclass(frozen=True)
class RuntimeOp:
    """One host CUDA API call. It is not device work."""

    name: str
    start_ns: int
    end_ns: int
    process_id: int | None = None


@dataclass(frozen=True)
class NvtxOp:
    """One NVTX range other than the step itself. It is not device work."""

    name: str
    start_ns: int
    end_ns: int
    process_id: int | None = None


@dataclass(frozen=True)
class StepWindow:
    """The single NVTX range and the CUDA records that meet it."""

    name: str
    start_ns: int
    end_ns: int
    device_ops: tuple[DeviceOp, ...]
    syncs: tuple[SyncOp, ...]
    runtime_ops: tuple[RuntimeOp, ...] = ()
    nvtx_ops: tuple[NvtxOp, ...] = ()

from typing import Final

from overrides import final
from overrides import override as override
from torch import device, dtype
from typing_extensions import TypeAlias

finaloverride = final

Device: TypeAlias = device

DataType: TypeAlias = dtype

CPU: Final = Device("cpu")

GPU: Final = Device("cuda")

META: Final = Device("meta")

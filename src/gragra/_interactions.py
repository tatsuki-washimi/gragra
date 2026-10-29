"""Physical interaction identity shared by model specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Protocol


class Interaction(Protocol):
    """Identify a physical law using a name and immutable scalar parameters."""

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class Newtonian:
    """The parameter-free Newtonian gravitational interaction."""

    name: ClassVar[str] = "newton"
    parameters: ClassVar[Mapping[str, object]] = MappingProxyType({})

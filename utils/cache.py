from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar


T = TypeVar("T")


@dataclass
class SimpleCache(Generic[T]):
    store: dict[Hashable, T] = field(default_factory=dict)

    def get(self, key: Hashable) -> T | None:
        return self.store.get(key)

    def set(self, key: Hashable, value: T) -> T:
        self.store[key] = value
        return value

    def clear(self) -> None:
        self.store.clear()


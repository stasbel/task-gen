import abc
from typing import List

import validator
from lazy import lazy

text_validator = validator.Text()
tag_validator = validator.Tag()


class Entity(abc.ABC):
    """Entity class."""

    @abc.abstractmethod
    def is_valid(self) -> bool:
        """Check if entity valid (in some sense)."""
        pass

    @abc.abstractmethod
    def norm(self) -> str:
        """Normalizing entity."""
        pass


class Text(Entity):
    """Text entity."""

    def __init__(self, raw: str):
        self.raw = raw

    @lazy
    def is_valid(self):
        return text_validator(self.raw) and self.norm

    @lazy
    def norm(self) -> str:
        return self.raw.strip().replace('\n', ' DCNL ').strip()


class Tag(Entity):
    """Tag entity."""

    def __init__(self, raw: str):
        self.raw = raw

    @lazy
    def is_valid(self):
        return tag_validator(self.raw) and self.norm

    @lazy
    def norm(self):
        return '-'.join(self.raw.strip().lower().split())


ENTITIES = [
    ('text', Text),
    ('tags', Tags)
]


def collect(raw):
    keys = set(n for n, _ in ENTITIES)
    data = {k: [] for k in keys}
    for i_raw in raw:
        pass

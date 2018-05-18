import abc
import re
from typing import Optional, List

import validator
from lazy import lazy
from sklearn.base import BaseEstimator, TransformerMixin

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
        """Normalizing entity into str value."""
        pass


class Text(Entity):
    """Text entity."""

    def __init__(self, raw: str):
        self.raw = raw

    @lazy
    def is_valid(self):
        return bool(text_validator(self.raw) and self.norm)

    _RES = [
        (re.compile(r'[^\x00-\x7F]+'), ' NASC '),
        (re.compile('\n\n'), '\n'),
        (re.compile('\n'), ' DCNL '),
        (re.compile(r'\s\s+'), ' ')
    ]

    @lazy
    def norm(self):
        text = self.raw
        for p, s in self._RES:
            text = p.sub(s, text).strip()
        return text


class Tag(Entity):
    """Tag entity."""

    def __init__(self, raw: str):
        self.raw = raw

    @lazy
    def is_valid(self):
        return bool(tag_validator(self.raw) and self.norm)

    @lazy
    def norm(self):
        return '-'.join(self.raw.strip().lower().split())


def linearize_tags(raw_tags: List[str]) -> Optional[str]:
    """Liniarize tags into str ot `None` if all tags are bad."""
    tags = [Tag(raw_tag) for raw_tag in raw_tags]
    sorted_tags = sorted(tag.norm for tag in tags if tag.is_valid)
    return ' '.join(sorted_tags) or None

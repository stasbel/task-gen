import abc
import re
from typing import List

import pandas as pd
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
        return bool(text_validator(self.raw) and self.norm)

    @lazy
    def norm(self) -> str:
        text = re.sub('\n\n', '\n', self.raw)
        return text.strip().replace('\n', ' DCNL ').strip()


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


def collect(raw):
    def linearize_tags(raw_tags):
        tags = [Tag(raw_tag) for raw_tag in raw_tags]
        stags = sorted(tag.norm for tag in tags if tag.is_valid)
        return ' '.join(stags) or None

    data = {'text': [], 'tags': []}

    for i_raw in raw:
        if 'text' not in i_raw:
            continue

        text = Text(i_raw['text'])
        if not text.is_valid:
            continue

        lin_tags = linearize_tags(i_raw['tags'] if 'tags' in i_raw else [])

        data['text'].append(text.norm)
        data['tags'].append(lin_tags)

    return pd.DataFrame.from_dict(data)[['text', 'tags']].fillna('NaN')

import abc
from string import ascii_lowercase
from typing import List, Sequence

import spacy
from langdetect import detect as detect_lang
from spacy_cld import LanguageDetector


class Validator(abc.ABC):
    """Simple functor, performing check of some sort."""

    @abc.abstractmethod
    def __call__(self, x) -> bool:
        """Check if `x` valid (in some sense)."""
        pass


# Base `Validator`s

class LenValidator(Validator):
    """Length boundaries `Validator`."""

    def __init__(self, bounds: tuple):
        lb, up = bounds
        assert 0 <= lb <= up, "Sanity check"

        self.lb, self.up = lb, up

    def __call__(self, seq: Sequence) -> bool:
        return self.lb <= len(seq) <= self.up


class LanguageValidator(Validator):
    """Sensible text in language `Validator`."""

    def __init__(self, language: str = 'en'):
        assert language == 'en', "Only support en for now"

        self.language = language
        self.nlp = spacy.load(language)
        self.nlp.add_pipe(LanguageDetector())

    def __call__(self, text: str) -> bool:
        return self._is_infer_lang_spacy_cld(text) \
               and self._is_infer_langdetect(text)

    def _is_infer_lang_spacy_cld(self, text):
        languages = self.nlp(text)._.languages
        return len(languages) and languages[0] == self.language

    def _is_infer_langdetect(self, text):
        try:
            return detect_lang(text) == self.language
        except:
            return False


ALPHAS = set(ascii_lowercase + ascii_lowercase.upper())


class CharsValidator(Validator):
    """Good chars `Validator`."""

    GOOD_CHARS = ALPHAS

    def __init__(self, arate: float = 0.75):
        assert 0 <= arate <= 1, "Sanity check"

        self.arate = arate

    def __call__(self, text: str) -> bool:
        return self._is_ascii(text) and self._is_good_arate(text)

    def _is_ascii(self, text):
        return all(31 < ord(c) < 128 for c in text)

    def _is_good_arate(self, text):
        m = sum(bool(c in self.GOOD_CHARS) for c in text)
        return m / len(text) >= self.arate


class ConstructorValidator(Validator):
    """Simple sequence of `Validator`s."""

    def __init__(self, validators: List[Validator] = None):
        self.validators = validators

    def __call__(self, x) -> bool:
        return all(validator(x) for validator in self.validators)


class AsciiValidator(Validator):
    """Simple ascii-only chars `Validator`."""

    GOOD_CHARS = ALPHAS | {' '}

    def __call__(self, text: str) -> bool:
        return all(c in self.GOOD_CHARS for c in text)


class NoStopwordsValidator(Validator):
    """Simple no stopwords `Validator`."""

    def __init__(self, language: str = 'en'):
        assert language == 'en', "Only support en for now"

        self.nlp = spacy.load(language)

    def __call__(self, text: str) -> bool:
        return all(not self.nlp.vocab[word].is_stop for word in text.split())


class SubstringsValidator(Validator):
    """Simple no specific substrings `Validator`."""

    def __init__(self, exs: List[str] = None):
        self.exs = exs

    def __call__(self, text: str) -> bool:
        return all(ex not in text for ex in self.exs)


# Complex `Validator`s

class TextValidator(ConstructorValidator):
    """Sensible and good text `Validator`."""

    def __init__(self, bounds=(25, 3000), language='en', arate=0.75):
        super().__init__([
            LenValidator(bounds),
            LanguageValidator(language),
            CharsValidator(arate)
        ])


class TagsValidator(ConstructorValidator):
    """Sensible and good tags `Validator`."""

    EXS = ['dcnl', 'doesn\\\'t', 'it\\', 'user\\', 'return', ':return']

    def __init__(self, bounds=(2, 24), arate=0.75,
                 language='en', exs=EXS):
        super().__init__([
            LenValidator(bounds),
            CharsValidator(arate),
            AsciiValidator(),
            NoStopwordsValidator(language),
            SubstringsValidator(exs)
        ])

    def __call__(self, tags: List[str]) -> bool:
        method = super().__call__
        return all(method(i_tags) for i_tags in tags)

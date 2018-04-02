from string import ascii_lowercase

import spacy
from langdetect import detect as detect_lang
from spacy_cld import LanguageDetector

ALPHAS = ascii_lowercase + ascii_lowercase.upper()


class Validator:
    """Simple functor, perfoeming plausible text check."""

    def __init__(self, language='en', len_b=(25, 3000), arate=0.75):
        assert language == 'en', "Only support en for now"
        assert 0 < arate <= 1, ""

        self.language = language
        self.nlp = spacy.load(language)
        self.nlp.add_pipe(LanguageDetector())
        self.len_lb, self.len_up = len_b
        self.arate = arate

        self.checks = [
            self._len_check,
            self._language_check,
            self._chars_check
        ]

    def __call__(self, word):
        return all(check(word) for check in self.checks)

    def _len_check(self, text):
        return self.len_lb <= len(text) <= self.len_up

    def _language_check(self, text):
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

    def _chars_check(self, text):
        return self._is_ascii(text) and self._is_good_arate(text)

    def _is_ascii(self, text):
        return all(31 < ord(c) < 128 for c in text)

    def _is_good_arate(self, text):
        m = sum(bool(c in ALPHAS) for c in text)
        return m / len(text) >= self.arate

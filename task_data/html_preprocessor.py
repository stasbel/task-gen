import copy
import re

from bs4 import BeautifulSoup
from utils import monitor_apply_gen
from sklearn.base import BaseEstimator, TransformerMixin


class HTMLPurifier:
    def __init__(self, delete_code=True, code_replacer=' CODE ',
                 delete_r=True, treshold=0.9):
        self.delete_code = delete_code
        self.code_replacer = code_replacer
        self.delete_r = delete_r
        self.treshold = treshold

    def __call__(self, raw_text):
        text = self._replace_unescape(raw_text)
        self._soup = BeautifulSoup(text, 'lxml')

        self._codes = []
        new_soup = copy.copy(self._soup)
        self._change_code(new_soup)

        text = str(new_soup.get_text())
        text = text.replace('\r', '') if self.delete_r else text

        return text

    @staticmethod
    def _replace_unescape(text,
                          unescape_dict=None):
        if unescape_dict is None:
            unescape_dict = {'&lt;': '<', '&gt;': '>', '&amp;': '&'}

        def round_(text):
            for k, v in unescape_dict.items():
                text = text.replace(k, v)
            return text

        old_text, text = text, round_(text)
        while old_text != text:
            old_text, text = text, round_(text)

        return text

    def _change_code(self, tag):
        for child in tag.findChildren():
            if child.name == 'code':
                code = child.text
                self._codes.append(code)
                if self.delete_code: child.replaceWith(self.code_replacer)


class HTMLPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, monitor=None, do_tqdm=False):
        self.monitor = monitor
        self.do_tqdm = do_tqdm

        self.purifier = HTMLPurifier()

    def fit(self, X):
        return self

    def transform(self, X):
        return list(monitor_apply_gen(self.purifier, X,
                                      self.monitor, self.do_tqdm))

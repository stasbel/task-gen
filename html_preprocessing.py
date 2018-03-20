import re
import copy
import warnings
import pandas as pd

from tqdm import tqdm
from lazy import lazy
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from IPython.core.display import HTML
from sklearn.base import BaseEstimator, TransformerMixin


# Switch warning off for `bs4`.
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


class TextExtractor:
    def __init__(self, raw_text, 
                 delete_code=True, code_replacer='CODE',
                 delete_r=True, treshold=0.9, nl_replacer='DCNL'):
        self.raw_text = raw_text
        self.delete_code = delete_code
        self.code_replacer = code_replacer
        self.delete_r = delete_r
        self.treshold = treshold
        self.nl_replacer = nl_replacer

    @lazy
    def text(self):
        text = self._replace_unescape(self.raw_text)
        self._soup = BeautifulSoup(text, 'lxml')

        self._codes = []
        new_soup = copy.copy(self._soup)
        self._change_code(new_soup)

        text = str(new_soup.get_text())
        text = text.replace('\r', '') if self.delete_r else text
        
        text = text.strip().replace('\n', f' {self.nl_replacer} ')
        return text
    
    @staticmethod
    def _replace_unescape(text, 
                          unescape_dict={'&lt;': '<', '&gt;': '>', '&amp;': '&'}):
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
    def __init__(self):
        pass
    
    def fit(self, X):
        return self

    def transform(self, X):
        return [TextExtractor(x).text for x in tqdm(X)]

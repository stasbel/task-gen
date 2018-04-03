import abc

import nltk
from lazy import lazy
from nltk.corpus import stopwords
from rake_nltk import Rake

STOP_WORDS = {
    'doesn\\\'t',
    'it\\',
    'user\\',
    'return',
    ':return'
}


class Extractor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, text):
        """Extract list of entities from `text`."""
        pass


class NLTKExtractor(Extractor):
    """TODO: Not working due to versioning errors.

    Reference: https://gist.github.com/alexbowe/879414
    """

    SENTENCE_RE = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:$?\d+(' \
                  r'?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'

    GRAMMAR_RE = r'''
        NBAR:
            {<NN.*|JJ>*<NN.*>}

        NP:
            {<NBAR><IN><NBAR>}
            {<NBAR>}
    '''

    def __init__(self):
        self.chuncker = nltk.RegexpParser(GRAMMAR_RE)
        self.stopwords = stopwords.words('english')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def __call__(self, text):
        print(SENTENCE_RE)
        toks = nltk.regexp_tokenize(text, SENTENCE_RE)
        postoks = nltk.tag.pos_tag(toks)
        tree = self.chuncker.parse(postoks)
        terms = self._get_terms(tree)

    def _get_terms(self, tree):
        for leaf in self._leaves(tree):
            term = [self._normalise(w)
                    for w, t in leaf if self._acceptable_word(w)]
            yield term

    def _leaves(self, tree):
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            yield subtree.leaves()

    def _normalize(self, word):
        word = word.lower()
        word = self.stemmer.stem_word(word)
        word = self.lemmatizer.lemmatize(word)
        return word

    def _acceptable_word(self, word):
        accepted = bool(2 <= len(word) <= 40
                        and word.lower() not in stopwords)
        return accepted


class Tags:
    def __init__(self, raw_tags):
        self.raw_tags = raw_tags

    @lazy
    def is_valid(self):
        def is_valid(tag):
            pass

    @lazy
    def norm(self):
        def norm(tag):
            return '-'.join(tag.lower().split())

        return ' '.join(sorted(norm(tag) for tag in self.raw_tags))


class RakeExtractor(Extractor):
    def __init__(self):
        self.r = Rake()

    def __call__(self, text):
        return self.r.get_ranked_phrases(text)

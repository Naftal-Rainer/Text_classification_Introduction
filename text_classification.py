import nltk 
import random 
from sklearn.feature_extraction.text import CountVectorizer


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer_count = CountVectorizer()
features = vectorizer_count.fit_transform(corpus)
vectorizer_count.get_feature_names_out()


print(features.toarray())

vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()

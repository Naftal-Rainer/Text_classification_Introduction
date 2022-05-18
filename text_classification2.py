import nltk 
import random 
from sklearn.feature_extraction.text import CountVectorizer


sentences = ['We are using the Bag of Word Model',' Bag of word model is used for extracting the features']

# Building a Bag of Words

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences).todense()

# Print the feature vectors which can be used for machine learning
print(vectorizer.vocabulary_)

X = vectorizer.fit_transform(sentences)
vectorizer.get_feature_names_out()
print(X.toarray())

import os
import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer

# Set the path to the data directory
data_dir = 'aclImdb/train'

# Read in the vocabulary and sentiment polarity scores
with open('aclImdb/imdb.vocab', 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f.readlines()]
with open('aclImdb/ImdbEr.txt', 'r', encoding='utf-8') as f:
    scores = np.array([float(line.strip()) for line in f.readlines()])

# Load the data
reviews = []
labels = []
for label in ['pos', 'neg', 'unsup']:
    path = os.path.join(data_dir, label)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            review = f.read()
        reviews.append(review)
        if label in ['pos', 'neg']:
            labels.append(label)

# Preprocess the data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', vocabulary=vocab)
X = vectorizer.fit_transform(reviews)

# Compute the sentiment-weighted word embeddings
num_topics = 10
lda_model = lda.LDA(n_topics=num_topics, n_iter=500, random_state=1)
lda_model.fit(X)
topic_word = lda_model.topic_word_
word_embeddings = np.zeros((len(vocab), num_topics))
for i, word in enumerate(vocab):
    word_embeddings[i] = scores[i] * topic_word[:, vectorizer.vocabulary_[word]]

# Print the word embeddings for the first 10 words
print(word_embeddings[:10])

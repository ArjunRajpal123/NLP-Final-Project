# Import necessary libraries
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
nltk.download('stopwords')
import lda

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt



root_path = './aclImdb'
vocab_path = os.path.join(root_path, 'imdb.vocab')
train_path = os.path.join(root_path, 'train')
test_path = os.path.join(root_path, 'test')




with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = f.read().strip().split('\n')
    
train_pos = []
train_neg = []
for filename in os.listdir(os.path.join(train_path, 'pos')):
    with open(os.path.join(train_path, 'pos', filename), 'r', encoding='utf-8') as f:
        train_pos.append(f.read())
for filename in os.listdir(os.path.join(train_path, 'neg')):
    with open(os.path.join(train_path, 'neg', filename), 'r', encoding='utf-8') as f:
        train_neg.append(f.read())
        
test_pos = []
test_neg = []
for filename in os.listdir(os.path.join(test_path, 'pos')):
    with open(os.path.join(test_path, 'pos', filename), 'r', encoding='utf-8') as f:
        test_pos.append(f.read())
for filename in os.listdir(os.path.join(test_path, 'neg')):
    with open(os.path.join(test_path, 'neg', filename), 'r', encoding='utf-8') as f:
        test_neg.append(f.read())

train_text = train_pos + train_neg
train_labels = np.concatenate((np.ones(len(train_pos)), np.zeros(len(train_neg))))
test_text = test_pos + test_neg
test_labels = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))


stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t.isalpha()]
    return ' '.join(tokens)

train_text = [preprocess_text(t) for t in train_text]
test_text = [preprocess_text(t) for t in test_text]


X_train, X_val, y_train, y_val = train_test_split(train_text, train_labels, test_size=0.2, random_state=42)


vectorizer = CountVectorizer(vocabulary=vocab)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(test_text)


lda1 = LatentDirichletAllocation(n_components=10, random_state=42)
lda2 = lda.LDA(n_topics=10, n_iter=1500, random_state=1)
lda2.fit(X_train_vec)
print('finished Latent DirichletAllocation')


train_topic_distributions = lda2.fit_transform(X_train_vec)
val_topic_distributions = lda2.fit_transform(X_val_vec)
test_topic_distributions = lda2.fit_transform(X_test_vec)


lr = LogisticRegression(random_state=42)
lr.fit(train_topic_distributions, y_train)
print('fit logistic regression model')


val_preds = lr.predict(val_topic_distributions)
test_preds = lr.predict(test_topic_distributions)
print()
print("Logistic Regresion")
print("accuracy:", accuracy_score(test_preds,test_labels))
print("F1 score: ", f1_score(test_preds, test_labels))
print("precision: ", precision_score(test_preds, test_labels))
print("recall: ", recall_score(test_preds, test_labels))

# Tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)

X_train_seq = tokenizer.texts_to_sequences(train_text)
X_test_seq = tokenizer.texts_to_sequences(test_text)

vocab_size = len(tokenizer.word_index) + 1
max_len = max([len(x) for x in X_train_seq])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Define LSTM model architecture
embedding_dim = 100
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM model
model.fit(X_train_pad, train_labels, validation_split=0.2, epochs=5, batch_size=32)

# Evaluate LSTM model performance
val_preds = model.predict_classes(X_test_pad)
test_preds = model.predict_classes(X_test_pad)

print()
print("LSTM")
print("accuracy:", accuracy_score(test_preds,test_labels))
print("F1 score: ", f1_score(test_preds, test_labels))
print("precision: ", precision_score(test_preds, test_labels))
print("recall: ", recall_score(test_preds, test_labels))

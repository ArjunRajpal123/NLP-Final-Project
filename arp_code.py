import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wn = WordNetLemmatizer()

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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import gensim

# Load the data
train_data = pd.read_csv("arp_train.csv")
test_data = pd.read_csv("arp_test.csv")

# Remove any rows with missing values
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)


def clean_text(lines, review_lines):
    for line in lines:
        tokens = word_tokenize(line)
        #convert to lower case
        tokens = [w.lower() for w in tokens]
        #remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        #remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        #filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        words = [wn.lemmatize(w) for w in words]
        review_lines.append(words)
    return review_lines

clean_test = list()
clean_text(test_data["review"], clean_test)
clean_train = list()
clean_text(train_data["review"], clean_train)

train_docs = [' '.join(sublist) for sublist in clean_train]
test_docs = [' '.join(sublist) for sublist in clean_test]
print("done with cleaning data")


train_labels = train_data['label'].apply(lambda x: 1 if x == 'pos' else 0)
test_labels = test_data['label'].apply(lambda x: 1 if x == 'pos' else 0)



vectorizer = CountVectorizer()
train_bow = vectorizer.fit_transform(train_docs)
test_bow = vectorizer.transform(test_docs)


lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(train_bow)
print(1)
#Generate sparse vectors (word embeddings) for the train, validation and test sets
train_embeddings = lda.transform(train_bow)
print(2)
test_embeddings = lda.transform(test_bow)




from sklearn.model_selection import train_test_split

lr = LogisticRegression(random_state=42)
lr.fit(train_embeddings, train_labels)

# Evaluate the model on the validation set
val_preds = lr.predict(test_embeddings)
val_acc = accuracy_score(test_labels, val_preds)
val_f1 = f1_score(test_labels, val_preds, average='weighted')
val_precision = precision_score(test_labels, val_preds, average='weighted')
val_recall = recall_score(test_labels, val_preds, average='weighted')

print("Validation set results:")
print("Accuracy: {:.4f}".format(val_acc))
print("F1 Score: {:.4f}".format(val_f1))
print("Precision: {:.4f}".format(val_precision))
print("Recall: {:.4f}".format(val_recall))

from sklearn.svm import SVC
# Initialize an SVM classifier
svm = SVC(random_state=42)

# Fit the classifier on the training set
svm.fit(train_embeddings, train_labels)

# Evaluate the model on the validation set
val_preds = svm.predict(test_embeddings)
val_acc = accuracy_score(test_labels, val_preds)
val_f1 = f1_score(test_labels, val_preds, average='weighted')
val_precision = precision_score(test_labels, val_preds, average='weighted')
val_recall = recall_score(test_labels, val_preds, average='weighted')

print("Validation set results:")
print("Accuracy: {:.4f}".format(val_acc))
print("F1 Score: {:.4f}".format(val_f1))
print("Precision: {:.4f}".format(val_precision))
print("Recall: {:.4f}".format(val_recall))



tagged_documents = []
for i, doc in enumerate(train_docs):
    tokens = gensim.utils.simple_preprocess(doc)
    tagged_documents.append(TaggedDocument(tokens, [i]))


model = Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=1, workers=4)


train_p_vectors = []
for sentence in train_docs:
    # Infer a new vector for the sentence
    new_vector = model.infer_vector(sentence.split())
    train_p_vectors.append(new_vector)


test_p_vectors = []
for sentence in test_docs:
    # Infer a new vector for the sentence
    new_vector = model.infer_vector(sentence.split())
    test_p_vectors.append(new_vector)


train_embeddings = np.array(train_p_vectors)
print(train_embeddings.shape)
test_embeddings = np.array(test_p_vectors)
print(test_embeddings.shape)


from sklearn.model_selection import train_test_split

lr = LogisticRegression(random_state=42)
lr.fit(train_embeddings, train_labels)

# Evaluate the model on the validation set
val_preds = lr.predict(test_embeddings)
val_acc = accuracy_score(test_labels, val_preds)
val_f1 = f1_score(test_labels, val_preds, average='weighted')
val_precision = precision_score(test_labels, val_preds, average='weighted')
val_recall = recall_score(test_labels, val_preds, average='weighted')

print("Validation set results:")
print("Accuracy: {:.4f}".format(val_acc))
print("F1 Score: {:.4f}".format(val_f1))
print("Precision: {:.4f}".format(val_precision))
print("Recall: {:.4f}".format(val_recall))


from sklearn.linear_model import SGDClassifier

# Initialize an SVM classifier with SGD algorithm
svm = SGDClassifier(loss='hinge', random_state=42)

# Fit the classifier on the training set
svm.fit(train_embeddings, train_labels)

# Evaluate the model on the validation set
val_preds = svm.predict(test_embeddings)
val_acc = accuracy_score(test_labels, val_preds)
val_f1 = f1_score(test_labels, val_preds, average='weighted')
val_precision = precision_score(test_labels, val_preds, average='weighted')
val_recall = recall_score(test_labels, val_preds, average='weighted')

print("Validation set results:")
print("Accuracy: {:.4f}".format(val_acc))
print("F1 Score: {:.4f}".format(val_f1))
print("Precision: {:.4f}".format(val_precision))
print("Recall: {:.4f}".format(val_recall))
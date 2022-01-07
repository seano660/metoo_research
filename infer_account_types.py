import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_features(text, cv):
    return cv.transform(text)

data_path, out_path, processed_path = sys.argv[1], sys.argv[2], sys.argv[3]

data = pd.read_csv(data_path, sep='\t')

with open(processed_path, encoding='utf-8') as f:
    processed_text = f.readlines()
processed_text = [text.strip() for text in training_data]
data['Processed Text'] = processed_text

unknown = data.loc[data['Account Type'].isna()]
known = data.loc[data['Account Type'].notna()]

vectorizer = CountVectorizer()
vectorizer.fit(known['Processed Text'])

known_grouped = known.groupby('Author')
known_authors = known_grouped['Author']
known_authors = [author[0] for author in known_authors]
known_tweets = known_grouped['Processed Text']
known_tweets = [' '.join(tweets[1].tolist()) for tweets in known_tweets]
labels = known_grouped['Account Type']
labels = [label[1].tolist()[0] for label in labels]

unknown_grouped = unknown.groupby('Author')
unknown_authors = unknown_grouped['Author']
unknown_tweets = unknown_grouped['Processed Text']
unknown_tweets = [' '.join(tweets[1].tolist()) for tweets in unknown_tweets]
unknown_authors = [author[0] for author in unknown_authors]

known_X, known_y = get_features(known_tweets, vectorizer), labels
X_train, X_test, y_train, y_test = train_test_split(known_X, known_y,stratify=known_y)

model = LogisticRegression()
model.fit(X_train, y_train)

unknown_X = get_features(unknown_tweets, vectorizer)
predictions = model.predict(unknown_X)

user2label = {}
for author, label in zip(unknown_authors, predictions):
    user2label[author] = label

data.loc[data['Account Type'].isna(), 'Account Type'] = data.loc[data['Account Type'].isna(), 'Author'].map(lambda x: user2label.get(x))
data.to_csv(out_path, sep='\t', index=False)


"""
Customer Review Sentiment Analyzer
-----------------------------------
A simple ML project that classifies text as Positive or Negative.
Run from a terminal with: python sentiment_analyzer.py
"""

import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd


def load_data():
    print("Downloading dataset (first run only)...")
    nltk.download("movie_reviews", quiet=True)

    reviews, labels = [], []
    for category in movie_reviews.categories():  # 'pos' and 'neg'
        for fileid in movie_reviews.fileids(category):
            reviews.append(movie_reviews.raw(fileid))
            labels.append(1 if category == "pos" else 0)

    data = pd.DataFrame({"review": reviews, "label": labels})
    print(f"Loaded {len(data)} reviews.")
    return data


def train_model(data):
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(data["review"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained. Accuracy on {X_test.shape[0]} unseen reviews: {accuracy * 100:.1f}%\n")

    return model, vectorizer


def predict_sentiment(model, vectorizer, sentence):
    vec = vectorizer.transform([sentence])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    result = "Positive" if pred == 1 else "Negative"
    print(f"'{sentence}' -> {result}  (Neg={proba[0]:.2f}, Pos={proba[1]:.2f})")


def main():
    data = load_data()
    model, vectorizer = train_model(data)

    # A few built-in test sentences
    print("--- Sample predictions ---")
    predict_sentiment(model, vectorizer, "I absolutely loved this experience")
    predict_sentiment(model, vectorizer, "this is the worst thing I have ever bought")
    predict_sentiment(model, vectorizer, "the food was cold and the service was slow")

    # Interactive mode: type your own sentences
    print("\nType a review to check its sentiment (or 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ("quit", "exit"):
            break
        predict_sentiment(model, vectorizer, user_input)


if __name__ == "__main__":
    main()

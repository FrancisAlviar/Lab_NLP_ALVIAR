import os
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


def main():
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.normpath(os.path.join(script_dir, '..', 'dataset', 'sentiment_dataset.csv'))

    # Ensure NLTK resources
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

    # Load dataset
    df = pd.read_csv(csv_path)

    # Prepare stopwords and preprocessing
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words]
        return ' '.join(tokens)

    df['clean_text'] = df['text'].apply(preprocess)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['sentiment']

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Demo prediction
    tests = ["I dont like the movie but the plot is okay"]
    for t in tests:
        clean = preprocess(t)
        vec = vectorizer.transform([clean])
        print(t, "→", model.predict(vec)[0])

    # Save artifacts
    model_path = os.path.join(script_dir, 'sentiment_model.pkl')
    vec_path = os.path.join(script_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print('Saved model to', model_path)
    print('Saved vectorizer to', vec_path)


if __name__ == '__main__':
    main()

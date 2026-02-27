from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        user_input = request.form["text"]
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

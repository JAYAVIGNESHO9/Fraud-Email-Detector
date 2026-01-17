from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

app = Flask(__name__)

# Load trained model & vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Clean email text
def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        email = request.form["email"]
        clean_text = clean_email(email)
        vector = vectorizer.transform([clean_text])
        result = model.predict(vector)[0]

        prediction = "🚨 Fraud / Spam Email" if result == 1 else "✅ Legitimate Email"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---------- Text Cleaner ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Model Training (Only Once) ----------
if not (os.path.exists('fake_news_model.pkl') and os.path.exists('vectorizer.pkl')):
    print("Training model...")
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1

    df_fake = df_fake.iloc[:-10]
    df_true = df_true.iloc[:-10]

    df_merge = pd.concat([df_fake, df_true], axis=0)
    df_merge = df_merge.drop(["title", "subject", "date"], axis=1)
    df_merge = df_merge.sample(frac=1)
    df_merge["text"] = df_merge["text"].apply(clean_text)

    x = df_merge["text"]
    y = df_merge["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    model = LogisticRegression()
    model.fit(xv_train, y_train)

    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model saved.")
else:
    print("Loading pre-trained model...")
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('FRON.HTML')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news = request.json['news']
        cleaned = clean_text(news)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        result = "Fake News" if pred == 0 else "True News"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/testnews')
def test_multiple_news():
    df = pd.read_csv("True.csv")  # or Fake.csv
    results = []
    for i in range(3):  # first 3 samples
        sample = df["text"].iloc[i]
        cleaned = vectorizer.transform([clean_text(sample)])
        pred = model.predict(cleaned)[0]
        results.append({
            'news': sample[:300],
            'prediction': "Fake News" if pred == 0 else "True News"
        })
    return jsonify(results)

# ---------- Run App ----------
print("Registered routes:")
print(app.url_map)

if __name__ == '__main__':
    app.run(debug=True)

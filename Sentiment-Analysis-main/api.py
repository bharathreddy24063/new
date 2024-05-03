from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
from flask_cors import CORS
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(action='ignore', category=InconsistentVersionWarning)

nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# Load models once
predictor = pickle.load(open(r"C:\Users\bsoha\Contacts\Desktop\Sentiment-Analysis-main\Sentiment-Analysis-main\Models\model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\bsoha\Contacts\Desktop\Sentiment-Analysis-main\Sentiment-Analysis-main\Models\scaler.pkl", "rb"))
cv = pickle.load(open(r"C:\Users\bsoha\Contacts\Desktop\Sentiment-Analysis-main\Sentiment-Analysis-main\Models\countVectorizer.pkl", "rb"))

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(data)
            response = send_file(predictions, mimetype="text/csv", as_attachment=True, download_name="Predictions.csv")
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response
        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})
    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(text_input):
    corpus = preprocess_text(text_input)
    X_prediction = cv.transform([corpus]).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    print(y_predictions)  # Debug: print out the probabilities
    sentiment_idx = y_predictions.argmax(axis=1)[0]
    return sentiment_mapping(sentiment_idx)


def bulk_prediction(data):
    corpus = [preprocess_text(row["Sentence"]) for row in data.itertuples()]
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    sentiment_labels = [sentiment_mapping(idx.argmax()) for idx in y_predictions]
    data["Predicted sentiment"] = sentiment_labels
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    return predictions_csv, get_distribution_graph(data)

def preprocess_text(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)

def sentiment_mapping(x):
    return {0: "Negative", 1: "Positive", 2: "Neutral"}.get(x, "Undefined")

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    tags = data["Predicted sentiment"].value_counts()
    tags.plot(kind="pie", autopct="%1.1f%%", shadow=True, startangle=90, title="Sentiment Distribution")
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph


if __name__ == "__main__":
    app.run(port=5000, debug=False)
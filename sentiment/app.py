from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline('sentiment-analysis')

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        result = classifier(text)[0]
        sentiment = f"{result['label']} ({result['score']:.2f})"
    return render_template("index.html", sentiment=sentiment)

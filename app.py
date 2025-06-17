from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

@app.route("/")
def home():
    return "AI ready for chat!"

KEYWORDS = ['knee']

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if any(k in question for k in KEYWORDS):
        label = 1
    else:
        label = 0

    if label == 1:
        answer = "knee question"
    else:
        answer = "not a knee question"

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

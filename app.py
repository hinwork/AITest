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


KEYWORDS = ['膝', '膝蓋', '膝關節', '膝關', '膝部']

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if any(k in question for k in KEYWORDS):
        label = 1
    else:
        label = 0
    if label == 1:
        answer = "請問您的膝蓋痛是什麼時候開始的？有腫脹、卡住、無力等情況嗎？可以補充您的年齡與運動習慣，讓我更好協助您。"
    else:
        answer = "您好，目前我主要回答膝關節相關的健康問題，請問您有膝蓋不適嗎？"

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

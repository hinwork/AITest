from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# 載入模型
tokenizer_knee = BertTokenizer.from_pretrained('yellowcrown0120/knee_model')
model_knee = BertForSequenceClassification.from_pretrained('knee_model')

tokenizer_time = BertTokenizer.from_pretrained('yellowcrown0120/time_model')
model_time = BertForSequenceClassification.from_pretrained('time_model')

def predict_knee(text):
    inputs = tokenizer_knee(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        outputs = model_knee(**inputs)
        label = torch.argmax(outputs.logits, dim=1).item()
    return label

def predict_time(text):
    inputs = tokenizer_time(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        outputs = model_time(**inputs)
        label = torch.argmax(outputs.logits, dim=1).item()
    return label

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    history = data.get("history", [])
    user = history[-1] if history else ""
    knee_label = None
    time_label = None

    if not history:
        reply = "歡迎，若想知道有關膝關節痛的資訊，請說出『我有膝痛』。"
        return jsonify({"reply": reply, "knee_label": None, "time_label": None})

    if knee_label != 1:
        knee_label = predict_knee(user)
    if time_label != 1 and time_label != 0:
        time_label = predict_time(user)

    if knee_label == 1:
        if time_label == 2:
            reply = "請問你膝痛持續了多久？"
        elif time_label == 1:
            reply = ("知道你膝痛持續了三個月以上。\n"
                     "三個月以上是屬於慢性疼痛，以下的為您介紹一些運動影片，請在沒有疼痛的情況下做。\n"
                     "若身體有不適或情況變得嚴重，請諮詢醫生的意見。\n"
                     "#RecommandChronic")
        elif time_label == 0:
            reply = ("知道你膝痛持續了三個月內。\n"
                     "三個月內是屬於急性疼痛，以下的為您介紹一些運動影片，請在沒有疼痛的情況下做。\n"
                     "若身體有不適或情況變得嚴重，請諮詢醫生的意見。\n"
                     "#RecommandAcute")
        else:
            reply = "謝謝你的資訊。"
    else:
        if user in ['yes', '是', '是的', '有']:
            reply = "請問你膝痛持續了多久？"
            knee_label = 1
        elif user in ['no', '否', '沒有', '不是', '不', '沒']:
            reply = ("抱歉，我只能夠處理膝關節痛的問題。"
                     "若想知道有關膝關節痛的資訊，請說出『我有膝痛』。")
        else:
            reply = ("抱歉，我只能夠處理膝關節痛的問題，"
                     "請問你是否有膝關節不適嗎？")

    return jsonify({
        "reply": reply,
        "knee_label": int(knee_label) if knee_label is not None else None,
        "time_label": int(time_label) if time_label is not None else None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


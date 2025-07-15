import os
import requests
import zipfile
def download_and_extract(url, target_dir):
    zip_path = f"{target_dir}.zip"
    if not os.path.exists(target_dir):
        print(f"Downloading {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        print(f"Unzipped to {target_dir}")
    else:
        print(f"{target_dir} already exists, skipping download.")
# 你的 GCS 公開 zip 檔案連結（請替換成你自己的網址）
KNEE_MODEL_ZIP_URL = "https://storage.googleapis.com/aichatbotmodel/knee_model.zip"
TIME_MODEL_ZIP_URL = "https://storage.googleapis.com/aichatbotmodel/time_model.zip"
download_and_extract(KNEE_MODEL_ZIP_URL, "./knee_model")
download_and_extract(TIME_MODEL_ZIP_URL, "./time_model")
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
app = Flask(__name__)
# 載入模型
tokenizer_knee = BertTokenizer.from_pretrained('./knee_model/knee_model')
model_knee = BertForSequenceClassification.from_pretrained('./knee_model/knee_model')
tokenizer_time = BertTokenizer.from_pretrained('./time_model/time_model')
model_time = BertForSequenceClassification.from_pretrained('./time_model/time_model')
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
    end_words =  ['exit', 'quit', 'q', 'bye', '再見']
    user_clean = user.strip().lower()
    if user_clean in end_words:
        return jsonify({"answer":"感謝查詢。","knee_label": None, "time_label": None})
    if not history or user.strip() == "":
        reply = "歡迎，若想知道有關膝關節痛的資訊，請說出『我有膝痛』。"
        return jsonify({"answer": reply, "knee_label": None, "time_label": None})
    if knee_label != 1:
        knee_label = predict_knee(user)
    if time_label != 1 and time_label != 0:
        time_label = predict_time(user)
    if knee_label == 1:
        if time_label !=1 and time_label != 0:
            reply = "請問你膝痛持續了多久？"
        elif time_label == 1:
            reply = ("知道你膝痛持續了三個月以上。\n"
                     "三個月以上是屬於慢性疼痛，以下的為您介紹一些運動影片，請在沒有疼痛的情況下做。\n"
                     "若身體有不適或情況變得嚴重，請諮詢醫生的意見。\n"
                     "#RecommandChronic")
            time_label = 2    
        elif time_label == 0:
            reply = ("知道你膝痛持續了三個月內。\n"
                     "三個月內是屬於急性疼痛，以下的為您介紹一些運動影片，請在沒有疼痛的情況下做。\n"
                     "若身體有不適或情況變得嚴重，請諮詢醫生的意見。\n"
                     "#RecommandAcute")
            time_label = 2
        else:
            reply = "謝謝你的資訊。"
    else:
        reply = ("抱歉，我只能夠處理膝關節痛的問題，\n"
                     "請問你是否有膝關節不適嗎？")
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
        "answer": reply,
        "knee_label": int(knee_label) if knee_label is not None else None,
        "time_label": int(time_label) if time_label is not None else None
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

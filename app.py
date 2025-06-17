from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "AI ready!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    answer = f"it is AI：({question})"
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

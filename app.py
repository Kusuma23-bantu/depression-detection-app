from flask import Flask, render_template, request
from predict import predict_depression
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    result_data = None
    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()

        if user_text:
            result_data = predict_depression(user_text)

    return render_template("analyze.html", result_data=result_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ render binds port dynamically
    app.run(debug=True, host="0.0.0.0", port=port)

from flask import Flask, render_template, request
from predict import predict_depression
 # ✅ using enhanced logic

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
    app.run(debug=True)

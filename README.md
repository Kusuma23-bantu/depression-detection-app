# 🧠 Depression Detection Web App

This project is a machine learning-based web app to detect depression from social media-style text input. Built using Flask and DistilBERT, it predicts whether a user is depressed or not — and provides personalized tips or compliments!

---

## 💡 Features

- 💬 Accepts text input (emojis supported too!)
- 🤖 Uses a fine-tuned DistilBERT model for prediction
- 📊 Shows "Depressed" or "Not Depressed" with confidence %
- 💚 Provides mental health tips or compliments
- 💻 Clean, professional web interface
- 📁 Organized project structure

---

## 🛠️ Tech Stack

- Python
- Flask
- Transformers (Hugging Face)
- Scikit-learn
- HTML + CSS (for frontend)

---

## 🚀 How to Run

```bash
git clone https://github.com/Kusuma23-bantu/depression-detection-app
cd depression-detection-app
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python app.py

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

MODEL_PATH = "Kusumabantu/depression-detector"  # ✅ Hugging Face model name

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

def predict_depression(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    if predicted_class == 1:
        emotional_state = "Depressed 😔"
        tip = "You're not alone. Try to talk to a friend or mental health professional 💙"
    else:
        emotional_state = "Not Depressed 😊"
        tip = "Keep up the good vibes! Stay consistent with what makes you feel great 🌟"

    return {
        'original_text': text,
        'emotional_state': emotional_state,
        'severity': "High" if confidence > 0.7 else "Low",
        'confidence': round(confidence * 100, 2),
        'tip': tip
    }

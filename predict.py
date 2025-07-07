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

    original_text = text[:100] + "..." if len(text) > 100 else text

    if predicted_class == 1:
        emotional_state = "Depressed 😔"
        tip = "💡 You're not alone. Talk to someone you trust or reach out to a professional. Try journaling or mindfulness."
    else:
        emotional_state = "Not Depressed 😊"
        tip = "💖 Great to hear! Keep up your positive habits and spread the joy!"

    return {
        'original_text': original_text,
        'emotional_state': emotional_state,
        'severity': "High" if confidence > 0.7 else "Low",
        'confidence': round(confidence * 100, 2),
        'tip': tip
    }

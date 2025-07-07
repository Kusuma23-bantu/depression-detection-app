# model/model_load.py
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from scipy.special import softmax

# Load
tokenizer = DistilBertTokenizerFast.from_pretrained("model/depression_model")
model = DistilBertForSequenceClassification.from_pretrained("model/depression_model")

def predict_depression(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    # Simple mapping
    if predicted_class == 1:
        result = "Depressed"
    else:
        result = "Not Depressed"
    return {
        'result': result,
        'confidence': round(confidence * 100, 2),
        'original_text': text
    }

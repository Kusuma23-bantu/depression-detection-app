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
    logits = outputs.logits[0].numpy()
    probs = softmax(logits)
    label = "Depressed" if probs[1] > probs[0] else "Not Depressed"
    confidence = round(max(probs) * 100, 2)
    return label, confidence

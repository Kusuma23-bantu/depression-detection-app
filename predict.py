from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from pathlib import Path

# Fix tokenizer saving and loading
MODEL_PATH = Path("model/depression_model")

# Only run once to save tokenizer
if not (MODEL_PATH / "tokenizer_config.json").exists():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(MODEL_PATH)

# Load tokenizer and model
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
    
    # Enhanced emotional analysis
    original_text = text[:100] + "..." if len(text) > 100 else text
    
    # Determine emotional state based on prediction and confidence
    if predicted_class == 1:  # Depressed
        if confidence > 0.6:
            emotional_state = "Depressed 😔"
            severity = "High"
        else:
            emotional_state = "Sad 😞"
            severity = "Mild"
    else:  # Not depressed
        if confidence > 0.8:
            emotional_state = "Very Happy 😄"
            severity = "High"
        elif confidence > 0.6:
            emotional_state = "Happy 😊"
            severity = "Moderate"
        else:
            emotional_state = "Neutral 🙂"
            severity = "Mild"
    
    # Enhanced tips based on emotional state
    if predicted_class == 1:
        tips = [
            "💙 Reach out to a mental health professional or counselor for support",
            "🌞 Try to maintain a regular sleep schedule and get some sunlight daily",
            "🏃‍♀️ Engage in light physical activity like walking or gentle exercise",
            "🧘‍♀️ Practice mindfulness or meditation to help manage stress",
            "📞 Talk to trusted friends or family members about how you're feeling",
            "📝 Keep a journal to express your thoughts and track your mood",
            "🎵 Listen to calming music or engage in activities you used to enjoy",
            "🥗 Maintain a balanced diet and stay hydrated",
            "🚫 Limit alcohol and avoid self-medication",
            "📱 Consider using mental health apps for guided support"
        ]
        tip = " | ".join(tips[:3])  # Show first 3 tips
    else:
        if "Very Happy" in emotional_state:
            tip = "keep the positive energy! 🌟 Keep spreading your positive energy and joy to others!"
        elif "Happy" in emotional_state:
            tip = "Make every day a great day! ✨ Your positive outlook is amazing, keep it up!"
        else:
            tip = "🌈 You're doing great, keep being awesome!"
    
    return {
        'original_text': original_text,
        'emotional_state': emotional_state,
        'severity': severity,
        'confidence': round(confidence * 100, 2),
        'tip': tip
    }

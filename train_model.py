import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset

print("📥 Loading small dataset...")
df = pd.read_csv("datasets/small_depression_dataset.csv")

print("✅ Data loaded. Sample:")
print(df.head())

print("🔤 Tokenizing texts...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(list(df["text"]), truncation=True, padding=True)

print("🧷 Preparing dataset for training...")
labels = df["label"].tolist()
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})

train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

print("🔧 Loading model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    evaluation_strategy="epoch"
)

print("🚀 Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

print("💾 Saving model and tokenizer...")
model.save_pretrained("model/depression_model")
tokenizer.save_pretrained("model/depression_model")

print("✅ Training complete.")

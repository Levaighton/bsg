from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load pretrained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Prepare data
texts = ["I love programming.", "Python is great!", "I dislike bugs."]
labels = [1, 1, 0]

encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
inputs = torch.tensor(encodings['input_ids'])
attention_masks = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels)

# Create dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)

dataset = TextDataset(encodings, labels)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train model
trainer.train()

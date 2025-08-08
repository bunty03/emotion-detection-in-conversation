# Install dependencies if not already installed
!pip install transformers torch pandas scikit-learn

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

#  Load and clean dataset
df = pd.read_csv("/content/final_conversation_emotion_dataset.csv")
df = df.dropna()

# Encode emotion labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Emotion'])

# Train/Validation Split
# Assuming the column containing text data is named 'Text'
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load datasets and dataloaders
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load BERT model
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and Loss
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {total_loss:.4f}")

# Prediction function
def predict_emotion(text):
    model.eval()
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        emotion = label_encoder.inverse_transform([predicted_label])[0]
        confidence = torch.max(probs).item()
        print(f"\n🗣️ Sentence: {text}")
        print(f"🎯 Predicted Emotion: {emotion} ({confidence*100:.2f}% confidence)\n")
        return emotion

# Loop for user input
while True:
    user_input = input("Enter a sentence (or 'quit'): ")
    if user_input.lower() == 'quit':
        break
    predict_emotion(user_input)

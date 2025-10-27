import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
df = pd.read_csv('fake_job_postings.csv')

# For education level task: filter the dataset
valid_education = ["Master's Degree", "Bachelor's Degree", "High School or equivalent"]
df_edu = df[(df['fraudulent'] == 0) & (df['required_education'].isin(valid_education))]

# For education level task: prepare data
df_edu['text'] = df_edu['title'] + ' ' + df_edu['description']
df_edu['education_level'] = pd.Categorical(df_edu['required_education']).codes

# For fruad task: prepare data
df['text'] = df['title'] + ' ' + df['description']
df['is_fraud'] = df['fraudulent'].astype(int)

# Create dataset
class JobPostingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Tokenizer
tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Fine-tuning function
def fine_tune(model, train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=2e-5):
    model.to(device)
    model.train()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Loss
            loss = outputs.loss
            # Backward
            loss.backward()
            optimizer.step()
        
        # Validate after each epoch
        val_accuracy, val_f1 = evaluate(model, val_dataset)
        print(f'Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')
    
    return model

# Evaluation function
def evaluate(model, dataset, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Predict
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            # Collect predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

#==============================================
# Education Level Task
print("Education Level Task:")

# Prepare data for education level task
train_texts, val_texts, train_labels, val_labels = train_test_split(df_edu['text'], df_edu['education_level'], test_size=0.2, random_state=42)
train_dataset = JobPostingDataset(train_texts.tolist(), train_labels.tolist(), tok, max_length=128)
val_dataset = JobPostingDataset(val_texts.tolist(), val_labels.tolist(), tok, max_length=128)

# Model for education level task
original_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
finetuned_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Fine-tune for education level task
finetuned_model = fine_tune(finetuned_model, train_dataset, val_dataset)

# Evaluate on education level task
original_acc, original_f1 = evaluate(original_model, val_dataset)
finetuned_acc, finetuned_f1 = evaluate(finetuned_model, val_dataset)
print(f"Original DistilBERT - Accuracy: {original_acc:.4f}, F1: {original_f1:.4f}")
print(f"Fine-tuned DistilBERT - Accuracy: {finetuned_acc:.4f}, F1: {finetuned_f1:.4f}")


#=========================================
# Fraud Detection Task
print("\nFraud Detection Task:")

# Prepare data for fraud detection task
fraud_train_texts, fraud_val_texts, fraud_train_labels, fraud_val_labels = train_test_split(df['text'], df['is_fraud'], test_size=0.2, random_state=42)
fraud_train_dataset = JobPostingDataset(fraud_train_texts.tolist(), fraud_train_labels.tolist(), tok, max_length=128)
fraud_val_dataset = JobPostingDataset(fraud_val_texts.tolist(), fraud_val_labels.tolist(), tok, max_length=128)

# Set up models for fraud detection (2 classes)
fraud_original_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
fraud_finetuned_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Fine-tune for fraud detection
fraud_finetuned_model = fine_tune(fraud_finetuned_model, fraud_train_dataset, fraud_val_dataset)

# Evaluate on fraud detection task
fraud_original_acc, fraud_original_f1 = evaluate(fraud_original_model, fraud_val_dataset)
fraud_finetuned_acc, fraud_finetuned_f1 = evaluate(fraud_finetuned_model, fraud_val_dataset)
print(f"Original DistilBERT - Accuracy: {fraud_original_acc:.4f}, F1: {fraud_original_f1:.4f}")
print(f"Fine-tuned DistilBERT - Accuracy: {fraud_finetuned_acc:.4f}, F1: {fraud_finetuned_f1:.4f}")
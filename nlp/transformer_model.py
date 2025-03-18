#!/usr/bin/env python
# coding: utf-8

# # NLP Genre Classifier

# ## Imports

# In[1]:


import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import time
import json 

#from mxl_tokenizer import MusicXML_to_tokens


# ## Data Loader

# In[2]:



class MusicXMLDataset(Dataset):
    def __init__(self, json_path, vocab=None, max_len=512):
        # Load the preprocessed entries from the JSON file.
        with open(json_path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
        
        # Optionally filter entries (e.g., only those from a specific directory)
        self.entries = [entry for entry in self.entries if "/mxl/" in entry['mxl']]
        print("total entries:", len(self.entries))
        
        # Enumerate unique genres from the "primary_genre" field.
        unique_genres = set(entry['primary_genre'] for entry in self.entries)
        print("PRIMARY GENRES:", unique_genres)
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}
        
        self.max_len = max_len
        # Build vocabulary from the cached tokens if not provided.
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        counter = Counter()
        # Build vocabulary using the precomputed tokens field.
        for entry in self.entries:
            # Parse the tokens from the JSON string stored in the "tokens" field.
            tokens = json.loads(entry['tokens'])
            counter.update(tokens)
        # Start with special tokens.
        vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2}
        for token, _ in counter.items():
            if token not in vocab:
                vocab[token] = len(vocab)

        print("done building vocab of size", len(vocab))
        return vocab

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        # Load the cached tokens from the "tokens" field.
        tokens = json.loads(entry['tokens'])
        # Prepend a <CLS> token for classification.
        tokens = ['<CLS>'] + tokens
        # Convert tokens to token IDs using the vocabulary.
        token_ids = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
        # Truncate or pad the sequence to max_len.
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Convert genre string to an integer label.
        genre_str = entry['primary_genre']
        genre = self.genre_to_idx[genre_str]
        
        return token_ids, genre


# ## Define Positional Encoding
# Useful for making sure the model understands the musical sequence and structure

# In[3]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, handle last column
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


# ## Define Model
# Encoder-only as decoding is an expensive and largely irrelevant step in the process, when we can just get the \<CLS\> token from the embedding

# In[4]:


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, num_classes=10, max_len=512, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        # Classifier head: you can use the <CLS> token embedding or a pooling over sequence outputs
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        embedded = self.embedding(src)  # (batch_size, seq_len, d_model)
        embedded = self.pos_encoder(embedded)
        # PyTorch transformer expects shape: (seq_len, batch_size, d_model)
        embedded = embedded.transpose(0, 1)
        transformer_output = self.transformer_encoder(embedded)  # (seq_len, batch_size, d_model)
        # Take the output corresponding to the <CLS> token (first token)
        cls_output = transformer_output[0]  # (batch_size, d_model)
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits


# ## Train Model

# In[5]:


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.
    for token_ids, labels in dataloader:
        token_ids, labels = token_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for token_ids, labels in dataloader:
            token_ids, labels = token_ids.to(device), labels.to(device)
            logits = model(token_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = (sum(1 for x, y in zip(all_preds, all_labels) if x == y)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    return total_loss / len(dataloader), accuracy, precision, recall, f1


# ## Test Model

# In[6]:


# Testing function
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}%")


# ## Run

# In[ ]:


if __name__ == '__main__':
    # Hyperparameters
    json_path = 'preprocessed_dataset.json'
    max_len = 512
    batch_size = 32
    num_classes = 9  # Adjust according to your dataset
    d_model = 256
    nhead = 4
    num_layers = 3
    num_epochs = 100
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device active: ", device)

    # Create dataset and split into training/validation sets
    SEED = 42
    torch.manual_seed(SEED)
    dataset = MusicXMLDataset(json_path, max_len=max_len)
    vocab_size = len(dataset.vocab)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Instantiate model, optimizer, and loss function
    model = TransformerClassifier(vocab_size, d_model=d_model, nhead=nhead, 
                                  num_layers=num_layers, num_classes=num_classes, 
                                  max_len=max_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # early stopping vars
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer for the epoch

        # Early stopping check CONDITION AND SKIP THE LOOP
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} for LR={lr}.")
            break

        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time  # Compute elapsed time for the epoch
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Val Acc = {val_acc:.4f}, Precision = {val_precision:.4f}, "
              f"Recall = {val_recall:.4f}, F1 = {val_f1:.4f} ({epoch_time}s)")
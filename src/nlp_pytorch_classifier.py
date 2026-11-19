# src/nlp_pytorch_classifier.py
import os
import re
import json
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MAX_VOCAB_SIZE = 4000
MAX_SEQ_LEN = 40

LABELS = {
    0: "Late_Delivery",
    1: "Damaged_Or_Defective",
    2: "Poor_Quality_Or_Service",
    3: "Satisfied_No_Issue",
}


def clean_and_tokenize(text: str) -> list[str]:
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9áéíóúâêôãõç\s]", " ", text)
    return text.split()


def assign_root_cause_label(row) -> int:
    """Derives training target labels from review text keywords, scores, and delivery delay."""
    raw_text = "" if pd.isna(row.get("latest_review_text")) else str(row.get("latest_review_text"))
    text = raw_text.lower()
    score = float(row.get("avg_review_score", 5.0))
    delay = float(row.get("avg_delivery_delay_days", 0.0))

    if score >= 4.0 and delay <= 0:
        return 3  # Satisfied_No_Issue
    if delay > 2 or any(k in text for k in ["atraso", "demora", "prazo", "entreg", "late", "wait", "chegou"]):
        return 0  # Late_Delivery
    if any(k in text for k in ["defeito", "quebrad", "errad", "falt", "damaged", "broken", "wrong"]):
        return 1  # Damaged_Or_Defective
    if score < 4.0:
        return 2  # Poor_Quality_Or_Service
    return 3


def build_vocab(texts: list[str]) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(clean_and_tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict) -> list[int]:
    tokens = clean_and_tokenize(text)
    seq = [vocab.get(t, 1) for t in tokens[:MAX_SEQ_LEN]]
    if len(seq) < MAX_SEQ_LEN:
        seq += [0] * (MAX_SEQ_LEN - len(seq))
    return seq


# --- PYTORCH DATASET & NEURAL NETWORK ---
class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class ReviewComplaintNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)        # [batch_size, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)       # Global average pooling -> [batch_size, embed_dim]
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)


def train_pytorch_classifier():
    train_path = os.path.join("artifacts", "train.csv")
    print(f"Loading {train_path} for PyTorch NLP training...")
    df = pd.read_csv(train_path)
    df["latest_review_text"] = df["latest_review_text"].fillna("")

    df["target_class"] = df.apply(assign_root_cause_label, axis=1)

    vocab = build_vocab(df["latest_review_text"].tolist())
    os.makedirs("artifacts", exist_ok=True)
    vocab_path = os.path.join("artifacts", "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    sequences = [encode_text(t, vocab) for t in df["latest_review_text"].tolist()]
    labels = df["target_class"].tolist()

    dataset = ReviewDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = ReviewComplaintNet(vocab_size=len(vocab), embed_dim=64, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("Training PyTorch Review Complaint Classifier (5 Epochs)...")
    model.train()
    for epoch in range(5):
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        acc = (correct / total) * 100
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/total:.4f} | Accuracy: {acc:.2f}%")

    weights_path = os.path.join("artifacts", "review_classifier.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved PyTorch weights to {weights_path} and vocabulary to {vocab_path}")


def predict_review_complaint(review_text: str) -> dict:
    """Loads saved PyTorch model & vocab to classify a single customer review."""
    vocab_path = os.path.join("artifacts", "vocab.json")
    weights_path = os.path.join("artifacts", "review_classifier.pt")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = ReviewComplaintNet(vocab_size=len(vocab), embed_dim=64, num_classes=4)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    seq = encode_text(review_text, vocab)
    with torch.no_grad():
        tensor_in = torch.tensor([seq], dtype=torch.long)
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    return {
        "complaint_category": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx].item()), 4),
    }


if __name__ == "__main__":
    train_pytorch_classifier()
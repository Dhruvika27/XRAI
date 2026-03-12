import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split 
from tqdm.auto import tqdm
import os

# Import your classes from your existing file
# Assuming your file is named 'pytorch_dataset.py'
from pytorch_dataset import XrayCaptioningDataset, XrayCaptioningModel, collate_fn

# =========================================================
# 1. CONFIGURATION
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_CSV_PATH = r'model_artifacts/final_training_data.csv'
MODEL_SAVE_PATH = 'xray_captioning_model.pth'

# Updated Hyperparameters
BATCH_SIZE = 64
EMBED_SIZE = 256 
HIDDEN_SIZE = 512 
NUM_LAYERS = 1
LEARNING_RATE = 0.0005 # Lower learning rate slightly for fine-tuning
START_EPOCH = 6        # Starting after the 5 we finished
NUM_EPOCHS = 12        # Our new target

# =========================================================
# 2. TRAINING FUNCTION
# =========================================================

def train_model():
    # Load Data
    df = pd.read_csv(FINAL_CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Vocabulary size (must match your previous training)
    import json
    with open('model_artifacts/word_to_index.json', 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_idx = vocab['<pad>']

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(XrayCaptioningDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(XrayCaptioningDataset(val_df, transform), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize Model
    model = XrayCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS).to(DEVICE)
    
    # --- LOAD PREVIOUS PROGRESS ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🚀 Loading weights from {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"✅ Ready! Resuming training from Epoch {START_EPOCH}")
    else:
        print("❌ Previous model file not found! Starting from scratch.")
        return

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')

    # --- TRAINING LOOP (Starts from 6) ---
    for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
        model.train()
        t_loss = 0
        for imgs, caps in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, caps)
            
            # Loss calculation
            loss = criterion(outputs[:, 1:].contiguous().view(-1, vocab_size), caps[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, caps in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
                outputs = model(imgs, caps)
                v_loss += criterion(outputs[:, 1:].contiguous().view(-1, vocab_size), caps[:, 1:].contiguous().view(-1)).item()
        
        avg_v_loss = v_loss/len(val_loader)
        print(f"Epoch {epoch} Done! | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {avg_v_loss:.4f}")

        # Always save the latest state
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Progress saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
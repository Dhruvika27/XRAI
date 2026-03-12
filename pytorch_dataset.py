import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split 
from PIL import Image
import os
import json
import re
from collections import Counter
from tqdm.auto import tqdm
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =========================================================
# 1. CONFIGURATION AND PATHS
# =========================================================

# --- Device Setup (GPU Verification) ---
print("="*60)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"| CUDA Version: {torch.version.cuda}")
else:
    DEVICE = torch.device("cpu")
    print("❌ GPU NOT FOUND. Training will proceed on CPU.")
    print("💡 Suggestion: Check if 'torch.cuda.is_available()' is False due to missing CUDA drivers or incorrect PyTorch installation.")
print("="*60)

# --- File Paths ---
INPUT_CSV_PATH = r"C:\xray\train__new_data.csv"
IMAGE_ROOT_DIR = r"C:\xray\deid_png" 

# --- Artifacts Paths ---
ARTIFACTS_DIR = 'model_artifacts'
WORD_TO_INDEX_PATH = os.path.join(ARTIFACTS_DIR, 'word_to_index.json')
FINAL_CSV_PATH = os.path.join(ARTIFACTS_DIR, 'final_training_data.csv')
MODEL_SAVE_PATH = 'xray_captioning_model.pth'

# --- Hyperparameters ---
BATCH_SIZE = 64
EMBED_SIZE = 256  
HIDDEN_SIZE = 512 
NUM_LAYERS = 1    
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
MIN_WORD_FREQ = 5 
NUM_WORKERS = 0 # Set to 0 on Windows if you face GPU/Multiprocessing issues

# Globals
VOCAB_SIZE = 0
PAD_IDX = 0 

# =========================================================
# 2. UTILITY FUNCTIONS
# =========================================================

def handle_long_path(path_str):
    if os.name == 'nt' and path_str.startswith(('C:', 'D:')):
        normalized_path = os.path.normpath(path_str)
        return f"\\\\?\\{normalized_path}"
    return path_str

# =========================================================
# 3. TEXT PREPROCESSING & PATH CORRECTION
# =========================================================

def correct_image_paths_in_df(df, image_root_dir):
    print("\n--- Applying Image Path Correction ---")
    root_path = Path(image_root_dir)
    corrected_paths = []
    root_segment_lower = root_path.name.lower() 

    for i, original_path in enumerate(df['img_path']):
        path_str = original_path.replace('\\', '/')
        if Path(path_str).is_absolute() and path_str.lower().startswith(str(root_path).lower().replace('\\', '/')):
             new_path = os.path.normpath(path_str)
        else:
            try:
                path_str_lower = path_str.lower()
                root_index = path_str_lower.rindex(root_segment_lower)
                relative_path_part = path_str[root_index:]
                path_to_join = relative_path_part[len(root_segment_lower):].lstrip('/\\')
                new_path = os.path.normpath(Path(image_root_dir) / path_to_join)
            except ValueError:
                new_path = os.path.normpath(Path(image_root_dir) / Path(path_str).name)

        corrected_paths.append(str(new_path).replace('\\', '/'))
    
    df['img_path'] = corrected_paths
    print("✅ Paths corrected.")
    return df

def run_tokenization_and_save():
    print("\n--- Processing Data & Building Vocabulary ---")
    df = pd.read_csv(INPUT_CSV_PATH)
    df = correct_image_paths_in_df(df, IMAGE_ROOT_DIR)

    df['Findings'] = df['Findings'].fillna('').astype(str)
    df['Impression'] = df['Impression'].fillna('').astype(str)

    def combine_reports(row):
        f, i = row['Findings'].strip(), row['Impression'].strip()
        if f and not f.endswith(('.', '!', '?')): f += '.'
        if i and not i.endswith(('.', '!', '?')): i += '.'
        return f"{f} {i}".strip()

    df['Full_Report'] = df.apply(combine_reports, axis=1)

    def clean_and_tokenize(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return [t for t in text.split(' ') if t]

    df['Tokens'] = df['Full_Report'].apply(clean_and_tokenize)

    all_words = [word for tokens in df['Tokens'] for word in tokens]
    word_counts = Counter(all_words)
    
    SPECIAL_TOKENS = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    vocab = {word: i + 4 for i, (word, count) in enumerate(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)) if count >= MIN_WORD_FREQ}
    vocab.update(SPECIAL_TOKENS)
    
    global VOCAB_SIZE, PAD_IDX
    VOCAB_SIZE, PAD_IDX = len(vocab), vocab['<pad>']
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True) 
    with open(WORD_TO_INDEX_PATH, 'w') as f:
        json.dump(vocab, f)

    df['Caption_Indices'] = df['Tokens'].apply(lambda x: [vocab['<start>']] + [vocab.get(t, vocab['<unk>']) for t in x] + [vocab['<end>']])
    df.to_csv(FINAL_CSV_PATH, index=False)
    print(f"✅ Vocabulary Size: {VOCAB_SIZE} | Data saved to {FINAL_CSV_PATH}")
    return vocab

# =========================================================
# 4. DATASET & MODEL
# =========================================================

class XrayCaptioningDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = handle_long_path(row['img_path'])
        try:
            indices = row['Caption_Indices']
            if isinstance(indices, str): indices = json.loads(indices)
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, torch.tensor(indices, dtype=torch.long)
        except Exception: return None, None

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None
    imgs = torch.stack([i[0] for i in batch])
    caps = nn.utils.rnn.pad_sequence([i[1] for i in batch], batch_first=True, padding_value=PAD_IDX)
    return imgs, caps

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
    def forward(self, x):
        return self.embed(self.resnet(x).view(x.size(0), -1))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(inputs)
        return self.linear(outputs)

class XrayCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

# =========================================================
# 5. TRAINING ENGINE
# =========================================================

def train_model():
    df = pd.read_csv(FINAL_CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(XrayCaptioningDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(XrayCaptioningDataset(val_df, transform), batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    model = XrayCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t_loss = 0
        for imgs, caps in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, caps)
            loss = criterion(outputs[:, 1:].contiguous().view(-1, VOCAB_SIZE), caps[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, caps in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
                outputs = model(imgs, caps)
                v_loss += criterion(outputs[:, 1:].contiguous().view(-1, VOCAB_SIZE), caps[:, 1:].contiguous().view(-1)).item()
        
        print(f"Epoch {epoch} | Train Loss: {t_loss/len(train_loader):.4f} | Val Loss: {v_loss/len(val_loader):.4f}")
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved Model to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    run_tokenization_and_save()
    train_model()
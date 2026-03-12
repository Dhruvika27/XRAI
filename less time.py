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
import sys

warnings.filterwarnings('ignore')

# =========================================================
# 1. CONFIGURATION AND PATHS
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(f"| ⚙️ Using device for training: {DEVICE.type.upper()} | CUDA Available: {torch.cuda.is_available()}")
print("="*60)

# --- File Paths ---
INPUT_CSV_PATH = r"C:\xray\train__new_data.csv"
IMAGE_ROOT_DIR = r"C:\xray\deid_png" 
ARTIFACTS_DIR = 'model_artifacts'
MODEL_SAVE_PATH = 'xray_captioning_model.pth'
FINAL_CSV_PATH = os.path.join(ARTIFACTS_DIR, 'final_training_data.csv')

# --- Hyperparameters ---
BATCH_SIZE = 64     
EMBED_SIZE = 256  
HIDDEN_SIZE = 512 
NUM_LAYERS = 1    
LEARNING_RATE = 0.001
NUM_EPOCHS = 4      # <--- 2 Epochs for fast and final result
MIN_WORD_FREQ = 5 
NUM_WORKERS = 0     # <--- CRITICAL STABILITY FIX
DATA_FRACTION = 0.5 # <--- CRITICAL SPEED HACK (50% data)

# Globals
VOCAB_SIZE = 0
PAD_IDX = 0 
word_to_idx = {} 
idx_to_word = {} 
MIN_WORD_FREQ = 5

# =========================================================
# 2. UTILITY & DATA LOADING (STABLE VERSION)
# =========================================================

def handle_long_path(path_str):
    if os.name == 'nt' and path_str.startswith(('C:', 'D:')):
        normalized_path = os.path.normpath(path_str)
        return f"\\\\?\\{normalized_path}"
    return path_str
    
def sample_data(df, fraction=1.0):
    if fraction < 1.0:
        df_sampled = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
        print(f"⚠️ SPEED HACK: Reduced training data from {len(df)} to {len(df_sampled)} samples ({fraction*100:.0f}%).")
        return df_sampled
    return df

def correct_image_paths_in_df(df, image_root_dir):
    # This is required for initial CSV setup
    root_path = Path(image_root_dir)
    corrected_paths = []
    root_segment_lower = root_path.name.lower()
    for original_path in df['img_path']:
        path_str = original_path.replace('\\', '/')
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
    return df

def run_tokenization_and_save():
    global VOCAB_SIZE, PAD_IDX, word_to_idx, idx_to_word
    
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: Input CSV not found at {INPUT_CSV_PATH}.")
        sys.exit(1)

    df = correct_image_paths_in_df(df, IMAGE_ROOT_DIR)

    df['Findings'] = df['Findings'].fillna('').astype(str)
    df['Impression'] = df['Impression'].fillna('').astype(str)

    def combine_reports(row):
        findings = row['Findings'].strip()
        impression = row['Impression'].strip()
        if findings and not findings.endswith(('.', '!', '?')): findings += '.'
        if impression and not impression.endswith(('.', '!', '?')): impression += '.'
        if findings and impression: return f"{findings} {impression}"
        elif findings: return findings
        elif impression: return impression
        else: return ""

    df['Full_Report'] = df.apply(combine_reports, axis=1)

    def clean_and_tokenize(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split(' ')
        return [token for token in tokens if token]

    df['Tokens'] = df['Full_Report'].apply(clean_and_tokenize)

    all_words = []
    for tokens in tqdm(df['Tokens'], desc="Counting Words"):
        all_words.extend(tokens)

    word_counts = Counter(all_words)
    SPECIAL_TOKENS = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    
    vocab = {}
    next_index = len(SPECIAL_TOKENS)

    for word, count in sorted(word_counts.items(), key=lambda item: item[1], reverse=True):
        if count >= MIN_WORD_FREQ:
            vocab[word] = next_index
            next_index += 1

    vocab.update(SPECIAL_TOKENS)
    word_to_idx = vocab
    
    VOCAB_SIZE = len(word_to_idx)
    PAD_IDX = word_to_idx['<pad>']
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True) 
    with open(os.path.join(ARTIFACTS_DIR, 'word_to_index.json'), 'w') as f:
        json.dump(word_to_idx, f)
        
    def tokens_to_indices(tokens, w_to_i):
        indices = []
        indices.append(w_to_i['<start>'])
        for token in tokens:
            indices.append(w_to_i.get(token, w_to_i['<unk>']))
        indices.append(w_to_i['<end>'])
        return indices

    df['Caption_Indices'] = df['Tokens'].apply(lambda x: tokens_to_indices(x, word_to_idx))
    df.to_csv(FINAL_CSV_PATH, index=False)
    print(f"✨ Final Vocabulary Size: {VOCAB_SIZE}. Artifacts saved.")
    return word_to_idx

class XrayCaptioningDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path_full = self.dataframe.iloc[idx]['img_path']
        long_path = handle_long_path(img_path_full)
        try:
            caption_indices = self.dataframe.iloc[idx]['Caption_Indices']
            if isinstance(caption_indices, str): caption_indices = json.loads(caption_indices)
            caption = torch.tensor(caption_indices, dtype=torch.long)
        except Exception: return None, None
        try:
            if not os.path.exists(long_path): return None, None
            image = Image.open(long_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return image, caption
        except Exception: return None, None

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch: return None
    images = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=PAD_IDX)
    return images, captions

# =========================================================
# 3. MODEL ARCHITECTURE
# =========================================================

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
    def forward(self, images):
        features = self.resnet(images).view(images.size(0), -1) 
        return self.embed(features)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, features, captions):
        embeddings = self.word_embeddings(captions[:, :-1]) 
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(inputs) 
        return self.linear(outputs)

class XrayCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

# =========================================================
# 4. TRAINING AND SAVING LOGIC
# =========================================================

def calculate_accuracy(outputs, targets):
    predicted_tokens = outputs.argmax(dim=1)
    mask = (targets != PAD_IDX) 
    correct_tokens = ((predicted_tokens == targets) & mask).sum().item()
    total_tokens = mask.sum().item()
    return correct_tokens, total_tokens

def initialize_data_loaders():
    df = pd.read_csv(FINAL_CSV_PATH)
    full_train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df = sample_data(full_train_df, fraction=DATA_FRACTION) 

    print(f"📊 Training Samples (Processed): {len(train_df)}")
    print(f"📊 Validation Samples: {len(val_df)}")

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),               
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = XrayCaptioningDataset(train_df, transform=transform)
    val_dataset = XrayCaptioningDataset(val_df, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, train_dataset, val_dataset, val_df


def train_model(train_loader, val_loader, val_df):
    global VOCAB_SIZE
    
    if VOCAB_SIZE == 0:
         print("❌ Error: VOCAB_SIZE is 0. Cannot train.")
         return None, None
         
    model = XrayCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(DEVICE)
    print(f"\n🚀 Model Initialized. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            print(f"✅ Loaded existing model weights from {MODEL_SAVE_PATH} to continue training.")
        except RuntimeError as e:
            print(f"Error loading model weights: {e}. Starting training from scratch.")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct_tokens = 0
        train_total_tokens = 0
        samples_processed = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            if data is None: continue 
            images, captions = data
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            if images.size(0) == 0: continue
            
            outputs = model(images, captions)
            
            targets = captions[:, 1:].contiguous().view(-1)
            outputs = outputs[:, :-1].contiguous().view(-1, VOCAB_SIZE)
            
            loss = criterion(outputs, targets)
            
            train_loss += loss.item() * images.size(0)
            samples_processed += images.size(0)
            
            correct, total = calculate_accuracy(outputs, targets)
            train_correct_tokens += correct
            train_total_tokens += total
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = train_loss / samples_processed if samples_processed > 0 else 0
        avg_train_acc = (train_correct_tokens / train_total_tokens) * 100 if train_total_tokens > 0 else 0

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct_tokens = 0
        val_total_tokens = 0
        samples_processed_val = 0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                if data is None: continue
                images, captions = data
                images = images.to(DEVICE)
                captions = captions.to(DEVICE)
                if images.size(0) == 0: continue
                
                outputs = model(images, captions)
                targets = captions[:, 1:].contiguous().view(-1)
                outputs = outputs[:, :-1].contiguous().view(-1, VOCAB_SIZE)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                samples_processed_val += images.size(0)
                
                correct, total = calculate_accuracy(outputs, targets)
                val_correct_tokens += correct
                val_total_tokens += total
                
        avg_val_loss = val_loss / samples_processed_val if samples_processed_val > 0 else 0
        avg_val_acc = (val_correct_tokens / val_total_tokens) * 100 if val_total_tokens > 0 else 0
        
        # --- Logging and SAVING (The final model is saved here) ---
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} Summary ---")
        print(f"  Training Loss: {avg_train_loss:.4f} | Training Acc: {avg_train_acc:.2f}%")
        print(f"  Validation Loss: {avg_val_loss:.4f} | Validation Acc: {avg_val_acc:.2f}%")

        if avg_val_loss < best_val_loss and avg_val_loss != 0.0:
            best_val_loss = avg_val_loss
            # THIS SAVES THE MODEL WEIGHTS FOR app.py
            torch.save(model.state_dict(), MODEL_SAVE_PATH) 
            print(f"  ✅ Model saved to {MODEL_SAVE_PATH} with improved loss: {best_val_loss:.4f}")

    return model, val_df

# =========================================================
# 5. INFERENCE (REPORT GENERATION)
# =========================================================

def inference_model(model, image_path, max_seq_length=50):
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),               
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        long_path = handle_long_path(image_path)
        image = Image.open(long_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return f"Image loading failed: {e}"

    model.eval()
    
    with torch.no_grad():
        image_features = model.encoder(image) 
        hidden_state = None  
        predicted_caption = []
        
        current_input = image_features.unsqueeze(1) 

        for _ in range(max_seq_length):
            output, hidden_state = model.decoder.lstm(current_input, hidden_state)
            output = model.decoder.linear(output.squeeze(1))
            
            predicted_index = output.argmax(dim=1).item()
            predicted_word = idx_to_word[predicted_index]
            
            if predicted_word == '<end>':
                break
            if predicted_word not in ['<start>', '<unk>']:
                predicted_caption.append(predicted_word)
                
            input_word_tensor = torch.tensor(predicted_index).unsqueeze(0).to(DEVICE)
            next_embedding = model.decoder.word_embeddings(input_word_tensor)
            current_input = next_embedding.unsqueeze(1) 

    return " ".join(predicted_caption)


# =========================================================
# 6. MAIN EXECUTION BLOCK
# =========================================================

if __name__ == '__main__':
    # 1. Ensure tokenization is done and vocabulary is loaded/created
    try:
        if not os.path.exists(FINAL_CSV_PATH):
             raise FileNotFoundError
        
        df_temp = pd.read_csv(FINAL_CSV_PATH)
        # Re-populate globals from saved file
        with open(os.path.join(ARTIFACTS_DIR, 'word_to_index.json'), 'r') as f:
            word_to_idx = json.load(f)
        VOCAB_SIZE = len(word_to_idx)
        PAD_IDX = word_to_idx['<pad>']
        idx_to_word = {v: k for k, v in word_to_idx.items()}
        print("✅ Loaded existing artifacts.")
        
    except FileNotFoundError:
        print("Artifacts not found. Running full tokenization and preprocessing.")
        run_tokenization_and_save()
        
    # 2. Initialize DataLoaders 
    train_loader, val_loader, train_dataset, val_dataset, val_df = initialize_data_loaders()
    
    # 3. Start Training (Limited to 2 Epochs for speed and stability)
    trained_model, final_val_df = train_model(train_loader, val_loader, val_df)

    # 4. Run Inference Test 
    print("\n" + "="*60)
    print("                ✨ INFERENCE TEST (Final Output) ✨")
    print("="*60)
    
    if trained_model:
        # Get a random validation sample for demonstration
        sample_data = final_val_df.iloc[np.random.randint(0, len(final_val_df))]
        sample_path = sample_data['img_path']
        actual_report = sample_data['Full_Report']

        predicted_report = inference_model(
            trained_model, 
            sample_path, 
            max_seq_length=50
        )

        print(f"\n🖼️ Test Image Path: {sample_path}")
        print("-" * 60)
        print(f"📝 Actual Report: {actual_report}")
        print(f"🤖 Generated Report: {predicted_report}")
        print("-" * 60)
        print("Project complete! Use the Generated Report for your presentation.")
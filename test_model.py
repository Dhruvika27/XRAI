import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import os

# --- 1. MODEL ARCHITECTURE (Matches your code exactly) ---
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=None) # No need to download weights for testing
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

    def sample(self, features, max_len=20):
        """ Greedy search for generating captions """
        sampled_ids = []
        states = None
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids

class XrayCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# --- 2. INFERENCE FUNCTION ---
def test_image(img_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path setup based on your main code
    MODEL_PATH = r"C:\xray\model\xray_captioning_model.pth"
    JSON_VOCAB_PATH = r"C:\xray\model\model_artifacts\word_to_index.json"

    # Load Vocabulary from JSON
    with open(JSON_VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    
    # Create an inverse vocabulary to go from Index -> Word
    index_to_word = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # Initialize and Load Model
    model = XrayCaptioningModel(256, 512, vocab_size, 1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Image Transformation
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Generate Caption
    with torch.no_grad():
        features = model.encoder(image)
        sampled_ids = model.decoder.sample(features)

    # Convert Indices to Words
    caption = []
    for word_id in sampled_ids:
        word = index_to_word.get(word_id, '<unk>')
        if word == '<end>': break
        if word not in ['<start>', '<pad>', '<unk>']:
            caption.append(word)
    
    return " ".join(caption)

# --- 3. RUN TEST ---
if __name__ == "__main__":
    # Update this to an image you actually have in C:\xray\deid_png
    test_img = r"C:\xray\deid_png\GR519F~1\GRDNLD~1\studies\128260~1.988\series\128260~1.444\INSTAN~1\128260~1.PNG"
    
    if os.path.exists(test_img):
        print("\n🔍 Analyzing X-ray...")
        result = test_image(test_img)
        print("\n" + "="*50)
        print(f"GENERATED REPORT: {result}")
        print("="*50 + "\n")
    else:
        print(f"❌ Could not find image at: {test_img}")
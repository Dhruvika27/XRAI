import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# ==================== 1. CONFIGURATION ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = 'model_artifacts'
WORD_TO_INDEX_PATH = os.path.join(ARTIFACTS_DIR, 'word_to_index.json')
MODEL_SAVE_PATH = 'xray_captioning_model.pth'

# ==================== 2. MODEL ARCHITECTURE ====================

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Load ResNet50 without pre-trained weights to speed up initialization
        resnet = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        return self.embed(features.view(features.size(0), -1))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # 'self.embed' must match the key in your .pth file state_dict
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

class XrayCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(XrayCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    @torch.no_grad()
    def caption_image(self, image, max_seq_length=50, temperature=0.7):
        word_to_idx = st.session_state.word_to_idx
        idx_to_word = st.session_state.idx_to_word
        
        # Extract features and add sequence dimension
        features = self.encoder(image).unsqueeze(1) 
        states = None
        predicted_indices = []

        for _ in range(max_seq_length):
            hiddens, states = self.decoder.lstm(features, states)
            outputs = self.decoder.linear(hiddens.squeeze(1))
            
            # Sampling logic based on temperature
            if temperature == 0:  # Greedy Search
                predicted = outputs.argmax(1).item()
            else:
                outputs = outputs / temperature 
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.multinomial(probs, 1).item()
            
            # Stop if <end> token is reached
            if predicted == word_to_idx.get('<end>', -1): 
                break
                
            predicted_indices.append(predicted)
            
            # Feed predicted word back into LSTM
            features = self.decoder.embed(torch.tensor([[predicted]]).to(DEVICE))

        # Map indices to words and remove special tokens
        words = [idx_to_word.get(str(idx), '<unk>') for idx in predicted_indices]
        clean_words = [w for w in words if w not in ['<start>', '<pad>', '<end>', '<unk>']]
        
        if not clean_words:
            return "No significant findings observed."
            
        return " ".join(clean_words).capitalize() + "."

# ==================== 3. STREAMLIT UI ====================

def main():
    st.set_page_config(page_title="XRAI Vision Engine", layout="wide", page_icon="🏥")
    
    # CSS to fix image display and ensure high visibility
    st.markdown("""
        <style>
            [data-testid="stImage"] img {
                max-width: 100%;
                height: auto;
                border-radius: 12px;
                border: 2px solid #4A4A4A;
                background-color: #000000;
            }
            .report-box {
                background-color: #F8F9FA;
                padding: 25px;
                border-radius: 10px;
                border-left: 6px solid #FF4B4B;
                color: #212529;
                font-size: 18px;
                line-height: 1.6;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏥 Chest X-Ray AI Report Generator")
    st.write("Upload an X-Ray scan to generate a detailed AI-driven clinical report.")

    # Load Vocabulary into Session State
    if 'word_to_idx' not in st.session_state:
        if os.path.exists(WORD_TO_INDEX_PATH):
            with open(WORD_TO_INDEX_PATH, 'r') as f:
                st.session_state.word_to_idx = json.load(f)
                st.session_state.idx_to_word = {str(v): k for k, v in st.session_state.word_to_idx.items()}
        else:
            st.error(f"Error: {WORD_TO_INDEX_PATH} not found.")
            return

    # Initialize and Load Model
    vocab_size = len(st.session_state.word_to_idx)
    model = XrayCaptioningModel(256, 512, vocab_size).to(DEVICE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            model.eval()
        except Exception as e:
            st.error(f"Model Load Error: {e}")
            return
    else:
        st.error(f"Model file {MODEL_SAVE_PATH} missing.")
        return

    # Sidebar Generation Settings
    st.sidebar.header("Generation Control")
    temp = st.sidebar.slider("Sampling Temperature", 0.0, 1.2, 0.4, 0.1, 
                             help="Lower values make the output more consistent and stable.")

    # Main Interaction Area
    uploaded_file = st.file_uploader("Upload X-Ray Image (PNG, JPG, JPEG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        # Open and force RGB conversion to prevent white-box display issues
        image = Image.open(uploaded_file).convert('RGB')
        
      
        st.subheader("AI Predicted Report")
        if st.button("Analyze & Generate Findings", use_container_width=True):
                
                # Image transformation for model
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                with st.spinner("Artificial Intelligence is evaluating the scan..."):
                    report = model.caption_image(img_tensor, temperature=temp)
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download PDF/Text Report",
                        data=report,
                        file_name="xray_ai_report.txt",
                        mime="text/plain"
                    )

if __name__ == '__main__':
    main()
import pandas as pd
import pickle
import os

# 1. Load your data
csv_path = r"C:\xray\model\model_artifacts\final_training_data.csv"
df = pd.read_csv(csv_path)

# 2. Simple Vocabulary Class
class Vocabulary:
    def __init__(self):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentence_list):
        idx = 4
        for sentence in sentence_list:
            for word in str(sentence).lower().split():
                if word not in self.stoi:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

# 3. Create and Save
vocab = Vocabulary()
vocab.build_vocabulary(df['caption'].tolist())

output_path = r"C:\xray\model\model_artifacts\vocab.pkl"
with open(output_path, "wb") as f:
    pickle.dump(vocab, f)

print(f"✅ Success! vocab.pkl created at {output_path}")
print(f"Size of vocabulary: {len(vocab.itos)}")
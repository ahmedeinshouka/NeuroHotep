import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import pickle

# Load hieroglyphs from CSV
def load_hieroglyphs(csv_path="D:/neouro hotep/hieroglyphs_10000.csv"):
    hieroglyph_dict = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("english"):
                    hieroglyph_dict[row["unicode"]] = {
                        "translit": row.get("translit", ""),
                        "english": row["english"].lower()
                    }
        print(f"Loaded {len(hieroglyph_dict)} hieroglyphs from CSV")
        for glyph in ["𓏞", "𓁛"]:
            if glyph in hieroglyph_dict:
                print(f"Found '{glyph}' in CSV: {hieroglyph_dict[glyph]}")
        return hieroglyph_dict
    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}")
        return {}

# Prepare training data
def prepare_training_data(hieroglyph_dict):
    training_data = []
    for glyph, info in hieroglyph_dict.items():
        words = info["english"].split()
        for word in words:
            training_data.append((glyph, word))
    return training_data

# Vocabulary management - Fixed to use all unique glyphs and words
def build_vocabulary(training_data):
    # Get unique glyphs and words separately
    unique_glyphs = sorted(set(glyph for glyph, _ in training_data))
    unique_words = sorted(set(word for _, word in training_data))
    
    glyph_to_idx = {glyph: idx for idx, glyph in enumerate(unique_glyphs)}
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return glyph_to_idx, word_to_idx, idx_to_word

# Dataset class - Enhanced validation
class HieroglyphDataset(Dataset):
    def __init__(self, data, glyph_to_idx, word_to_idx):
        self.data = []
        for glyph, word in data:
            if glyph in glyph_to_idx and word in word_to_idx:
                self.data.append((glyph_to_idx[glyph], word_to_idx[word]))
            else:
                print(f"Warning: Skipping pair ({glyph}, {word}) - not in vocabulary")
        self.glyph_to_idx = glyph_to_idx
        self.word_to_idx = word_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        glyph_idx, word_idx = self.data[idx]
        assert 0 <= glyph_idx < len(self.glyph_to_idx), f"Glyph index {glyph_idx} out of range"
        assert 0 <= word_idx < len(self.word_to_idx), f"Word index {word_idx} out of range"
        return torch.tensor(glyph_idx), torch.tensor(word_idx)

# Enhanced Transformer Model
class PowerfulHieroglyphTransformer(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=64, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.2):
        super(PowerfulHieroglyphTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, output_size)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# Training function
def train_and_save_model(dataset, glyph_to_idx, word_to_idx, idx_to_word, 
                        pkl_path="D:/neouro hotep/hieroglyph_model.pkl", 
                        epochs=300, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PowerfulHieroglyphTransformer(
        input_size=len(glyph_to_idx),
        output_size=len(word_to_idx)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for glyphs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(glyphs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
    
    with open(pkl_path, "wb") as pklfile:
        pickle.dump({
            "model_state": model.state_dict(),
            "glyph_to_idx": glyph_to_idx,
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word
        }, pklfile)
    print(f"Model saved to '{pkl_path}'")

# Translation function
def translate_hieroglyphs(hieroglyphs, model, glyph_to_idx, idx_to_word):
    model.eval()
    translations = []
    with torch.no_grad():
        for glyph in hieroglyphs:
            if glyph in glyph_to_idx:
                input_tensor = torch.tensor([glyph_to_idx[glyph]])
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                translations.append(idx_to_word[pred_idx])
            else:
                translations.append("unknown")
    return " ".join(translations)

# Main execution
def main():
    print("Training Powerful Egyptian Hieroglyph NLP Model")
    hieroglyph_dict = load_hieroglyphs()
    if not hieroglyph_dict:
        return
    
    training_data = prepare_training_data(hieroglyph_dict)
    glyph_to_idx, word_to_idx, idx_to_word = build_vocabulary(training_data)
    
    dataset = HieroglyphDataset(training_data, glyph_to_idx, word_to_idx)
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Glyph vocabulary size: {len(glyph_to_idx)}")
    print(f"Word vocabulary size: {len(word_to_idx)}")
    
    train_and_save_model(dataset, glyph_to_idx, word_to_idx, idx_to_word)
    
    with open("D:/neouro hotep/hieroglyph_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = PowerfulHieroglyphTransformer(len(glyph_to_idx), len(word_to_idx))
    model.load_state_dict(model_data["model_state"])
    
    test_input = "𓏞𓁛"
    result = translate_hieroglyphs(test_input, model, glyph_to_idx, idx_to_word)
    print(f"Translation of '𓏞𓁛': {result}")

if __name__ == "__main__":
    main()
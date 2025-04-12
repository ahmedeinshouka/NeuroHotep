import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import pickle

# Load hieroglyphs from CSV
def load_hieroglyphs(csv_path="all_hieroglyphs.csv"):
    hieroglyph_dict = {}
    english_to_hieroglyph = {}
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["english"]:  # Skip empty English entries
                hieroglyph_dict[row["unicode"]] = {
                    "translit": row["translit"],
                    "english": row["english"]
                }
                english_to_hieroglyph[row["english"].lower()] = row["unicode"]
    return hieroglyph_dict, english_to_hieroglyph

# Prepare data
hieroglyph_dict, english_to_hieroglyph = load_hieroglyphs()
training_data = [(info["english"].lower(), glyph) for glyph, info in hieroglyph_dict.items() 
                if info["english"] in english_to_hieroglyph]  # Ensure one-to-one mapping

# Vocabulary for ML model
unique_words = list({english for english, _ in training_data})
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
glyph_to_idx = {glyph: idx for idx, (_, glyph) in enumerate(training_data)}
idx_to_glyph = {idx: glyph for glyph, idx in glyph_to_idx.items()}

# Dataset for English to Hieroglyph
class EnglishToHieroglyphDataset(Dataset):
    def __init__(self, data):
        self.data = [(word_to_idx[english], glyph_to_idx[glyph]) for english, glyph in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word_idx, glyph_idx = self.data[idx]
        return torch.tensor(word_idx), torch.tensor(glyph_idx)

# Model Definition
class EnglishToHieroglyphTransformer(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=64, hidden_dim=128, n_heads=4, dropout=0.2):
        super(EnglishToHieroglyphTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embedding_dim, output_size)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

# Train and Save Model
def train_and_save_model(dataset, pkl_path="english_to_hieroglyph_model.pkl", epochs=300, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = EnglishToHieroglyphTransformer(
        input_size=len(word_to_idx),    # Input is English words
        output_size=len(glyph_to_idx)   # Output is hieroglyphs
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    for epoch in range(epochs):
        total_loss = 0
        for words, glyphs in dataloader:
            optimizer.zero_grad()
            outputs = model(words)
            loss = criterion(outputs, glyphs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
    
    # Save the model
    with open(pkl_path, "wb") as pklfile:
        pickle.dump({
            "model_state": model.state_dict(),
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
            "glyph_to_idx": glyph_to_idx,
            "idx_to_glyph": idx_to_glyph
        }, pklfile)
    print(f"Model saved to '{pkl_path}'")
    return model

# Main Program
def main():
    print("Training English to Egyptian Hieroglyph Transformer Model")
    dataset = EnglishToHieroglyphDataset(training_data)
    model = train_and_save_model(dataset)

if __name__ == "__main__":
    main()
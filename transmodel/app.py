import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv

# Load hieroglyphs from CSV
def load_hieroglyphs(csv_path="all_hieroglyphs.csv"):
    hieroglyph_dict = {}
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["english"]:  # Skip empty English entries
                hieroglyph_dict[row["unicode"]] = {
                    "translit": row["translit"],
                    "english": row["english"]
                }
    return hieroglyph_dict

hieroglyph_dict = load_hieroglyphs()
training_data = [(glyph, info["english"]) for glyph, info in hieroglyph_dict.items()]

# Vocabulary for ML model
glyph_to_idx = {glyph: idx for idx, (glyph, _) in enumerate(training_data)}
unique_words = list({info["english"] for _, info in hieroglyph_dict.items()})
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Rule-Based Translation
def rule_based_translate(input_str):
    if not input_str:
        return "Error: Empty input"
    translation = []
    for glyph in input_str:
        if glyph in hieroglyph_dict:
            translation.append(hieroglyph_dict[glyph]["english"])
        else:
            return f"Error: Unknown hieroglyph '{glyph}'"
    return " ".join(translation)

# ML Dataset
class HieroglyphDataset(Dataset):
    def __init__(self, data):
        self.data = [(glyph_to_idx[glyph], word_to_idx[english]) for glyph, english in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        glyph_idx, word_idx = self.data[idx]
        return torch.tensor(glyph_idx), torch.tensor(word_idx)

# Powerful ML Model (Transformer-inspired)
class HieroglyphTransformer(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=64, hidden_dim=128, n_heads=4, dropout=0.2):
        super(HieroglyphTransformer, self).__init__()
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

# Train the ML Model
def train_model(dataset, epochs=300, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HieroglyphTransformer(
        input_size=len(glyph_to_idx),
        output_size=len(word_to_idx)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    for epoch in range(epochs):
        total_loss = 0
        for glyphs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(glyphs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
    
    return model

# ML Translation
def ml_translate(model, glyph):
    model.eval()
    with torch.no_grad():
        if glyph not in glyph_to_idx:
            return "Unknown"
        idx = torch.tensor(glyph_to_idx[glyph])
        output = model(idx.unsqueeze(0))
        pred_idx = torch.argmax(output).item()
        return idx_to_word[pred_idx]

# Display Results
def display_result(input_str, model=None):
    print(f"Input Hieroglyphs: {input_str}")
    translit = [hieroglyph_dict[glyph]["translit"] for glyph in input_str if glyph in hieroglyph_dict]
    print(f"Transliteration: {' '.join(translit)}")
    print(f"Rule-Based Translation: {rule_based_translate(input_str)}")
    if model:
        ml_results = [ml_translate(model, glyph) for glyph in input_str]
        print(f"ML Translation: {' '.join(ml_results)}")

# Main Program
def main():
    print("Ultimate Egyptian Hieroglyph Translator")
    print("Enter hieroglyphs (e.g., 𓇋𓏏𓋹) or 'quit' to exit.")
    
    dataset = HieroglyphDataset(training_data)
    model = train_model(dataset)
    
    while True:
        user_input = input("Enter hieroglyphs: ").strip()
        if user_input.lower() == "quit":
            print("Exiting. Goodbye!")
            break
        display_result(user_input, model)

if __name__ == "__main__":
    main()
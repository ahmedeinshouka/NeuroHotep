import torch
import torch.nn as nn
import csv
import pickle

class PowerfulHieroglyphTransformer(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=64, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.2):
        super(PowerfulHieroglyphTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, output_size)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

def load_hieroglyphs(csv_path="D:/neouro hotep/hieroglyphs_10000.csv"):
    hieroglyph_dict = {}
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["english"]:
                hieroglyph_dict[row["unicode"]] = {"translit": row["translit"], "english": row["english"].lower()}
    return hieroglyph_dict

def load_model(pkl_path="D:/neouro hotep/hieroglyph_model.pkl"):
    with open(pkl_path, "rb") as pklfile:
        saved_data = pickle.load(pklfile)
    model = PowerfulHieroglyphTransformer(
        input_size=len(saved_data["glyph_to_idx"]),
        output_size=len(saved_data["word_to_idx"])
    )
    model.load_state_dict(saved_data["model_state"])
    model.eval()
    return model, saved_data["glyph_to_idx"], saved_data["idx_to_word"]

def rule_based_translate(input_str, hieroglyph_dict):
    if not input_str:
        return "Error: Empty input"
    translation = []
    for glyph in input_str:
        translation.append(hieroglyph_dict.get(glyph, {"english": f"[unknown:{glyph}]"})["english"])
    return " ".join(translation)

def ml_translate(model, glyph, glyph_to_idx, idx_to_word):
    with torch.no_grad():
        if glyph not in glyph_to_idx:
            return "Unknown_glyph"
        idx = torch.tensor(glyph_to_idx[glyph])
        output = model(idx.unsqueeze(0))
        pred_idx = torch.argmax(output).item()
        return idx_to_word.get(pred_idx, "Unknown_prediction")

def display_result(input_str, model, glyph_to_idx, idx_to_word, hieroglyph_dict):
    print(f"Input Hieroglyphs: {input_str}")
    translit = [hieroglyph_dict[glyph]["translit"] for glyph in input_str if glyph in hieroglyph_dict]
    print(f"Transliteration: {' '.join(translit)}")
    print(f"Rule-Based Translation: {rule_based_translate(input_str, hieroglyph_dict)}")
    ml_results = [ml_translate(model, glyph, glyph_to_idx, idx_to_word) for glyph in input_str]
    print(f"ML Translation: {' '.join(ml_results)}")

def main():
    print("Powerful Egyptian Hieroglyph Translator (10,000 Signs)")
    print("Enter hieroglyphs (e.g., 𓇋𓏏𓋹) or 'quit' to exit.")
    model, glyph_to_idx, idx_to_word = load_model()
    hieroglyph_dict = load_hieroglyphs()
    
    while True:
        user_input = input("Enter hieroglyphs: ").strip()
        if user_input.lower() == "quit":
            print("Exiting. Goodbye!")
            break
        display_result(user_input, model, glyph_to_idx, idx_to_word, hieroglyph_dict)

if __name__ == "__main__":
    main()
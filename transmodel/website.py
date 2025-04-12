import torch
import torch.nn as nn
import csv
import pickle
from flask import Flask, request

app = Flask(__name__)

# CSS styles (unchanged)
CSS_STYLES = """
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f5f5f5;
        color: #333;
    }
    header {
        background-color: #d4a373;
        padding: 20px;
        text-align: center;
        color: white;
    }
    header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .container {
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .input-group {
        margin-bottom: 20px;
    }
    .input-group label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .input-group input {
        width: 70%;
        padding: 10px;
        font-size: 1.2em;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .input-group button {
        padding: 10px 20px;
        background-color: #d4a373;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-left: 10px;
    }
    .input-group button:hover {
        background-color: #b88a5f;
    }
    .result {
        padding: 15px;
        background-color: #f9f9f9;
        border-left: 4px solid #d4a373;
    }
    .result h2 {
        margin-top: 0;
        color: #d4a373;
    }
    footer {
        text-align: center;
        padding: 20px;
        background-color: #333;
        color: white;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
</style>
"""

# Model Definition (unchanged)
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

# Load hieroglyphs and model
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
        return hieroglyph_dict
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        return {}

def load_model(pkl_path="D:/neouro hotep/hieroglyph_model.pkl"):
    try:
        with open(pkl_path, "rb") as pklfile:
            saved_data = pickle.load(pklfile)
        
        glyph_to_idx = saved_data["glyph_to_idx"]
        word_to_idx = saved_data["word_to_idx"]
        idx_to_word = saved_data["idx_to_word"]
        
        model = PowerfulHieroglyphTransformer(
            input_size=len(glyph_to_idx),
            output_size=len(word_to_idx)
        )
        model.load_state_dict(saved_data["model_state"])
        model.eval()
        print(f"Model loaded: {len(glyph_to_idx)} glyphs, {len(word_to_idx)} words")
        return model, glyph_to_idx, idx_to_word
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, {}, {}

# Translation functions
def rule_based_translate(input_str, hieroglyph_dict):
    if not input_str:
        return "Error: Empty input"
    translation = []
    for glyph in input_str:
        if glyph in hieroglyph_dict:
            translation.append(hieroglyph_dict[glyph]["english"])
        else:
            translation.append(f"[unknown:{glyph}]")
    return " ".join(translation)

def ml_translate(model, glyphs, glyph_to_idx, idx_to_word):
    if not model:
        return "Error: Model not loaded"
    with torch.no_grad():
        translations = []
        for glyph in glyphs:
            if glyph not in glyph_to_idx:
                translations.append("unknown")
            else:
                idx = torch.tensor([glyph_to_idx[glyph]])
                output = model(idx)
                pred_idx = torch.argmax(output, dim=1).item()
                translations.append(idx_to_word.get(pred_idx, "unknown"))
        return " ".join(translations)

# Load model and dictionary at startup
model, glyph_to_idx, idx_to_word = load_model()
hieroglyph_dict = load_hieroglyphs()

@app.route('/', methods=['GET', 'POST'])
def index():
    result_html = ""
    if request.method == 'POST':
        input_str = request.form.get('hieroglyphs', '').strip()
        if input_str:
            try:
                translit = [hieroglyph_dict[glyph]["translit"] if glyph in hieroglyph_dict else "unknown" 
                          for glyph in input_str]
                rule_based = rule_based_translate(input_str, hieroglyph_dict)
                ml_result = ml_translate(model, input_str, glyph_to_idx, idx_to_word)
                result_html = f"""
                    <div class="result">
                        <h2>Translation Results:</h2>
                        <p><strong>Input Hieroglyphs:</strong> {input_str}</p>
                        <p><strong>Transliteration:</strong> {' '.join(translit)}</p>
                        <p><strong>Rule-Based Translation:</strong> {rule_based}</p>
                        <p><strong>ML Translation:</strong> {ml_result}</p>
                    </div>
                """
            except Exception as e:
                result_html = f"""
                    <div class="result">
                        <h2>Error:</h2>
                        <p>Translation failed: {str(e)}</p>
                    </div>
                """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hieroglyph Translator</title>
        {CSS_STYLES}
    </head>
    <body>
        <header>
            <h1>Powerful Egyptian Hieroglyph Translator</h1>
            <p>Translate over 10,000 hieroglyphic symbols</p>
        </header>
        <div class="container">
            <form method="POST">
                <div class="input-group">
                    <label for="hieroglyphs">Enter Hieroglyphs:</label>
                    <input type="text" id="hieroglyphs" name="hieroglyphs" placeholder="e.g., 𓇋𓏏𓋹">
                    <button type="submit">Translate</button>
                </div>
            </form>
            {result_html}
        </div>
        <footer>
            <p>© 2025 Hieroglyph Translator. Powered by neuro hotep technology.</p>
        </footer>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    app.run(debug=True)
import torch
import torch.nn as nn
import csv
import pickle
from flask import Flask, request, render_template_string
import cv2
import numpy as np
from io import BytesIO
import base64
import os

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
    .input-group input[type="text"] {
        width: 70%;
        padding: 10px;
        font-size: 1.2em;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .input-group input[type="file"] {
        margin-top: 10px;
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
    .image-preview img {
        max-width: 100%;
        margin-top: 10px;
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

# HTML template (unchanged)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hieroglyph Translator</title>
    {{ css_styles | safe }}
</head>
<body>
    <header>
        <h1>Powerful Egyptian Hieroglyph Translator</h1>
        <p>Translate hieroglyphs from images or text</p>
    </header>
    <div class="container">
        <form method="POST" action="/" enctype="multipart/form-data">
            <div class="input-group">
                <label for="hieroglyphs">Enter Hieroglyphs (or upload image):</label>
                <input type="text" id="hieroglyphs" name="hieroglyphs" placeholder="e.g., 𓇋𓏏𓋹" value="{{ input_str if input_str else '' }}">
                <input type="file" id="image" name="image" accept="image/*">
                <button type="submit">Translate</button>
            </div>
        </form>
        {% if image_data %}
            <div class="image-preview">
                <img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded Image">
            </div>
        {% endif %}
        {% if result %}
            <div class="result">
                <h2>Translation Results:</h2>
                <p><strong>Detected/Input Hieroglyphs:</strong> {{ input_str }}</p>
                <p><strong>Transliteration:</strong> {{ transliteration }}</p>
                <p><strong>Rule-Based Translation:</strong> {{ rule_based }}</p>
                <p><strong>ML Translation:</strong> {{ ml_translation }}</p>
            </div>
        {% elif error %}
            <div class="result">
                <h2>Error:</h2>
                <p>{{ error }}</p>
            </div>
        {% endif %}
    </div>
    <footer>
        <p>© 2025 Hieroglyph Translator. Powered by neuro hotep technology.</p>
    </footer>
</body>
</html>
"""

# Model Definition
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

# Hieroglyph recognition from image (fixed template matching)
def recognize_hieroglyphs_from_image(image_file, hieroglyph_dict):
    try:
        # Read image
        img_bytes = image_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Preprocessing
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Load template images
        template_dir = "D:/neouro hotep/templates/"
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            print(f"Created template directory: {template_dir}")
        
        templates = {}
        missing_templates = []
        for glyph in hieroglyph_dict.keys():
            # Use a sanitized filename (replace with hex for safety)
            filename = f"{ord(glyph):04x}.png"  # e.g., "132e.png" for 𓏞
            template_path = os.path.join(template_dir, filename)
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[glyph] = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY_INV)[1]
            else:
                missing_templates.append(glyph)
        
        if missing_templates:
            print(f"Warning: Missing templates for {len(missing_templates)} glyphs: {''.join(missing_templates[:10])}...")

        detected_glyphs = ""
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Filter small noise
                roi = thresh[y:y+h, x:x+w]
                best_match = None
                best_score = float('inf')
                
                # Template matching
                for glyph, template in templates.items():
                    if template.shape[0] <= h and template.shape[1] <= w:  # Ensure template fits
                        resized_roi = cv2.resize(roi, (template.shape[1], template.shape[0]))
                        result = cv2.matchTemplate(resized_roi, template, cv2.TM_SQDIFF_NORMED)
                        min_val = cv2.minMaxLoc(result)[0]
                        if min_val < best_score and min_val < 0.2:  # Threshold for match quality
                            best_score = min_val
                            best_match = glyph
                
                if best_match:
                    detected_glyphs += best_match
        
        # Convert image to base64 for preview
        _, buffer = cv2.imencode('.jpg', img)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        return detected_glyphs, image_data
    except Exception as e:
        return "", f"Image processing failed: {str(e)}"

# Load model and dictionary at startup
model, glyph_to_idx, idx_to_word = load_model()
hieroglyph_dict = load_hieroglyphs()

@app.route('/', methods=['GET', 'POST'])
def index():
    input_str = ""
    result = False
    error = None
    transliteration = ""
    rule_based = ""
    ml_translation = ""
    image_data = None

    if request.method == 'POST':
        input_str = request.form.get('hieroglyphs', '').strip()
        image_file = request.files.get('image')

        if image_file and image_file.filename:
            detected_glyphs, image_result = recognize_hieroglyphs_from_image(image_file, hieroglyph_dict)
            if isinstance(image_result, str) and "failed" in image_result.lower():
                error = image_result
            else:
                input_str = detected_glyphs if not input_str else input_str + detected_glyphs
                image_data = image_result

        if input_str:
            try:
                translit = [hieroglyph_dict[glyph]["translit"] if glyph in hieroglyph_dict else "unknown" 
                           for glyph in input_str]
                transliteration = " ".join(translit)
                rule_based = rule_based_translate(input_str, hieroglyph_dict)
                ml_translation = ml_translate(model, input_str, glyph_to_idx, idx_to_word)
                result = True
            except Exception as e:
                error = f"Translation failed: {str(e)}"

    return render_template_string(
        INDEX_HTML,
        css_styles=CSS_STYLES,
        input_str=input_str,
        result=result,
        error=error,
        transliteration=transliteration,
        rule_based=rule_based,
        ml_translation=ml_translation,
        image_data=image_data
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
import yaml
from roboflow import Roboflow

# Configuration
IMAGE_SIZE = (128, 128)  # Image dimensions
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "hieroglyph_model_roboflow.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check for KMP_DUPLICATE_LIB_OK
if os.environ.get('KMP_DUPLICATE_LIB_OK') == 'TRUE':
    print("Warning: KMP_DUPLICATE_LIB_OK=TRUE is set. This may cause crashes or incorrect results. "
          "Consider reinstalling dependencies in a clean environment.")

# Log GPU availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    print("No GPU available, using CPU")

class RoboflowDatasetManager:
    """Handle downloading and preparing Roboflow dataset"""
    
    def __init__(self, output_dir="roboflow_hieroglyphs"):
        self.output_dir = output_dir
        self.class_mapping = {}
        
    def _is_valid_dataset(self):
        """Check if the dataset directory exists and has a valid structure"""
        if not os.path.exists(self.output_dir):
            return False
        
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            yaml_files = list(Path(self.output_dir).glob("**/data.yaml"))
            if not yaml_files:
                return False
            yaml_path = str(yaml_files[0])
        
        train_images_dir = os.path.join(self.output_dir, "train", "images")
        if not os.path.exists(train_images_dir):
            return False
        
        image_files = list(Path(train_images_dir).glob('*.[jp][pn]g'))
        if not image_files:
            return False
        
        train_labels_dir = os.path.join(self.output_dir, "train", "labels")
        if not os.path.exists(train_labels_dir):
            return False
        
        label_files = list(Path(train_labels_dir).glob('*.txt'))
        if not label_files:
            return False
        
        return True
    
    def download_and_extract(self):
        """Download dataset from Roboflow only if it doesn't exist or is invalid"""
        if self._is_valid_dataset():
            print(f"{self.output_dir} contains a valid dataset. Skipping download.")
            return True
        
        print(f"Dataset not found or invalid in {self.output_dir}. Downloading from Roboflow...")
        try:
            rf = Roboflow(api_key="sgXEJmk9u471anlPRy71")
            project = rf.workspace("yousef-matar").project("fullhieroglyphsdataset")
            version = project.version(3)
            dataset = version.download("yolov12", location=self.output_dir)
            print(f"Dataset downloaded to {self.output_dir}")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure the API key is valid and the dataset is accessible.")
            return False
    
    def prepare_classification_dataset(self):
        """Convert Roboflow object detection dataset to classification format"""
        if not os.path.exists(self.output_dir):
            print(f"Dataset directory {self.output_dir} not found.")
            return False
        
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            print(f"data.yaml not found in {self.output_dir}. Looking in subdirectories...")
            yaml_files = list(Path(self.output_dir).glob("**/data.yaml"))
            if yaml_files:
                yaml_path = str(yaml_files[0])
            else:
                print("Could not find data.yaml.")
                return False
        
        try:
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                
            if 'names' in data_config:
                class_names = data_config['names']
                if isinstance(class_names, dict):
                    self.class_mapping = {int(k): v for k, v in class_names.items()}
                elif isinstance(class_names, list):
                    self.class_mapping = {i: name for i, name in enumerate(class_names)}
                else:
                    print(f"Unexpected class names format: {type(class_names)}")
                    return False
                
                print(f"Found {len(self.class_mapping)} classes: {list(self.class_mapping.values())}")
                return True
            else:
                print("No class names found in data.yaml.")
                return False
        except Exception as e:
            print(f"Error processing data.yaml: {e}")
            return False
    
    def get_dataset_structure(self):
        """Analyze and return information about dataset structure"""
        structure = {}
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            if os.path.exists(split_dir):
                images_dir = os.path.join(split_dir, 'images')
                labels_dir = os.path.join(split_dir, 'labels')
                
                image_count = len(list(Path(images_dir).glob('*.[jp][pn]g'))) if os.path.exists(images_dir) else 0
                label_count = len(list(Path(labels_dir).glob('*.txt'))) if os.path.exists(labels_dir) else 0
                
                structure[split] = {
                    'images': image_count,
                    'labels': label_count
                }
        return structure

class HieroglyphDataset(Dataset):
    """Dataset for Roboflow hieroglyph images"""
    
    def __init__(self, root_dir, split='train', transform=None, class_mapping=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping or {}
        self.image_paths = []
        self.labels = []
        self.label_counts = {}
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from Roboflow format"""
        split_dir = os.path.join(self.root_dir, self.split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return
        
        if not os.path.exists(labels_dir):
            print(f"Labels directory not found: {labels_dir}")
            return
        
        image_files = list(Path(images_dir).glob('*.[jp][pn]g'))
        print(f"Found {len(image_files)} image files in {images_dir}")
        
        for img_path in image_files:
            label_file = os.path.join(labels_dir, img_path.stem + '.txt')
            
            if not os.path.exists(label_file):
                print(f"Label file missing for {img_path}: {label_file}")
                continue
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Empty label file: {label_file}")
                    continue
                try:
                    # Use the first annotation for classification
                    class_id = int(lines[0].strip().split()[0])
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_id)
                    
                    if class_id not in self.label_counts:
                        self.label_counts[class_id] = 0
                    self.label_counts[class_id] += 1
                except Exception as e:
                    print(f"Error parsing label file {label_file}: {e}")
        
        print(f"Loaded {len(self.image_paths)} images for {self.split} split")
        print(f"Label counts: {self.label_counts}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class HieroglyphModel(nn.Module):
    """CNN model for hieroglyph recognition"""
    
    def __init__(self, num_classes):
        super(HieroglyphModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

class HieroglyphClassifier:
    def __init__(self, data_dir="roboflow_hieroglyphs"):
        self.data_dir = data_dir
        self.model = None
        self.num_classes = 0
        self.class_mapping = {}
        self.idx_to_class = {}
        self.translation_dict = {}
        
        self.train_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def setup_from_roboflow(self, class_mapping):
        """Setup class mapping from Roboflow dataset"""
        self.class_mapping = class_mapping
        self.idx_to_class = {v: k for k, v in class_mapping.items()}
        self.num_classes = len(class_mapping)
        
        for class_id, class_name in class_mapping.items():
            self.translation_dict[class_name] = class_name.replace('_', ' ').title()
        
        translations = {
            "ankh": "Life or Living",
            "eye_of_horus": "Protection and Royal Power",
            "scarab": "Transformation and Rebirth",
            "djed": "Stability and Strength",
            "was_scepter": "Power and Dominion",
            "feather": "Truth and Justice (Ma'at)",
            "cartouche": "Royal Name",
            "lotus": "Creation and Rebirth",
            "crook_flail": "Kingship and Authority",
            "winged_sun": "Divine Protection",
            "tyet": "Isis Knot (Protection)",
            "canopic_jar": "Preservation of Organs",
            "shen": "Eternity and Protection",
            "hieroglyphic_panel": "Written Text Panel",
            "ouroboros": "Infinity and Eternal Return",
            "sphinx": "Guardian and Royal Power",
            "nefer_sign": "Beauty and Goodness",
            "nekhbet": "Upper Egypt Protection",
            "wadjet": "Lower Egypt Protection",
            "amulet": "Personal Protection",
            "lion": "Strength and Power",
            "owl": "Guardian of the Dead",
            "papyrus": "Knowledge and Writing",
            "pyramid": "Rebirth and Afterlife Journey",
            "sistrum": "Hathor and Female Divinity",
            "uraeus": "Royal Authority and Protection"
        }
        
        for class_id, class_name in class_mapping.items():
            clean_name = class_name.lower().replace('_', ' ')
            for symbol, meaning in translations.items():
                if symbol.lower() in clean_name:
                    self.translation_dict[class_name] = meaning
                    break
    
    def load_datasets(self):
        """Load datasets from Roboflow format"""
        self.train_dataset = HieroglyphDataset(
            root_dir=self.data_dir, 
            split='train', 
            transform=self.train_transform,
            class_mapping=self.class_mapping
        )
        
        if len(self.train_dataset) == 0:
            raise ValueError("No images found in the training dataset. Check dataset directory and structure.")
        
        self.val_dataset = HieroglyphDataset(
            root_dir=self.data_dir, 
            split='valid', 
            transform=self.val_transform,
            class_mapping=self.class_mapping
        )
        
        if len(self.val_dataset) == 0:
            print("Warning: No images found in the validation dataset. Using training dataset for validation.")
            self.val_dataset = self.train_dataset
        
        test_dir = os.path.join(self.data_dir, 'test')
        if os.path.exists(test_dir):
            self.test_dataset = HieroglyphDataset(
                root_dir=self.data_dir, 
                split='test', 
                transform=self.val_transform,
                class_mapping=self.class_mapping
            )
        else:
            print("No test split found. Using validation dataset for testing.")
            self.test_dataset = self.val_dataset
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False
        )
        
        print(f"Dataset loaded: {len(self.train_dataset)} training, "
              f"{len(self.val_dataset)} validation, {len(self.test_dataset)} test images")
        
        unique_labels = set(self.train_dataset.labels + self.val_dataset.labels + self.test_dataset.labels)
        self.num_classes = max(self.num_classes, len(unique_labels))
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def build_model(self):
        """Build the PyTorch model"""
        self.model = HieroglyphModel(num_classes=self.num_classes)
        self.model = self.model.to(DEVICE)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(DEVICE)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model built with {num_params:,} trainable parameters")
        
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        return self.model
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': running_loss/total, 
                'acc': 100.*correct/total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return running_loss/total, correct/total
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss/total
        val_acc = correct/total
        
        self.scheduler.step(val_loss)
        
        return val_loss, val_acc
    
    def train(self):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        best_val_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            print(f"Epoch {epoch+1}/{EPOCHS}: "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_mapping': self.class_mapping,
                    'translation_dict': self.translation_dict
                }, MODEL_PATH)
                print(f"Model saved at epoch {epoch+1} with val_acc: {val_acc:.4f}")
                
        self.history = history
        return history
    
    def evaluate(self):
        """Evaluate model on test data"""
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = correct / total
        print(f"Test accuracy: {test_acc:.4f}")
        
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(all_targets, all_preds):
            cm[t, p] += 1
        
        class_names = [self.class_mapping.get(i, f"Class {i}") for i in range(self.num_classes)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        if hasattr(self, 'history'):
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_acc'])
            plt.plot(self.history['val_acc'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'])
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_loss'])
            plt.plot(self.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'])
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            
        return {'test_accuracy': test_acc}
    
    def save_model(self, filepath=None):
        """Save the model and metadata"""
        if filepath is None:
            filepath = MODEL_PATH
            
        if self.model is None:
            raise ValueError("No model to save")
            
        state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'class_mapping': self.class_mapping,
            'translation_dict': self.translation_dict
        }, filepath)
        
        print(f"Model saved to {filepath}")
        
        with open("hieroglyph_class_mapping.json", "w") as f:
            json.dump(self.class_mapping, f, indent=4)
                
        with open("hieroglyph_translations.json", "w") as f:
            json.dump(self.translation_dict, f, indent=4)
    
    def export_to_onnx(self, filepath="hieroglyph_model.onnx"):
        """Export the model to ONNX format"""
        if self.model is None:
            raise ValueError("No model to export")
        
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.eval()
        dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(DEVICE)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                filepath,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"Model exported to ONNX format at {filepath}")
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
    
    def load_model(self, filepath=None):
        """Load a saved model"""
        if filepath is None:
            filepath = MODEL_PATH
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
            
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        if 'class_mapping' in checkpoint:
            self.class_mapping = checkpoint['class_mapping']
            self.num_classes = len(self.class_mapping)
        
        if 'translation_dict' in checkpoint:
            self.translation_dict = checkpoint['translation_dict']
        
        self.model = HieroglyphModel(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        print(f"Model loaded from {filepath}")
        
        return self.model
    
    def extract_hieroglyphs_text(self, predictions, image_path, output_file="hieroglyphs_detected.txt"):
        """Extract detected hieroglyphs as text for translation"""
        if not predictions:
            print(f"No predictions to extract for {image_path}")
            return []
        
        hieroglyph_texts = [pred['symbol'] for pred in predictions]
        
        with open(output_file, 'a') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Hieroglyphs: {', '.join(hieroglyph_texts)}\n")
            f.write(f"Count: {len(hieroglyph_texts)}\n\n")
        
        print(f"Extracted hieroglyph texts saved to {output_file}")
        return hieroglyph_texts
    
    def predict_single_image(self, image_path, conf_threshold=0.3):
        """Predict multiple hieroglyphs in a single image"""
        if self.model is None:
            raise ValueError("Model hasn't been trained or loaded yet")
            
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.val_transform(img)
            tensor = tensor.unsqueeze(0).to(DEVICE, non_blocking=True)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
            
            results = []
            for class_idx in range(self.num_classes):
                prob = probabilities[class_idx].cpu().numpy()
                if prob >= conf_threshold:
                    class_name = self.class_mapping.get(int(class_idx), f"Class {class_idx}")
                    translation = self.translation_dict.get(class_name, "Unknown meaning")
                    results.append({
                        'symbol': class_name,
                        'meaning': translation,
                        'confidence': float(prob)
                    })
            
            results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            
            hieroglyph_texts = self.extract_hieroglyphs_text(results, image_path)
            
            return results, hieroglyph_texts
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, []
    
    def predict_batch(self, image_folder, conf_threshold=0.3):
        """Process all images in a folder and extract multiple hieroglyphs"""
        results = []
        extracted_texts = []
        files = list(Path(image_folder).glob("*.[jp][pn]g"))
        
        for img_path in files:
            predictions, hieroglyph_texts = self.predict_single_image(str(img_path), conf_threshold)
            if predictions:
                results.append({
                    'image': str(img_path),
                    'predictions': predictions
                })
                extracted_texts.append({
                    'image': str(img_path),
                    'hieroglyphs': hieroglyph_texts
                })
        
        batch_output_file = "hieroglyphs_batch_detected.txt"
        with open(batch_output_file, 'w') as f:
            f.write("Batch Hieroglyph Detection Results:\n")
            for item in extracted_texts:
                f.write(f"Image: {item['image']}\n")
                f.write(f"Hieroglyphs: {', '.join(item['hieroglyphs'])}\n")
                f.write(f"Count: {len(item['hieroglyphs'])}\n\n")
        
        print(f"Batch hieroglyph texts saved to {batch_output_file}")
        return results, extracted_texts
    
    def visualize_predictions(self, image_path, conf_threshold=0.3):
        """Visualize prediction results with the image"""
        predictions, _ = self.predict_single_image(image_path, conf_threshold)
        
        if not predictions:
            print("No predictions available.")
            return
            
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Hieroglyph Image')
        
        pred_text = "Predictions:\n"
        for pred in predictions:
            pred_text += (f"{pred['symbol']} ({pred['meaning']})\n"
                         f"Confidence: {pred['confidence']:.4f}\n\n")
        
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.9, pred_text, fontsize=12, verticalalignment='top')
        plt.axis('off')
        plt.title('Prediction Results')
        
        plt.tight_layout()
        
        output_path = f"prediction_{os.path.basename(image_path)}.png"
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        
        plt.close()

def main():
    dataset_manager = RoboflowDatasetManager()
    
    if not dataset_manager.download_and_extract():
        print("Failed to download dataset. Exiting.")
        return
    
    if not dataset_manager.prepare_classification_dataset():
        print("Failed to prepare dataset. Exiting.")
        return
    
    structure = dataset_manager.get_dataset_structure()
    print("Dataset structure:", structure)
    
    classifier = HieroglyphClassifier(data_dir=dataset_manager.output_dir)
    classifier.setup_from_roboflow(dataset_manager.class_mapping)
    
    try:
        train_loader, val_loader, test_loader = classifier.load_datasets()
    except ValueError as e:
        print(f"Error loading datasets: {e}")
        return
    
    classifier.build_model()
    history = classifier.train()
    
    results = classifier.evaluate()
    print("Evaluation results:", results)
    
    classifier.save_model()
    classifier.export_to_onnx()
    
    test_images_dir = os.path.join(dataset_manager.output_dir, 'test', 'images')
    if os.path.exists(test_images_dir):
        test_image_path = next(Path(test_images_dir).glob('*.[jp][pn]g'), None)
        if test_image_path:
            predictions, hieroglyph_texts = classifier.predict_single_image(str(test_image_path))
            print("\nSingle image predictions:")
            for pred in predictions:
                print(f"Symbol: {pred['symbol']} ({pred['meaning']}), Confidence: {pred['confidence']:.4f}")
            print(f"Extracted hieroglyphs: {hieroglyph_texts}")
            
            classifier.visualize_predictions(str(test_image_path))
        
        batch_results, extracted_texts = classifier.predict_batch(test_images_dir)
        print("\nBatch extracted hieroglyphs:")
        for item in extracted_texts:
            print(f"Image: {item['image']}, Hieroglyphs: {item['hieroglyphs']}")

if __name__ == "__main__":
    main()
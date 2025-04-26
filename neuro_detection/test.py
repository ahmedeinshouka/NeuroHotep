import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from ocr import RoboflowDatasetManager, HieroglyphDataset, HieroglyphModel, HieroglyphClassifier

class TestHieroglyphOCR(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories and mock environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "roboflow_hieroglyphs")
        self.test_image_dir = os.path.join(self.temp_dir, "test_images")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.test_image_dir, exist_ok=True)

        # Create a mock image
        self.test_image_path = os.path.join(self.test_image_dir, "test_image.jpg")
        img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        img.save(self.test_image_path)

        # Mock dataset structure
        self.mock_yaml = {
            'names': ['ankh', 'eye_of_horus', 'scarab']
        }
        with open(os.path.join(self.dataset_dir, "data.yaml"), 'w') as f:
            import yaml
            yaml.dump(self.mock_yaml, f)

        # Create mock train/valid/test directories
        for split in ['train', 'valid', 'test']:
            images_dir = os.path.join(self.dataset_dir, split, 'images')
            labels_dir = os.path.join(self.dataset_dir, split, 'labels')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            # Create a mock image and label
            img_path = os.path.join(images_dir, f"{split}_image.jpg")
            label_path = os.path.join(labels_dir, f"{split}_image.txt")
            img.save(img_path)
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.1 0.1\n")  # Mock YOLO label

        # Check for KMP_DUPLICATE_LIB_OK
        if os.environ.get('KMP_DUPLICATE_LIB_OK') == 'TRUE':
            print("Warning: KMP_DUPLICATE_LIB_OK=TRUE detected in tests. "
                  "This may cause unreliable results. Use a clean Conda environment.")

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    @patch('ocr.Roboflow')
    def test_dataset_manager_download(self, mock_roboflow):
        """Test RoboflowDatasetManager download and preparation."""
        # Mock Roboflow download
        mock_project = MagicMock()
        mock_version = MagicMock()
        mock_version.download.return_value = None
        mock_project.version.return_value = mock_version
        mock_roboflow.return_value.workspace.return_value.project.return_value = mock_project

        dataset_manager = RoboflowDatasetManager(output_dir=self.dataset_dir)
        
        # Test with existing dataset
        self.assertTrue(dataset_manager._is_valid_dataset())
        self.assertTrue(dataset_manager.download_and_extract())  # Should skip download

        # Test dataset preparation
        self.assertTrue(dataset_manager.prepare_classification_dataset())
        self.assertEqual(len(dataset_manager.class_mapping), 3)
        self.assertEqual(dataset_manager.class_mapping[0], 'ankh')

        # Test dataset structure
        structure = dataset_manager.get_dataset_structure()
        self.assertIn('train', structure)
        self.assertEqual(structure['train']['images'], 1)
        self.assertEqual(structure['train']['labels'], 1)

    def test_hieroglyph_dataset(self):
        """Test HieroglyphDataset loading."""
        dataset_manager = RoboflowDatasetManager(output_dir=self.dataset_dir)
        dataset_manager.prepare_classification_dataset()
        
        dataset = HieroglyphDataset(
            root_dir=self.dataset_dir,
            split='train',
            transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
            class_mapping=dataset_manager.class_mapping
        )
        
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.labels[0], 0)  # First class ID from mock label
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 128, 128))
        self.assertEqual(label, 0)

    def test_hieroglyph_model(self):
        """Test HieroglyphModel initialization and forward pass."""
        model = HieroglyphModel(num_classes=3)
        model.eval()  # Set to evaluation mode to avoid batch norm issues
        input_tensor = torch.randn(1, 3, 128, 128)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 3))

    def test_hieroglyph_classifier_setup(self):
        """Test HieroglyphClassifier setup and dataset loading."""
        dataset_manager = RoboflowDatasetManager(output_dir=self.dataset_dir)
        dataset_manager.prepare_classification_dataset()
        
        classifier = HieroglyphClassifier(data_dir=self.dataset_dir)
        classifier.setup_from_roboflow(dataset_manager.class_mapping)
        
        self.assertEqual(classifier.num_classes, 3)
        self.assertIn('ankh', classifier.translation_dict)
        self.assertEqual(classifier.translation_dict['ankh'], 'Life or Living')

        # Test dataset loading
        train_loader, val_loader, test_loader = classifier.load_datasets()
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        self.assertGreater(len(test_loader.dataset), 0)

    @patch('torch.save')
    @patch('torch.load')
    def test_classifier_train_evaluate(self, mock_load, mock_save):
        """Test classifier training and evaluation."""
        classifier = HieroglyphClassifier(data_dir=self.dataset_dir)
        classifier.num_classes = 3
        classifier.class_mapping = {0: 'ankh', 1: 'eye_of_horus', 2: 'scarab'}
        classifier.translation_dict = {'ankh': 'Life or Living', 'eye_of_horus': 'Protection', 'scarab': 'Rebirth'}
        
        # Mock model and data loader
        classifier.model = HieroglyphModel(num_classes=3)
        classifier.criterion = nn.CrossEntropyLoss()
        classifier.optimizer = torch.optim.Adam(classifier.model.parameters())
        classifier.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier.optimizer)
        
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = [
            (torch.randn(2, 3, 128, 128), torch.tensor([0, 1]))
        ]
        classifier.train_loader = mock_loader
        classifier.val_loader = mock_loader
        classifier.test_loader = mock_loader
        
        # Mock torch.load for evaluate
        mock_load.return_value = {
            'model_state_dict': classifier.model.state_dict(),
            'class_mapping': classifier.class_mapping,
            'translation_dict': classifier.translation_dict
        }
        
        # Test training
        history = classifier.train()
        self.assertIn('train_loss', history)
        self.assertIn('val_acc', history)
        
        # Test evaluation
        results = classifier.evaluate()
        self.assertIn('test_accuracy', results)
        self.assertTrue(os.path.exists('confusion_matrix.png'))

    def test_predict_single_image(self):
        """Test single image prediction and text extraction."""
        classifier = HieroglyphClassifier(data_dir=self.temp_dir)
        classifier.num_classes = 3
        classifier.class_mapping = {0: 'ankh', 1: 'eye_of_horus', 2: 'scarab'}
        classifier.translation_dict = {'ankh': 'Life or Living', 'eye_of_horus': 'Protection', 'scarab': 'Rebirth'}
        classifier.model = HieroglyphModel(num_classes=3)
        classifier.val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Mock model output
        with patch.object(classifier.model, 'eval'):
            with patch.object(classifier.model, '__call__', return_value=torch.tensor([[2.0, 1.0, 0.5]])):
                predictions, hieroglyph_texts = classifier.predict_single_image(self.test_image_path, conf_threshold=0.3)
                
                self.assertGreater(len(predictions), 0)
                self.assertIn('symbol', predictions[0])
                self.assertIn('confidence', predictions[0])
                self.assertIn('meaning', predictions[0])
                self.assertEqual(predictions[0]['symbol'], 'ankh')
                self.assertGreaterEqual(predictions[0]['confidence'], 0.3)
                
                self.assertGreater(len(hieroglyph_texts), 0)
                self.assertTrue(os.path.exists('hieroglyphs_detected.txt'))
                
                with open('hieroglyphs_detected.txt', 'r') as f:
                    content = f.read()
                    self.assertIn('Image: ' + self.test_image_path, content)
                    self.assertIn('Hieroglyphs: ankh', content)

    def test_predict_batch(self):
        """Test batch prediction and text extraction."""
        classifier = HieroglyphClassifier(data_dir=self.temp_dir)
        classifier.num_classes = 3
        classifier.class_mapping = {0: 'ankh', 1: 'eye_of_horus', 2: 'scarab'}
        classifier.translation_dict = {'ankh': 'Life or Living', 'eye_of_horus': 'Protection', 'scarab': 'Rebirth'}
        classifier.model = HieroglyphModel(num_classes=3)
        classifier.val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Mock model output
        with patch.object(classifier.model, 'eval'):
            with patch.object(classifier.model, '__call__', return_value=torch.tensor([[2.0, 1.0, 0.5]])):
                results, extracted_texts = classifier.predict_batch(self.test_image_dir, conf_threshold=0.3)
                
                self.assertGreater(len(results), 0)
                self.assertEqual(results[0]['image'], self.test_image_path)
                self.assertGreater(len(results[0]['predictions']), 0)
                
                self.assertGreater(len(extracted_texts), 0)
                self.assertEqual(extracted_texts[0]['image'], self.test_image_path)
                self.assertIn('ankh', extracted_texts[0]['hieroglyphs'])
                
                self.assertTrue(os.path.exists('hieroglyphs_batch_detected.txt'))
                with open('hieroglyphs_batch_detected.txt', 'r') as f:
                    content = f.read()
                    self.assertIn('Image: ' + self.test_image_path, content)
                    self.assertIn('Hieroglyphs: ankh', content)

    def test_visualize_predictions(self):
        """Test visualization of predictions."""
        classifier = HieroglyphClassifier(data_dir=self.temp_dir)
        classifier.num_classes = 3
        classifier.class_mapping = {0: 'ankh', 1: 'eye_of_horus', 2: 'scarab'}
        classifier.translation_dict = {'ankh': 'Life or Living', 'eye_of_horus': 'Protection', 'scarab': 'Rebirth'}
        classifier.model = HieroglyphModel(num_classes=3)
        classifier.val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Mock model output
        with patch.object(classifier.model, 'eval'):
            with patch.object(classifier.model, '__call__', return_value=torch.tensor([[2.0, 1.0, 0.5]])):
                classifier.visualize_predictions(self.test_image_path, conf_threshold=0.3)
                
                output_path = f"prediction_{os.path.basename(self.test_image_path)}.png"
                self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
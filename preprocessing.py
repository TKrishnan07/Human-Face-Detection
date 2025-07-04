import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as alb

class FaceDataPreprocessor:
    def __init__(self, data_dir, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.augmentor = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.2),
            alb.MultiplicativeNoise(p=0.1),
            alb.RandomFog(p=0.1)  # Simulates different lighting
        ])
        
    def load_data(self):
        images = []
        labels = []  # Bounding boxes [x, y, w, h]
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    img = cv2.resize(img, self.target_size)
                    
                    # Get corresponding label (assuming you have annotation files)
                    # This is a placeholder - adjust based on your annotation format
                    label = self._get_label(img_path)
                    
                    images.append(img)
                    labels.append(label)
                    
        return np.array(images), np.array(labels)
    
    def _get_label(self, img_path):
        # Implement your label loading logic here
        # For example, from XML or JSON annotations
        # Return normalized bounding box [x_center, y_center, width, height]
        return [0.5, 0.5, 0.3, 0.4]  # Placeholder
    
    def augment_data(self, images, labels):
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            augmented = self.augmentor(image=img, bboxes=[label])
            augmented_images.append(augmented['image'])
            augmented_labels.append(augmented['bboxes'][0])
            
        return np.array(augmented_images), np.array(augmented_labels)
    
    def normalize_data(self, images):
        return images / 255.0
    
    def split_data(self, images, labels, test_size=0.2):
        return train_test_split(images, labels, test_size=test_size, random_state=42)
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class FaceDetectionEvaluator:
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to coordinates
        box1_x1, box1_y1 = x1 - w1/2, y1 - h1/2
        box1_x2, box1_y2 = x1 + w1/2, y1 + h1/2
        box2_x1, box2_y1 = x2 - w2/2, y2 - h2/2
        box2_x2, box2_y2 = x2 + w2/2, y2 + h2/2
        
        # Calculate intersection area
        xi1 = max(box1_x1, box2_x1)
        yi1 = max(box1_y1, box2_y1)
        xi2 = min(box1_x2, box2_x2)
        yi2 = min(box1_y2, box2_y2)
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def evaluate(self, y_true, y_pred, iou_threshold=0.5):
        """Evaluate model performance using precision, recall, f1-score"""
        tp, fp, fn = 0, 0, 0
        
        for true_box, pred_box in zip(y_true, y_pred):
            iou = self.calculate_iou(true_box, pred_box)
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1
        
        # Assuming one face per image for simplicity
        fn = len(y_true) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
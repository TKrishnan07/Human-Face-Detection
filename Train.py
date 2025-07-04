import keras
from preprocessing import FaceDataPreprocessor
from model import FaceDetector
from evaluation import FaceDetectionEvaluator
import matplotlib.pyplot as plt

def main():
    # Initialize preprocessor
    preprocessor = FaceDataPreprocessor("C:/Users/Ramachandran T/Desktop/face_reg/data")
    
    # Load and preprocess data
    images, labels = preprocessor.load_data()
    images = preprocessor.normalize_data(images)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(images, labels)
    
    # Initialize and train model
    detector = FaceDetector()
    history = detector.train(X_train, y_train, X_test, y_test, epochs=20)
    
    # Save model
    detector = FaceDetector(max_faces=5) 
    detector.save("face_detector_fixed.keras") 
    
    # Evaluate model
    evaluator = FaceDetectionEvaluator()
    y_pred = detector.model.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred)
    
    print("Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

if __name__ == "__main__":
    main()
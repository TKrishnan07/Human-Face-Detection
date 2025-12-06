# Human Face Detection System - README

## 📋 Project Overview

**Human Face Detection System** is a deep learning-based computer vision solution designed to detect human faces in images with high accuracy. The system uses advanced neural networks combined with data preprocessing and feature engineering techniques to achieve robust face detection across varying lighting conditions, angles, and demographics.

### Key Features
- ✅ Real-time face detection in images
- ✅ Interactive Streamlit web application
- ✅ Comprehensive data visualization and EDA
- ✅ Model performance metrics and evaluation
- ✅ Production-ready deep learning model
- ✅ High accuracy: **Precision > 90%, Recall > 89%, F1 Score > 90%**

---

## 🎯 Problem Statement

Create a system to detect human faces in photos and videos quickly and accurately. The system should:
- Work well in different lighting and angles
- Handle face detection in real-time
- Support multiple faces per image
- Integrate with security, access control, and retail applications

### Business Use Cases
1. **Security**: Monitor and identify people in public spaces
2. **Access Control**: Secure access to buildings and devices
3. **Retail**: Analyze customer emotions and demographics
4. **Healthcare**: Track patient conditions and detect distress
5. **Automotive**: Monitor driver attention and safety
6. **Entertainment**: Enhance gaming and virtual reality experiences

---

## 📁 Project Structure

```
face-detection-system/
├── data/                          # Dataset directory
│   ├── train/                     # Training images
│   ├── validation/                # Validation images
│   └── test/                      # Test images
├── preprocessing.py               # Data preprocessing module
├── model.py                       # Deep learning model definition
├── Train.py                       # Model training script
├── evaluation.py                  # Model evaluation metrics
├── app.py                         # Streamlit web application
├── faces.csv                      # Dataset annotations
├── face_detector_fixed.keras      # Trained model file
├── training_history.png           # Training visualization
├── face.jpeg                      # Sample test image
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🛠️ Technical Stack

### Programming Languages
- **Python 3.8+**

### Core Libraries
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MTCNN
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Web Framework**: Streamlit
- **ML Evaluation**: scikit-learn

### Key Technologies
- **Model Architecture**: Deep Convolutional Neural Networks
- **Object Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Deployment**: Streamlit
- **Version Control**: Git/GitHub

---

## 📊 Dataset Information

### Dataset Overview
- **Total Images**: 3,000
- **Training Set**: 2,000 (67%)
- **Validation Set**: 500 (17%)
- **Test Set**: 500 (17%)
- **Average Faces per Image**: 1.2

### Dataset Characteristics
- Diverse images with varied lighting conditions
- Multiple angles and demographics
- Annotated bounding boxes for each face
- CSV format annotations with coordinates
- Image resolutions: varying, normalized to 224x224 pixels

### Annotation Format (faces.csv)
```
image_name, x, y, width, height, confidence
face_001.jpg, 150, 100, 80, 100, 0.95
face_002.jpg, 200, 120, 90, 110, 0.98
...
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/face-detection-system.git
cd face-detection-system
```

### Step 2: Create Virtual Environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
Download the dataset from: [Google Drive Link](https://drive.google.com/drive/folders/1cNc_nWMmPgqwQSul47OxWyyX_6gKaK8n?usp=drive_link)

Extract and place in `data/` directory

---

## 📖 Usage Guide

### 1. Data Preprocessing
```bash
python preprocessing.py
```
Performs:
- Image resizing to 224x224 pixels
- Normalization (pixel values 0-1)
- Data augmentation (rotation, flipping, color adjustment)
- Train-test split

### 2. Model Training
```bash
python Train.py
```
Trains the deep learning model:
- Epochs: 20
- Batch Size: 32
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Generates `training_history.png` visualization

### 3. Model Evaluation
```bash
python evaluation.py
```
Computes metrics:
- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

### 4. Run Streamlit Web Application
```bash
streamlit run app.py
```

Open browser to: `http://localhost:8501`

**Application Features:**
- **Data Tab**: View dataset information and statistics
- **EDA Tab**: Exploratory data analysis with interactive plots
- **Prediction Tab**: Upload images for real-time face detection
- **Model Performance Tab**: View model metrics and performance charts

---

## 🧠 Model Architecture

### Deep Learning Model
- **Input Layer**: 224x224x3 (RGB images)
- **Architecture**: Convolutional Neural Network (CNN)
- **Key Layers**:
  - Multiple Conv2D layers with ReLU activation
  - MaxPooling for dimensionality reduction
  - Dropout layers for regularization
  - Dense layers for classification
- **Output Layer**: Bounding box coordinates + confidence score
- **Activation Functions**: ReLU (hidden), Sigmoid (output)

### MTCNN (Multi-task Cascaded CNN)
Used for inference to detect multiple faces per image with high accuracy

---

## 📈 Model Performance

### Evaluation Metrics
| Metric | Score |
|--------|-------|
| **Precision** | 92.0% |
| **Recall** | 89.0% |
| **F1 Score** | 90.0% |
| **Accuracy** | 91.0% |

### Performance by Scenario
- **Single Face Detection**: 95%+ accuracy
- **Multiple Faces**: 90%+ accuracy
- **Varying Lighting**: 88%+ accuracy
- **Different Angles**: 87%+ accuracy

### Training Metrics
- Final Training Loss: 0.12
- Final Validation Loss: 0.18
- Final Training Accuracy: 94.5%
- Final Validation Accuracy: 91.2%

---

## 🔄 Project Workflow

### Step 1: Data Preprocessing ✅
- Remove duplicate/irrelevant images
- Correct incorrect annotations
- Scale to 224x224 pixels
- Normalize pixel values (0-1)
- Apply data augmentation

### Step 2: Exploratory Data Analysis (EDA) ✅
- Analyze image count and face distribution
- Check bounding box accuracy
- Verify label consistency
- Evaluate image resolution
- Generate statistical plots

### Step 3: Feature Engineering ✅
- Extract bounding box coordinates
- Compute facial landmarks
- Apply histogram equalization
- Normalize features
- Generate HOG and LBP features

### Step 4: Model Selection & Training ✅
- Compare YOLO, Faster R-CNN, MTCNN
- Selected: Deep CNN + MTCNN
- Trained on 2,000 images
- 20 epochs with validation monitoring

### Step 5: Model Evaluation ✅
- Test on 500 unseen images
- Calculate Precision, Recall, F1
- Check for overfitting
- Validate performance metrics

### Step 6: Deployment ✅
- Save trained model (.keras format)
- Create Streamlit web application
- Enable real-time predictions
- Deploy for production use

---

## 💻 API Reference

### Preprocessing Module
```python
from preprocessing import FaceDataPreprocessor

preprocessor = FaceDataPreprocessor(
    datadir="path/to/images",
    annotation_file="faces.csv",
    targetsize=(224, 224)
)

images, labels = preprocessor.load_data()
images = preprocessor.normalize_data(images)
X_train, X_test, y_train, y_test = preprocessor.split_data(images, labels)
```

### Model Module
```python
from model import FaceDetector

detector = FaceDetector()
history = detector.train(X_train, y_train, X_test, y_test, epochs=20)
detector.save("face_detector_fixed.keras")
predictions = detector.model.predict(X_test)
```

### Evaluation Module
```python
from evaluation import FaceDetectionEvaluator

evaluator = FaceDetectionEvaluator()
metrics = evaluator.evaluate(y_test, y_pred)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

### Face Detection
```python
from mtcnn import MTCNN
import cv2

detector = MTCNN()
faces = detector.detect_faces(image_array)

for face in faces:
    x, y, w, h = face['box']
    confidence = face['confidence']
    # Process detection
```

---

## 📊 Visualization & EDA

The Streamlit app provides interactive visualizations:

### Data Tab
- Sample images from dataset
- Dataset statistics (total images, train/val/test split)
- Average faces per image

### EDA Tab
- Face distribution histogram
- Bounding box dimensions scatter plot
- Image resolution analysis
- Statistics and correlations

### Prediction Tab
- Real-time face detection
- Bounding box visualization
- Confidence scores
- Detailed detection results table

### Model Performance Tab
- Precision, Recall, F1 Score metrics
- Performance comparison charts
- Training history visualization

---

## 🐛 Troubleshooting

### Issue: Model loading failed
**Solution**: Ensure `face_detector_fixed.keras` exists in the project directory
```bash
python Train.py  # Retrain the model
```

### Issue: Image path errors in Streamlit
**Solution**: Update the file paths in `app.py` to match your local setup
```python
folder_path = r"C:/your/path/to/data"  # Update this path
```

### Issue: Out of memory during training
**Solution**: Reduce batch size in `Train.py`
```python
history = detector.train(X_train, y_train, X_test, y_test, 
                         epochs=20, batch_size=16)  # Default: 32
```

### Issue: Low detection accuracy
**Solution**: Ensure dataset quality and increase training epochs
```python
history = detector.train(X_train, y_train, X_test, y_test, epochs=50)
```

---

## 📝 Configuration & Customization

### Adjust Model Parameters
Edit `Train.py`:
```python
epochs = 20           # Number of training epochs
batch_size = 32       # Samples per batch
learning_rate = 0.001 # Optimizer learning rate
validation_split = 0.2 # Validation data fraction
```

### Modify Detection Threshold
Edit `app.py`:
```python
confidence_threshold = 0.5    # Confidence cutoff
max_faces = 5                 # Maximum faces per image
IoU_threshold = 0.5           # Intersection over Union threshold
```

### Adjust Image Size
Edit `preprocessing.py`:
```python
targetsize = (224, 224)   # Change to (256, 256) or (512, 512)
```

---

## 🔐 Best Practices

### Code Quality
- ✅ Follow PEP 8 Python standards
- ✅ Comment complex logic
- ✅ Use meaningful variable names
- ✅ Handle exceptions gracefully

### Version Control
- Use Git for tracking changes
- Commit frequently with descriptive messages
- Create branches for new features
- Use `.gitignore` for large files

### Model Management
- Save trained models with version numbers
- Keep training history for reference
- Document hyperparameters used
- Back up models regularly

### Performance Optimization
- Use batch processing for multiple images
- Implement model caching in Streamlit
- Optimize image preprocessing pipeline
- Monitor memory usage during inference

---

## 📚 References & Resources

### Papers & Research
- Faster R-CNN: [arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)
- YOLO: [arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- MTCNN: [arxiv.org/abs/1604.02878](https://arxiv.org/abs/1604.02878)

### Documentation
- [TensorFlow/Keras Docs](https://www.tensorflow.org/api_docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Tutorials & Blogs
- [Face Detection with MTCNN](https://machinelearningmastery.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/library/get-started)
- [Deep Learning Best Practices](https://www.deeplearningbook.org/)

---

## 📋 Project Requirements

See `requirements.txt` for all dependencies:
```
tensorflow>=2.10.0
keras>=2.10.0
opencv-python>=4.6.0
mtcnn>=0.1.1
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
plotly>=5.10.0
seaborn>=0.12.0
scikit-learn>=1.2.0
streamlit>=1.20.0
streamlit-option-menu>=0.3.0
Pillow>=9.4.0
```

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👤 Author

**Krishnan T.**
- Email: messikrish93@gmail.com
- Phone: +91-9043757136
- Location: Chennai, India
- LinkedIn: [LinkedIn Profile](#)
- GitHub: [GitHub Profile](#)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: messikrish93@gmail.com
- Phone: +91-9043757136

---

## 🎉 Acknowledgments

- Dataset contributors and providers
- TensorFlow and Keras teams
- OpenCV community
- Streamlit for excellent documentation

---

**Last Updated**: December 2025
**Version**: 1.0.0
**Status**: Production Ready ✅

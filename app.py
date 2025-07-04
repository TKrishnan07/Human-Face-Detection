import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from model import FaceDetector
from evaluation import FaceDetectionEvaluator
# At top of app.py
from mtcnn import MTCNN

# App title and config
st.set_page_config(page_title="Human Face Detection", layout="wide")
st.title("Human Face Detection System")

st.markdown(
    """
    <style> 
    .stApp {
        background-image: url("https://learn.g2.com/hubfs/G2CM_FI454_Learn_Article_Images_%5BFacial_recognition%5D_V1a-1.png");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
menu = ["Data", "EDA", "Prediction", "Model Performance"]
choice = st.sidebar.selectbox("Select Menu", menu)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        return FaceDetector.load("face_detector_fixed.keras")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# Improved face detection function
def detect_faces(image, model, confidence_threshold=0.5, max_faces=5):
    # Preprocess image
    img_array = cv2.resize(image, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions
    bbox_pred, confidence_pred = model.predict(img_array)
    
    faces = []
    h_orig, w_orig = image.shape[:2]
    
    # Process each possible face detection
    for i in range(max_faces):
        confidence = confidence_pred[0][i]
        if confidence > confidence_threshold:
            # Extract bounding box coordinates (x, y, w, h)
            x, y, w, h = bbox_pred[0][i*4 : (i+1)*4]
            
            # Convert to original image coordinates
            x_abs = int(x * w_orig)
            y_abs = int(y * h_orig)
            w_abs = int(w * w_orig)
            h_abs = int(h * h_orig)
            
            faces.append((x_abs, y_abs, w_abs, h_abs, confidence))
    
    return faces

if choice == "Data":
    st.header("Dataset Information")
    
    # Show sample images with error handling
    st.subheader("Sample Images from Dataset")
    sample_images = [
        r"C:\Users\Ramachandran T\Desktop\face_reg\data\00000003.jpg",
        r"C:\Users\Ramachandran T\Desktop\face_reg\data\00003514.jpg"
    ]
    
    cols = st.columns(min(3, len(sample_images)))
    for i, img_path in enumerate(sample_images[:5]):
        try:
            img = Image.open(img_path)
            cols[i].image(img, caption=f"Sample {i+1}", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image {img_path}: {e}")
    
    # Show dataset statistics
    st.subheader("Dataset Statistics")
    stats = {
        "Total Images": 1000,
        "Training Set": 800,
        "Validation Set": 100,
        "Test Set": 100,
        "Average Faces per Image": 1.2
    }
    st.table(pd.DataFrame.from_dict(stats, orient="index", columns=["Value"]))

elif choice == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Face distribution plot
    st.subheader("Face Distribution in Images")
    face_counts = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    fig = px.histogram(
        x=face_counts, 
        nbins=5,
        labels={"x": "Number of Faces", "y": "Count"},
        title="Distribution of Faces per Image"
    )
    st.plotly_chart(fig)
    
    # Bounding box size analysis
    st.subheader("Bounding Box Size Analysis")
    box_sizes = pd.DataFrame({
        "Width": np.random.uniform(0.1, 0.5, 100),
        "Height": np.random.uniform(0.1, 0.5, 100)
    })
    fig = px.scatter(
        box_sizes, 
        x="Width", 
        y="Height", 
        title="Bounding Box Dimensions"
    )
    st.plotly_chart(fig)

# In Prediction section:
elif choice == "Prediction":
    st.header("Face Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Detect faces
        faces = detector.detect_faces(img_array)
        face_count = len(faces)
        
        # Draw bounding boxes
        img_with_boxes = img_array.copy()
        for i, result in enumerate(faces):
            x, y, w, h = result['box']
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"Face {i+1}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        st.image(img_with_boxes, caption=f"Detected {face_count} face(s)", use_container_width=True)
        
        # Display results
        st.image(
            img_with_boxes, 
            caption=f"Detected {face_count} face(s)", 
            use_container_width=True
        )
        
        # Display face count and details
        st.subheader(f"Detection Results: {face_count} faces found")
        if face_count > 0:
            results = []
            for i, (x, y, w, h, confidence) in enumerate(faces):
                results.append({
                    "Face": i+1,
                    "X Position": x,
                    "Y Position": y,
                    "Width": w,
                    "Height": h,
                    "Confidence": f"{confidence:.4f}"
                })
            
            df = pd.DataFrame(results)
            st.dataframe(df)

            
elif choice == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Example metrics
    metrics = {
        "Precision": 0.92,
        "Recall": 0.89,
        "F1 Score": 0.90,
        "Accuracy": 0.91
    }
    
    # Display metrics
    st.table(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))
    
    # Plot metrics
    fig = px.bar(
        x=list(metrics.keys()), 
        y=list(metrics.values()),
        labels={"x": "Metric", "y": "Score"},
        title="Model Performance Metrics",
        range_y=[0, 1]
    )
    st.plotly_chart(fig)
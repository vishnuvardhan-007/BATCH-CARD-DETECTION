import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("BatchCardModel.h5")

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model's input size
    image = img_to_array(image)      # Convert to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0           # Normalize pixel values
    return image

# Define a function to make a prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Output is a single probability
    # Flip the interpretation
    return "Batch Card Present" if prediction < 0.5 else "No Batch Card Detected"

# Streamlit interface
st.title("Batch Card Detection")
st.write("Upload an image to check if the batch card is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    # Make prediction
    result = predict(image)
    st.write(f"Prediction: **{result}**")

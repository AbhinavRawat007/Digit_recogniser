import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

# UI
st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a 28x28 handwritten digit image (or draw one).")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = ImageOps.invert(image)  # MNIST background is black
    image = image.resize((28, 28))
    
    st.image(image, caption="Processed Image", width=150)

    img_array = np.array(image).reshape(1, 784) / 255.0
    prediction = np.argmax(model.predict(img_array), axis=1)[0]

    st.subheader(f"🔢 Predicted Digit: {prediction}")

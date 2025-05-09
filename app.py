import streamlit as st
import numpy as np
import cv2

# Title for the Streamlit app
st.title("Test OpenCV (Headless) and NumPy")

# Display NumPy and OpenCV versions
st.write("NumPy version:", np.__version__)
st.write("OpenCV version:", cv2.__version__)

# Create an example array and display it as an image
arr = np.zeros((300, 300), dtype=np.uint8)  # Create a black image (300x300)
st.image(arr, caption='Test Image', use_column_width=True)

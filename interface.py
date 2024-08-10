import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('spar.jpeg')  # call the function with the image file path


# Load the model
model = load_model('/Users/ioannespapadakes/Desktop/Image_classification/best_model.keras')

# List of categories
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon']

# Image dimensions
img_height = 180
img_width = 180

# Layout
st.title('Fruit and Vegetable Detector')
st.markdown("""
This tool helps you identify various fruits and vegetables from images.
Please enter the name of the image file you would like to analyze.
""")

# User input
image_path = st.text_input('Enter image file name', 'leonidas.jpg')

# Load and preprocess the image
if image_path:
    image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    prediction = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    # Display results
    st.image(image, caption='Uploaded Image', width=300)
    st.subheader('Prediction Results')
    st.write(f'**Identified:** {prediction}')
    st.write(f'**Confidence:** {confidence:.2f}%')
else:
    st.write("Please enter a valid image file name.")


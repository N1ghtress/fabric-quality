import streamlit as st
import requests
import json
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="ML Projet | OD Textile",
    layout="wide",
)

# TODO: Favicon
'''
# Master 2 DS Machine Learning 1 Project 2023/2024
## Anomaly detection within textile images
*Victor Dupriez, Marc Brustolin, Victor Faure* 
'''

def load_image(image_file):
    img = Image.open(image_file)
    img = ImageOps.grayscale(img)
    return img

def feedback_click():
    report = {
        "data": np.array(img).flatten().tolist(),
        "y_true": feed_input,
        "y_pred": int(prediction),
    }
    json_report = json.dumps(report)
    response = requests.post('http://serving-api:8080/feedback', json_report)
    st.write(response.content)

# Prediction
st.header("Send data and to predict !")
pred_left, pred_right = st.columns(2)
image_file = pred_left.file_uploader("Upload a file: ", type=['jpeg'])
pred_box = pred_right.empty()
pred_box.text_input("Prediction", disabled=True)
if image_file:
    img = load_image(image_file)
    st.image(img)
    data = {
        "data": np.array(img).flatten().tolist()
    }
    json_data = json.dumps(data)
    x = requests.post('http://serving-api:8080/predict', json_data)
    prediction = pred_box.text_input("Prediction", value=int(x.content), disabled=True)

st.divider()

# Feedback 
st.header("Send feedback on our prediciton !")
feed_left, feed_right = st.columns(2)
if image_file:
    feed_input = feed_left.number_input("Enter real class: ", min_value=0, max_value=1)
    feed_button = feed_right.button("Send", on_click=feedback_click)
else:
    feed_input = feed_left.number_input("Enter real class: ", disabled=True, min_value=0, max_value=1)
    feed_button = feed_right.button("Send", disabled=True, on_click=feedback_click)

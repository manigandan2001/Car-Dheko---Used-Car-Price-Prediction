import json

import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie

st.set_page_config(
    page_title="Homepage",
    page_icon="🏠",
)

st.markdown(
    """
    <h1 style='text-align: center;'>
        <span style='color: #800080;'>Car Dheko - Used Car Price Prediction
    </h1>
    """, 
    unsafe_allow_html=True
)
st.sidebar.success("Select a page from above")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://lottie.host/6bcf218f-8907-46dc-b3eb-d2f50c7e3814/V6W7rwRHRe.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=None,
    width=None,
    key= "WELCOME",
)
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>
        Welcome to the Homepage of 
        <span style='color: #800080;'>Car Dheko</span> 
        <span style='color: lightblue;'>Used Car Price Predictio</span>
    </h1>
    """, 
    unsafe_allow_html=True
)
st.write("Imagine you are working as a data scientist in Car Dheko, your aim is to enhance the customer experience and streamline the pricing process by leveraging machine learning. You need to create an accurate and user-friendly streamlit tool that predicts the prices of used cars based on various features. This tool should be deployed as an interactive web application for both customers and sales representatives to use seamlessly..")
st.markdown("<span style='color: cyan;'>Result:</span>", unsafe_allow_html=True)
st.write("""We have historical data on used car prices from CarDekho, including various features such as make, model, year, fuel type, transmission type, and other relevant attributes from different cities. Your task as a data scientist is to develop a machine learning model that can accurately predict the prices of used cars based on these features. The model should be integrated into a Streamlit-based web application to allow users to input car details and receive an estimated price instantly.""")
st.markdown("<span style='color: cyan;'>Domain:</span> <span style='color: white;'>Automotive Industry , Data Science, Machine Learning</span>", unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://lottie.host/848871b3-eeda-4854-9bc8-7616e2ffe678/f86GBVkM7M.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low")
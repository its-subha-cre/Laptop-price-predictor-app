import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load the model and data
pipe = load('model2.joblib')
df = load('df2.joblib')

st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())
# ppi =st.selectbox('PPI',df['ppi'].unique())

if st.button('Predict Price'):
    # Convert touchscreen and ips to binary
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi_val = ((X_res**2 + Y_res**2)**0.5) / screen_size


    # Create input DataFrame for prediction
    query_dict = {
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi':ppi_val,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }

    query_df = pd.DataFrame([query_dict])

    # Predict price
    predicted_price = np.exp(pipe.predict(query_df)[0])

    st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")

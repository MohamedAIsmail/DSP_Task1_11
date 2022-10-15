from turtle import title
from scipy import signal
import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

import streamlit as st  # 🎈 data web app development
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.set_page_config(page_title="Sampling Studio",
                   page_icon=":bar_chart:", layout="wide")


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header("Signal Studio")

# ----------------------- Reading data and plotting it ------------------------------
with st.container():
  
    uploaded_Signal = st.file_uploader('Upload your Signal here!')

    if uploaded_Signal:
        df = pd.read_csv(uploaded_Signal)
        Time = df['Time (s)']
        Amplitude = df['Voltage (mV)']

        max_Time = df['Time (s)'].max()
        min_Time = df['Time (s)'].min()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Time, y=Amplitude, mode='lines'))

# ----------------------- Calculating Sampled frequency according to the points chosen -----------------------------

        SamplingRate = st.slider('Choose desired Sampling points', 0, 200)
        SampledAmp = signal.resample(Amplitude, SamplingRate)
        SampledFreq = np.linspace(min_Time, max_Time, SamplingRate)

        fig.add_trace(go.Scatter(
            x=SampledFreq, y=SampledAmp, mode='markers', line=dict(color='firebrick')))

        fig.update_layout()
        st.write(fig)

# ----------------------- Plotting Sampled Signal ------------------------------
        if st.button('Show Sampled Graph'):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
            x=SampledFreq, y=SampledAmp, mode='lines', line=dict(color='firebrick')))
            st.write(fig2)
            



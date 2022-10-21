from asyncore import read
import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard
import streamlit as st
import Functions as fn
import numpy as np


st.set_page_config(page_title="Sampling Studio",
                   page_icon=":bar_chart:", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Signal Studio")


signals = []


options = st.sidebar.radio(
    'Pages', options=['Signal Reconstructing', 'Signal Composer'])


# Navigation
# --------------------------- Signal Uploading - Plotting - Reconstructing --------------------------------

if options == 'Signal Reconstructing':
    uploaded_Signal = st.file_uploader('Upload your Signal here!')

    if uploaded_Signal:
        samplingRate = st.slider('Choose desired Sampling frequency',
                                 min_value=1, max_value=100, step=1, key='samplingFrequency')
        SignalFile = fn.read_file(uploaded_Signal)
        fn.SignalPlotting(SignalFile.iloc[:, 0].to_numpy(
        ), SignalFile.iloc[:, 1].to_numpy(), samplingRate)

        SNR = st.slider('SNR (dBw)', 0.01, 100.0, 20.0, key='SNRValue')
        fn.addNoise(SignalFile.iloc[:, 0].to_numpy(
        ), SignalFile.iloc[:, 1].to_numpy(), SNR)


if options == 'Signal Composer':

    col1, col2 = st.columns([3, 1])
    with col2:
        with st.form('sinusode composer'):
            t = np.linspace(0, 6.5*np.pi, 200)
            freq = st.number_input(
                'frequency', min_value=0.0, max_value=60.0, step=1.0)
            amp = st.number_input('amplitude', step=1.0)
            phi = st.number_input('phase', min_value=-
                                  2 * np.pi, max_value=2 * np.pi, step=1.0)
            view = st.form_submit_button('view')
            addsig = st.form_submit_button('add')

    with col1:
        if view:
            sig = amp * np.sin(2 * np.pi * freq * t + phi)
            fn.signalcomposer(sig, t)

        if addsig:
            signals.extend([amp, freq, phi])
            fn.summedsignal(signals, t)

from asyncore import read
from audioop import add
import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard
import streamlit as st
import Functions as fn
import numpy as np


st.set_page_config(page_title="Sampling Studio",
                   page_icon="sampling_studio.png", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# ---------------------- view selected signal-------------------------------------------

def view_selected_signal():
    if signalselect != None:
        sigpar = fn.findsig(signalselect)
        sig = sigpar[0] * np.sin(2 * np.pi * sigpar[1] * t )
        return sig


st.image("sampling_studio.png")


signals = []


options = st.sidebar.radio(
    'Pages', options=['Signal Reconstructing', 'Signal Composer'])


# Navigation
# --------------------------- Signal Uploading - Plotting - Reconstructing --------------------------------

if options == 'Signal Reconstructing':
    uploaded_Signal = st.file_uploader('Upload your Signal here!')
    samplingRate = st.sidebar.slider('Sampling Frequency (Hz)',
                                     min_value=1, max_value=100, step=1, key='samplingFrequency')
    SNR = st.sidebar.slider('SNR (dBw)', 0.01, 100.0, 20.0, key='SNRValue')

    if uploaded_Signal:

        SignalFile = fn.read_file(uploaded_Signal)
        fn.SignalPlotting(SignalFile.iloc[:, 1].to_numpy(
        ), SignalFile.iloc[:, 2].to_numpy(), samplingRate)

        fn.addNoise(SignalFile.iloc[:, 1].to_numpy(
        ), SignalFile.iloc[:, 2].to_numpy(), SNR)


if options == 'Signal Composer':

    if 'sigparameters' not in st.session_state:
        st.session_state['sigparameters'] = []
    if 'a_count' not in st.session_state:
        st.session_state['a_count'] = 0

    col1, col2 = st.columns(2)
    t = []
    sig = []

    t = np.linspace(-2, 2, 1000)
    freq = st.sidebar.number_input(
        'Frequency', min_value=0.0, max_value=60.0, step=1.0)
    amp = st.sidebar.number_input('Amplitude', step=1.0)
    
    sig = amp * np.sin(2 * np.pi * freq * t)

    viewText = 'Signal Viewer'

    addsig = st.sidebar.button('Add Signal')
    addText = 'Added Signals'

    selectedSignalText = 'Signal Selected'

    if addsig:
        st.session_state.a_count += 1
        name = 'Signal ' + str(st.session_state.a_count)
        signal = [amp, freq, name]
        st.session_state.sigparameters.append(signal)

    slct = []
    for i in range(len(st.session_state.sigparameters)):
        slct.append(st.session_state.sigparameters[i][2])

    signalselect = st.sidebar.selectbox(
        'Select a signal', slct, on_change=view_selected_signal)
    viewSelectedSignalText = 'Selected Signal'

    if (signalselect != None):
        signal_csv = fn.download_csv_file(t, fn.summedsignal(
            t), 'Time (s)', 'Voltage (V)')
        st.sidebar.download_button('Download Composed Signal File', signal_csv,
                                   'Composed Signal.csv')
        deletesig = st.sidebar.button(
            'Delete', on_click=fn.handle_click, args=(signalselect,))

    fn.Plotting(t, fn.summedsignal(t), addText, '#0fb7bd')

    with col1:
        fn.Plotting(t, sig, viewText, '#0fb7bd')

    with col2:
        if (signalselect != None):
            fn.Plotting(t, view_selected_signal(),
                        selectedSignalText, '#0fb7bd')
        else:
            fn.Plotting([], [], selectedSignalText, '#0fb7bd')

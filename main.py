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


# ---------------------- view selected signal-------------------------------------------

def view_selected_signal():
    if signalselect != None:
        sigpar = fn.findsig(signalselect)
        sig = sigpar[0] * np.sin(2 * np.pi * sigpar[1] * t + sigpar[2])


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

    if 'sigparameters' not in st.session_state:
        st.session_state['sigparameters'] = []
    if 'a_count' not in st.session_state:
        st.session_state['a_count'] = 0

    col1, col2, col3 = st.columns(3)

    with col3:
        with st.form('Signal Generator'):
            t = np.linspace(-2, 2, 10000)
            freq = st.number_input(
                'frequency', min_value=0.0, max_value=60.0, step=1.0)
            amp = st.number_input('amplitude', step=1.0)
            phi = st.number_input(
                'phase', min_value=-2 * np.pi, max_value=2 * np.pi, step=np.pi, value=0.0)
            view = st.form_submit_button('view')
            addsig = st.form_submit_button('add')

    with col1:
        if view:
            text = 'View Signal'
            sig = amp * np.sin(2 * np.pi * freq * t + phi)
            fn.Plotting(t, sig, text)

    with col2:
        if addsig:
            text = 'Add Signal'
            st.session_state.a_count += 1
            name = 'signal ' + str(st.session_state.a_count)
            signal = [amp, freq, phi, name]
            st.session_state.sigparameters.append(signal)
            
            fn.Plotting(t, fn.summedsignal(t), text)



    with col3:

        slct = []
        for i in range(len(st.session_state.sigparameters)):
            slct.append(st.session_state.sigparameters[i][3])

        signalselect = st.selectbox(
            'select a signal', slct, on_change=view_selected_signal)

        deletesig = st.button(
            'delete', on_click=fn.handle_click, args=(signalselect,))

    with col1:

        view_selected_signal()

        # if deletesig:
        #     if signalselect != None:

        # if (len(st.session_state.sigparameters)):
        #     ax[1].plot(t, fn.summedsignal(t))

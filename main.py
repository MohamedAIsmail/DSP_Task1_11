from asyncore import read
from audioop import add
import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard
import streamlit as st
import Functions as fn
import numpy as np
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Sampling Studio",
                   page_icon="sampling_studio.png", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if 'sigparameters' not in st.session_state:
    st.session_state['sigparameters'] = []
if 'a_count' not in st.session_state:
    st.session_state['a_count'] = 0


# ---------------------- view selected signal-------------------------------------------
def view_selected_signal():
    if signalselect != None:
        sigpar = fn.findsig(signalselect)
        sig = sigpar[0] * np.sin(2 * np.pi * sigpar[1] * composedT)
        return sig


# st.image("sampling_studio.png")


signals = []

option = option_menu(
    None, ["Uploading Signal", 'Generating Signal'], orientation="horizontal", default_index=0)

col1, col2, col3 = st.columns([1, 4, 0.7])

# with col1:
#     option = st.radio('Choose your option',
#                       ('Uploading Signal', 'Generating Signal'))


with col1:
    
    if option == 'Uploading Signal':
        st.header(" ")
        st.header(" ")
        uploaded_Signal = st.file_uploader(
            'Upload your Signal here!')  # Upload Signal here

    if option == 'Generating Signal':
        st.header(" ")
        st.header(" ")
        st.header('Adding Signal')
# Parameters for composed signal
        composedT = []
        composedSig = []
        composedT = np.linspace(0, 8, 1000)
        freq = st.number_input(
            'Frequency', min_value=0.0, max_value=60.0, step=1.0)
        amp = st.number_input('Amplitude', step=1.0)
        composedSig = amp * np.sin(2 * np.pi * freq * composedT)

    # Button for adding signal to the session state
        addsig = st.button('Add Signal')
        if (addsig):
            st.session_state.a_count += 1
            name = 'Signal ' + str(st.session_state.a_count)
            signal = [amp, freq, name]
            st.session_state.sigparameters.append(signal)

        slct = []
        for i in range(len(st.session_state.sigparameters)):
            slct.append(st.session_state.sigparameters[i][2])

    # Select box for selecting the signal to view it and delete it
        signalselect = st.selectbox(
            'Select a signal', slct, on_change=view_selected_signal)

        if (signalselect != None):
            signal_csv = fn.download_csv_file(composedT, fn.summedsignal(
                composedT), 'Time (s)', 'Voltage (V)')
            st.download_button('Download Composed Signal File', signal_csv,
                            'Composed Signal.csv')
            deletesig = st.button(
                'Delete', on_click=fn.handle_click, args=(signalselect,))


#####################################################################################################
with col3:
    if option == 'Uploading Signal':
        st.header(" ")
        st.header(" ")
        st.header('View')
        showUploadedSignal = st.checkbox('Uploaded Signal', value=True)
        showReconstructedSignal = st.checkbox('Reconstructed Signal')
        showSamplingPoints = st.checkbox('Sampling Points')
        ShowNoise = st.checkbox('Show Noise')
        SNR = 150
        samplingRate = 1

        if (showReconstructedSignal or showSamplingPoints):
            samplingRate = st.slider('Sampling Frequency (Hz)',
                                     min_value=1, max_value=100, step=1, key='samplingFrequency')
        if (ShowNoise):
            SNR = st.slider('SNR (dBw)', 0.01, 100.0,
                            20.0, step=0.5, key='SNRValue')

    if option == 'Generating Signal':
        # Sliders to take values of sampling frequency and SNR
        st.header(" ")
        st.header(" ")
        st.header('View')
        showSelectedSignal = st.checkbox('Selected Signal', value=True)
        showReconstructedSignal = st.checkbox('Reconstructed Signal')
        showSamplingPoints = st.checkbox('Sampling Points')
        showComposedSignals = st.checkbox('Added Signals')
        ShowNoise = st.checkbox('Show Noise')
        SNR = 150
        samplingRate = 1

        if (showReconstructedSignal or showSamplingPoints):
            samplingRate = st.slider('Sampling Frequency (Hz)',
                                     min_value=1, max_value=100, step=1, key='samplingFrequency')
        if (ShowNoise):
            SNR = st.slider('SNR (dBw)', 0.01, 100.0,
                            20.0, step=0.5, key='SNRValue')


with col2:
    if option == 'Uploading Signal':
        if uploaded_Signal:
            # Reading Csv file into a dataframe called SignalFile
            SignalFile = fn.read_file(uploaded_Signal)
            timeReadings = SignalFile.iloc[:, 1].to_numpy()
            ampltiudeReadings = SignalFile.iloc[:, 2].to_numpy()

            # Reading data according to the columns and plotting them, and also reconstruct according to the sampleRate
            fn.UploadedSignal(timeReadings, ampltiudeReadings,
                              samplingRate, ShowNoise, showReconstructedSignal, showUploadedSignal, showSamplingPoints, SNR)
        else:
            # Plotting empty plots for GUI incase there is no input
            fn.Plotting([], [], 'Main Viewer', '#0fb7bd')

    if option == 'Generating Signal':
        if (st.session_state.sigparameters != []):
            signal = view_selected_signal()
            fn.GeneratedSignal(composedT, composedSig, samplingRate, ShowNoise,
                               showReconstructedSignal, showSelectedSignal, showComposedSignals, showSamplingPoints, SNR, signal)
        else:
            st.warning("You must add signal first!")


with col3:
    if option == 'Uploading Signal':
        if uploaded_Signal:
            maxFrequency = st.metric(
                "Maximum Frequency", str(fn.GetMaximumFrequencyComponent(timeReadings, ampltiudeReadings)) + ' Hz')  # Viewing Fmax


# # Plotting all the graphs in the Composed Signal Page
#     fn.Plotting(t, fn.summedsignal(t) '#0fb7bd')

#     with col1:
#         fn.Plotting(t, sig '#0fb7bd')

#     with col2:
#         if (signalselect != None):
#             fn.Plotting(t, view_selected_signal(),
#                          '#0fb7bd')
#         else:
#             fn.Plotting([], [], selectedSignalText, '#0fb7bd')

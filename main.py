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
        sig = sigpar[0] * np.sin(2 * np.pi * sigpar[1] * t)
        return sig


st.image("sampling_studio.png")  # Our website Logo


signals = []  # Creating an empty list called signals to store the signals created in it

uploaded_Signal = st.sidebar.file_uploader(
    'Upload your Signal here!')  # Upload Signal here

options = st.sidebar.radio(
    'Pages', options=['Signal Reconstructing', 'Signal Composer'])  # radio buttons to swap the pages!


# Navigation
# --------------------------- Signal Uploading - Plotting - Reconstructing --------------------------------

if options == 'Signal Reconstructing':

    # Sliders to take values of sampling frequency and SNR
    samplingRate = st.sidebar.slider('Sampling Frequency (Hz)',
                                     min_value=1, max_value=100, step=1, key='samplingFrequency')
    SNR = st.sidebar.slider('SNR (dBw)', 0.01, 100.0,
                            20.0, step=0.5, key='SNRValue')
    AddNoise = st.sidebar.checkbox('Add Noise')

    if uploaded_Signal:
        # Reading Csv file into a dataframe called SignalFile
        SignalFile = fn.read_file(uploaded_Signal)
        timeReadings = SignalFile.iloc[:, 1].to_numpy()
        ampltiudeReadings = SignalFile.iloc[:, 2].to_numpy()

        # Reading data according to the columns and plotting them, and also reconstruct according to the sampleRate
        fn.SignalPlotting(timeReadings, ampltiudeReadings,
                          samplingRate, AddNoise, SNR)
        
        maxFrequency = st.sidebar.metric(
            "Maximum Frequency", str(fn.GetMaximumFrequencyComponent(SignalFile.iloc[:, 1].to_numpy(
            ), SignalFile.iloc[:, 2].to_numpy())) + ' Hz')  # Viewing Fmax

        Nyquistfreq = st.sidebar.metric(
            "Nyquist Frequency","Fs ≥ " + str(2 * fn.GetMaximumFrequencyComponent(SignalFile.iloc[:, 1].to_numpy(
            ), SignalFile.iloc[:, 2].to_numpy())) + ' Hz')  # Viewing Fs
    else:
        # Plotting empty plots for GUI incase there is no input
        fn.Plotting([], [], 'Signal Plot', '#0fb7bd')

        fn.Plotting([], [], 'Reconstructed Plot', '#0fb7bd')

if options == 'Signal Composer':

    # Here we are using session_state in streamlit to create a counter and a list, counter for counting the signals added
    if 'sigparameters' not in st.session_state:
        # list for containing the parameters of the signal such as amplitude and freq
        st.session_state['sigparameters'] = []
    if 'a_count' not in st.session_state:
        st.session_state['a_count'] = 0

    col1, col2 = st.columns(2)  # GUI Part
    t = []
    sig = []

# Parameters for composed signal
    t = np.linspace(-2, 2, 1000)
    freq = st.sidebar.number_input(
        'Frequency', min_value=0.0, max_value=60.0, step=1.0)
    amp = st.sidebar.number_input('Amplitude', step=1.0)
    sig = amp * np.sin(2 * np.pi * freq * t)

    viewText = 'Signal Viewer'  # GUI
    addText = 'Added Signals'
    selectedSignalText = 'Selected Signal'

# Button for adding signal to the session state
    addsig = st.sidebar.button('Add Signal')
    if addsig:
        st.session_state.a_count += 1
        name = 'Signal ' + str(st.session_state.a_count)
        signal = [amp, freq, name]
        st.session_state.sigparameters.append(signal)

    slct = []
    for i in range(len(st.session_state.sigparameters)):
        slct.append(st.session_state.sigparameters[i][2])

# Select box for selecting the signal to view it and delete it
    signalselect = st.sidebar.selectbox(
        'Select a signal', slct, on_change=view_selected_signal)
    viewSelectedSignalText = 'Selected Signal'

# Download and delte buttons for downloading the summed signal and deleting a specific signal
    if (signalselect != None):
        signal_csv = fn.download_csv_file(t, fn.summedsignal(
            t), 'Time (s)', 'Voltage (V)')
        st.sidebar.download_button('Download Composed Signal File', signal_csv,
                                   'Composed Signal.csv')
        deletesig = st.sidebar.button(
            'Delete', on_click=fn.handle_click, args=(signalselect,))

# Plotting all the graphs in the Composed Signal Page
    fn.Plotting(t, fn.summedsignal(t), addText, '#0fb7bd')

    with col1:
        fn.Plotting(t, sig, viewText, '#0fb7bd')

    with col2:
        if (signalselect != None):
            fn.Plotting(t, view_selected_signal(),
                        selectedSignalText, '#0fb7bd')
        else:
            fn.Plotting([], [], selectedSignalText, '#0fb7bd')

import streamlit as st
import Functions as fn
import numpy as np
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Sampling Studio",
                   page_icon="sampling_studio.png", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if 'sigparameters' not in st.session_state:
    st.session_state['sigparameters'] = [[1, 1, 'Signal 1']]
if 'a_count' not in st.session_state:
    st.session_state['a_count'] = 1

option = option_menu(
    None, ["Uploading Signal", 'Generating Signal'], orientation="horizontal", default_index=0)

col1, col2, col3 = st.columns([1, 4, 0.7])


with col1:

    if option == 'Uploading Signal':
        st.header(" ")
        uploaded_Signal = st.file_uploader(
            'Upload your Signal here!')  # Upload Signal here

    if option == 'Generating Signal':
        st.header(" ")
        st.header('Adding Signal')

# Parameters for composed signal
        composedT = np.linspace(0, 6, 1000)
        freq = st.number_input(
            'Frequency', min_value=0, max_value=60, step=1, value=1)
        amp = st.number_input('Amplitude', min_value=0, step=1, value=1)

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
            'Select a signal', slct)

        if (signalselect != None):
            signalparamters = fn.findsig(signalselect)
            st.write('Frequency : ', signalparamters[1], ' Hz')
            st.write('Amplitude : ', signalparamters[0], ' mV')
            deletesig = st.button(
                'Delete', on_click=fn.handle_click, args=(signalselect,))


#####################################################################################################
with col3:
    if option == 'Uploading Signal':
        st.header(" ")
        st.header('View')
        showReconstructedSignal = st.checkbox('Reconstructed Signal')
        showSamplingPoints = st.checkbox('Sampling Points')
        ShowNoise = st.checkbox('Add Noise')
        SNR = 150
        samplingRate = 1

        if (showReconstructedSignal or showSamplingPoints):
            samplingRate = st.slider('Sampling Frequency (Hz)',
                                     min_value=1, max_value=100, step=1, key='samplingFrequency')
        if (ShowNoise):
            SNR = st.slider('SNR (dBw)', 1, 100,
                            20, step=1, key='SNRValue')

    if option == 'Generating Signal':
        # Sliders to take values of sampling frequency and SNR
        st.header(" ")
        st.header('View')
        showReconstructedSignal = st.checkbox('Reconstructed Signal')
        showSamplingPoints = st.checkbox('Sampling Points')
        ShowNoise = st.checkbox('Add Noise')
        SNR = 150
        samplingRate = 1

        if (showReconstructedSignal or showSamplingPoints):
            samplingRate = st.slider('Sampling Frequency (Hz)',
                                     min_value=1, max_value=100, step=1, key='samplingFrequency')
        if (ShowNoise):
            SNR = st.slider('SNR (dBw)', 1, 100,
                            20, step=1, key='SNRValue')


with col2:
    if option == 'Uploading Signal':
        if uploaded_Signal:
            # Reading Csv file into a dataframe called SignalFile
            SignalFile = fn.read_file(uploaded_Signal)
            timeReadings = SignalFile.iloc[:, 1].to_numpy()
            ampltiudeReadings = SignalFile.iloc[:, 2].to_numpy()

            # Reading data according to the columns and plotting them, and also reconstruct according to the sampleRate
            reconstructedsignal_csv = fn.UploadedSignal(timeReadings, ampltiudeReadings,
                                                        samplingRate, ShowNoise, showReconstructedSignal, showSamplingPoints, SNR)
            with col3:
                st.download_button('Download Reconstructed Signal', reconstructedsignal_csv,
                                   'Reconstructed Signal.csv')

        else:
            # Plotting empty plots for GUI incase there is no input
            with open('Automated Sin Signal.csv', mode='r') as file:
                SignalFile = fn.read_file(file)
                timeReadings = SignalFile.iloc[:, 1].to_numpy()
                ampltiudeReadings = SignalFile.iloc[:, 2].to_numpy()
                reconstructedsignal_csv = fn.UploadedSignal(timeReadings, ampltiudeReadings,
                                                            samplingRate, ShowNoise, showReconstructedSignal, showSamplingPoints, SNR)
                with col3:
                    st.download_button('Download Reconstructed Signal', reconstructedsignal_csv,
                                       'Reconstructed Signal.csv')

    if option == 'Generating Signal':
        if (st.session_state.sigparameters != []):
            fn.GeneratedSignal(composedT, samplingRate, ShowNoise,
                               showReconstructedSignal, showSamplingPoints, SNR)
            if (signalselect != None):
                signal_csv = fn.download_csv_file(composedT, fn.summedsignal(
                    composedT), 'Time (s)', 'Voltage (V)')
                with col3:
                    st.download_button('Download Composed Signal', signal_csv,
                                       'Composed Signal.csv')
        else:
            st.warning("You must add signal first!")


with col1:
    if option == 'Uploading Signal':
        if uploaded_Signal:
            maxFrequency = st.metric(
                "Maximum Frequency", str(fn.GetMaximumFrequencyComponent(timeReadings, ampltiudeReadings)) + ' Hz')  # Viewing Fmax
        else:
            maxFrequency = st.metric(
                "Maximum Frequency", str('1 Hz'))  # Viewing Fmax

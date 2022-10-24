from select import select
from time import time
from turtle import title
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st  # 🎈 data web app development
import pandas as pd  # read csv, df manipulation
import scipy.fft
import numpy as np  # np mean, np random
from scipy.signal import find_peaks


# Transforming Signal to Frequency Domain to Capture Value of Maximum Frequency


def GetMaximumFrequencyComponent(time, amplitudes):

    # abs(scipy.fft.rfft()) returns the magnitude of the amplitude of each frequency component in frequency domain
    magnitudes = np.abs(scipy.fft.rfft(amplitudes))

    # scipy.fft.rfftfrec(Window length, Sample spacing)  returns a list containing the signal frequency components
    frequencies = scipy.fft.rfftfreq(len(time), (time[1] - time[0]))

    indices = find_peaks(magnitudes)[0]  # the indices of the peaks magnitudes

    maxfreq = 0
    for i in range(len(indices)):
        if (frequencies[indices[i]] > maxfreq):
            maxfreq = frequencies[indices[i]]
    return round(maxfreq)


# ----------------------- Function of plotting the Signal Resampling ------------------------------

def signalSampling(Amplitude, Time, sampleFreq, timeRange):

    PointSteps = int((len(Time)/timeRange)/(sampleFreq))

    if PointSteps == 0:
        PointSteps = 1

    sampledTime = Time[::PointSteps]
    sampledAmplitude = Amplitude[::PointSteps]

    return sampledAmplitude, sampledTime

# ----------------------- Function of adding noise ------------------------------


def addNoise(amplitudeReadings, snr_db):

    power_watt = amplitudeReadings**2
    power_avg_watt = np.mean(power_watt)
    power_avg_db = 10 * np.log10(power_avg_watt)
    noise_power_avg_db = power_avg_db - snr_db

    # convert P(dB) => P(watt)
    noise_power_avg_watts = 10 ** (noise_power_avg_db / 10)

    # Generate an sample of white noise
    noise_mean = 0

    noise_amplitudes = np.random.normal(
        noise_mean, np.sqrt(noise_power_avg_watts), len(power_watt))  # random samples from a normal (Gaussian) distribution.

    # adding noise to the original signal
    signal_with_noise = amplitudeReadings + noise_amplitudes

    return signal_with_noise

# ----------------------- Function of Reconstructing the Signal ------------------------------


def signalReconstructing(time_Points, sampledTime, sampledAmplitude):

    # Matrix containing all Timepoints
    TimeMatrix = np.resize(time_Points, (len(sampledTime), len(time_Points)))

    # The following equations is according to White- Shannon interpoltion formula ((t - nT)/T)
    # Transpose for TimeMatrix is a must for proper calculations (broadcasting)
    K = (TimeMatrix.T - sampledTime) / (sampledTime[1] - sampledTime[0])
    # Reconstructed Amplitude = x[n] sinc(v) -- Whitetaker Shannon
    finalMatrix = sampledAmplitude * np.sinc(K)

    # Summation of columns of the final matrix to get an array of reconstructed points
    ReconstructedSignal = np.sum(finalMatrix, axis=1)

    return ReconstructedSignal


# ----------------------- Function of plotting the composed signal ------------------------------

def signalcomposer(sig, t):
    # plotanysig(t,sig,'the composed signal')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sig,
                             mode='lines'))
    st.plotly_chart(fig)


# ----------------------- Function of plotting the summed signal ------------------------------

def summedsignal(t):
    ysum = 0
    if (st.session_state.sigparameters == []):
        return st.session_state.sigparameters
    else:
        for i in range(len(st.session_state.sigparameters)):
            ysum += st.session_state.sigparameters[i][0] * np.sin(
                2 * np.pi * st.session_state.sigparameters[i][1] * t)
        return ysum


# ----------------------- Function of plotting data and its reconstruction from file ------------------------------

def UploadedSignal(timeReadings, amplitudeReadings, samplingRate, AddNoiseCheckBox, showReconstructedSignal, showUploadedSignal, showSamplingPoints, snr_db
                   ):

    timeRange_max = max(timeReadings)
    timeRange_min = min(timeReadings)
    timeRange = timeRange_max - timeRange_min
    sampledAmplitude, sampledTime = signalSampling(
        amplitudeReadings, timeReadings, samplingRate, timeRange)

    fig = go.Figure()

    if (showUploadedSignal):
        fig.add_trace(go.Scatter(x=timeReadings, y=amplitudeReadings,
                                 mode='lines', name='Signal Plot', marker_color='#0fb7bd', line=dict(width=6)))
        sampledAmplitude, sampledTime = signalSampling(
            amplitudeReadings, timeReadings, samplingRate, timeRange)

    if (AddNoiseCheckBox):
        signal_with_Noise = addNoise(amplitudeReadings, snr_db)
        fig.add_trace(go.Scatter(x=timeReadings, y=signal_with_Noise,
                                 mode='lines', name='Noise', marker_color='#FF0EF3', line=dict(width=2.5)))
        sampledAmplitude, sampledTime = signalSampling(
            signal_with_Noise, timeReadings, samplingRate, timeRange)

# Sampling points on signal
    if (showSamplingPoints):
        fig.add_trace(go.Scatter(x=sampledTime, y=sampledAmplitude,
                                 mode='markers', name='Sampling', marker_color='red', marker=dict(size=7)))

    # Reconstructing the signal then plotting it
    reconstructedAmp = signalReconstructing(
        timeReadings, sampledTime, sampledAmplitude)

    if (showReconstructedSignal):
        fig.add_trace(go.Scatter(x=timeReadings, y=reconstructedAmp,
                                 mode='lines', name='Reconstructed Signal', marker_color='#FFF01F ', line=dict(width=3)))

    fig.update_xaxes(zeroline=True,
                     zerolinewidth=2, range=[0, timeRange_max])
    fig.update_yaxes(zeroline=True, zerolinewidth=2)
    fig.update_layout(font=dict(size=13),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
                      width=1200,
                      height=600,
                      title={
                          'text': "Main Viewer",
                          'y': 0.9,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                      title_font=dict(
                          family="Arial",
                          size=20,
    ))
    st.plotly_chart(fig, use_container_width=True)


# ----------------------- Generating SIGNAL ------------------------------

def GeneratedSignal(ComposedT, ComposedSig, samplingRate, AddNoiseCheckBox, showReconstructedSignal, showSelectedSignal, showComposedSignals, showSamplingPoints, snr_db, selectedSignal
                    ):
    timeRange_max = max(ComposedT)
    timeRange_min = min(ComposedT)
    timeRange = timeRange_max - timeRange_min

    sampledAmplitude, sampledTime = signalSampling(
        ComposedSig, ComposedT, samplingRate, timeRange)

    composedSignal = summedsignal(ComposedT)
    # selectedSignal = st.session_state['selected signal'][0] * \
    #     np.sin(2 * np.pi * st.session_state['selected signal'] * ComposedT)

    fig = go.Figure()
    if (showSelectedSignal):
        fig.add_trace(go.Scatter(x=ComposedT, y=selectedSignal,
                                 mode='lines', name='Signal Generated', marker_color='#0fb7bd'))
        sampledAmplitude, sampledTime = signalSampling(
            selectedSignal, ComposedT, samplingRate, timeRange)

    if (showComposedSignals):
        fig.add_trace(go.Scatter(x=ComposedT, y=composedSignal,
                                 mode='lines', name='Signal Summation', marker_color='#0fb7bd'))
        sampledAmplitude, sampledTime = signalSampling(
            composedSignal, ComposedT, samplingRate, timeRange)

    if (AddNoiseCheckBox):
        signal_with_Noise = addNoise(composedSignal, snr_db)
        fig.add_trace(go.Scatter(x=ComposedT, y=signal_with_Noise,
                                 mode='lines', name='Noise', marker_color='yellow'))
        sampledAmplitude, sampledTime = signalSampling(
            signal_with_Noise, ComposedT, samplingRate, timeRange)

# Sampling points on signal
    if (showSamplingPoints):
        fig.add_trace(go.Scatter(x=sampledTime, y=sampledAmplitude,
                                 mode='markers', name='Sampling Points', marker_color='red'))

    # Reconstructing the signal then plotting it
    reconstructedAmp = signalReconstructing(
        ComposedT, sampledTime, sampledAmplitude)

    if (showReconstructedSignal):
        fig.add_trace(go.Scatter(x=ComposedT, y=reconstructedAmp,
                                 mode='lines', name='Reconstructed Signal', marker_color='green'))

    fig.update_xaxes(title_text="Time (s)", zeroline=True,
                     zerolinewidth=2, range=[0, timeRange_max], title_font={"size": 20})
    fig.update_yaxes(title_text="Amplitude (mV)",
                     zeroline=True, zerolinewidth=2, title_font={"size": 20})
    fig.update_layout(width=800,
                      height=600,
                      font=dict(size=13),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
                      title={
                          'text': "Main Viewer",
                          'y': 0.9,
                          'x': 0.49,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                      title_font=dict(
                          family="Arial",
                          size=20,
                      ))
    st.plotly_chart(fig, use_container_width=True)
# ----------------------- Function of converting any signal to CSV FILE ------------------------------


def convert_to_dataframe(timeReading, ampltiudeReading, par1_name, par2_name):
    signal = []
    if (ampltiudeReading != []):
        for i in range(len(timeReading)):
            signal.append([timeReading[i], ampltiudeReading[i]])
        return pd.DataFrame(signal, columns=[f'{par1_name}', f'{par2_name}'])
    else:
        return None


def download_csv_file(timeReading, ampltiudeReading, x_axis_label, y_axis_label):
    signal_analysis_table = convert_to_dataframe(
        timeReading, ampltiudeReading, x_axis_label, y_axis_label)
    if (signal_analysis_table.empty != True):
        signal_csv = signal_analysis_table.to_csv()
        return signal_csv


# ----------------------- find selected signal----------------------------------
def findsig(name):
    for i in range(len(st.session_state.sigparameters)):
        if name == st.session_state.sigparameters[i][2]:
            return st.session_state.sigparameters[i]


# ----------------------- delete selected signal----------------------------------

def delsig(name):
    for i in range(len(st.session_state.sigparameters)):
        if name == st.session_state.sigparameters[i][2]:
            return i


# ---------------------- handle click -------------------------------------------

def handle_click(name):
    if name != None:
        indx = delsig(name)
        st.session_state.sigparameters.remove(
            st.session_state.sigparameters[indx])

# ---------------------- Read the CSV file -------------------------------------------


def read_file(file):
    df = pd.read_csv(file)
    return df


# ----------------------- Function of reading data from file and plotting ------------------------------

def Plotting(time, Signal, plotHeader, colorGiv,):
    Fig = go.Figure()
    Fig.add_trace(go.Scatter(
        x=time, y=Signal, mode='lines', marker_color=colorGiv))
    Fig.update_xaxes(title_text="Time (s)", zeroline=True,
                     zerolinewidth=2, range=[0, 1])
    Fig.update_yaxes(title_text="Amplitude (mV)",
                     zeroline=True)
    Fig.update_layout(width=800,
                      height=600,
                      font=dict(size=13),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
                      title={
                          'text': plotHeader,
                          'y': 0.9,
                          'x': 0.49,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                      title_font=dict(
                          family="Arial",
                          size=20,
                      ))
    st.plotly_chart(Fig, use_container_width=True)

from time import time
from turtle import title
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st  # 🎈 data web app development
import pandas as pd  # read csv, df manipulation
import scipy.fft
import numpy as np  # np mean, np random


# Transforming Signal to Frequency Domain to Capture Value of Maximum Frequency

def GetMaximumFrequencyComponent(timeReadings, amplitudeReadings):
    magnitudes = np.abs(scipy.fft.rfft(amplitudeReadings)) / \
        np.max(np.abs(scipy.fft.rfft(amplitudeReadings)))
    frequencies = scipy.fft.rfftfreq(
        len(timeReadings), (timeReadings[1] - timeReadings[0]))
    for index, frequency in enumerate(frequencies):
        if magnitudes[index] >= 0.05:
            maximumFrequency = frequency
    return round(maximumFrequency)


# ----------------------- Function of plotting the Signal Resampling ------------------------------

def signalSampling(Amplitude, Time, sampleFreq, timeRange):

    PointSteps = int((len(Time)/timeRange)/(sampleFreq))

    if PointSteps == 0:
        PointSteps = 1

    sampledTime = Time[::PointSteps]
    sampledAmplitude = Amplitude[::PointSteps]

    return sampledAmplitude, sampledTime


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

def summedsignal(sig, t):
    ysum = 0
    for i in range(len(sig)):
        ysum += sig[i][0] * np.sin(2 * np.pi * sig[i][1] * t)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=ysum,
                              mode='lines'))
    st.plotly_chart(fig2)


# ----------------------- Function of plotting data and its reconstruction from file ------------------------------

def SignalPlotting(timeReadings, amplitudeReadings, samplingRate):
    
    timeRange_max = max(timeReadings)
    timeRange_min = min(timeReadings)
    timeRange = timeRange_max - timeRange_min

    sampledAmplitude, sampledTime = signalSampling(
        amplitudeReadings, timeReadings, samplingRate, timeRange)

    left_column, right_column = st.columns(2)

    with left_column:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeReadings, y=amplitudeReadings,
                                 mode='lines', name='Signal Plot', marker_color='#0fb7bd'))

        maxF = GetMaximumFrequencyComponent(timeReadings, amplitudeReadings)

    # Sampling points on signal
        fig.add_trace(go.Scatter(x=sampledTime, y=sampledAmplitude,
                                 mode='markers', name='Sampling'))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude (mV)")
        fig.update_layout(title={
            'text': "Signal Plot",
            'y': 0.9,
            'x': 0.49,
            'xanchor': 'center',
            'yanchor': 'top'},
            title_font=dict(
                family="Arial",
                size=20,
        ))
        st.plotly_chart(fig, use_container_width=True)

    if samplingRate > 0:
        with right_column:
            reconstructedAmp = signalReconstructing(
                timeReadings, sampledTime, sampledAmplitude)
            Plotting(timeReadings, reconstructedAmp,
                     "Reconstructed Plot", '#61c6bd')


# ----------------------- Function of reading data from file and plotting ------------------------------

def read_file(file):
    df = pd.read_csv(file)
    return df


def Plotting(time, Signal, plotHeader, colorGiv):
    Fig = go.Figure()
    Fig.add_trace(go.Scatter(
        x=time, y=Signal, mode='lines', marker_color=colorGiv))
    Fig.update_xaxes(title_text="Time (s)")
    Fig.update_yaxes(title_text="Amplitude (mV)")
    Fig.update_layout(title={
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


# ----------------------- Function of adding noise ------------------------------


def addNoise(timeReadings, amplitudeReadings, snr_db):
    power_watt = amplitudeReadings**2
    power_avg_watt = np.mean(power_watt)
    power_avg_db = 10 * np.log10(power_avg_watt)
    noise_power_avg_db = power_avg_db - snr_db

    # convert P(dB) => P(watt)
    noise_power_avg_watts = 10 ** (noise_power_avg_db / 10)

    # Generate an sample of white noise
    noise_mean = 0

    noise_volts = np.random.normal(
        noise_mean, np.sqrt(noise_power_avg_watts), len(power_watt))

    signal_with_noise = amplitudeReadings + noise_volts

    Plotting(timeReadings, signal_with_noise, 'Noise', '#0fb7bd')


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

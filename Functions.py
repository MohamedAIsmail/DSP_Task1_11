from time import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st  # 🎈 data web app development
import pandas as pd  # read csv, df manipulation
from array import array
from scipy import signal
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
    st.write(maximumFrequency)
    return round(maximumFrequency)


# ----------------------- Function of plotting the Signal Resampling ------------------------------

def signalSampling(Amplitude, Time, sample_freq, timeRange):

    sampleRate = int((len(Time)/timeRange)/(sample_freq))

    if sampleRate == 0:
        sampleRate = 1

    sampledTime = Time[::sampleRate]
    sampledAmplitude = Amplitude[::sampleRate]

    return sampledAmplitude, sampledTime


# ----------------------- Function of Reconstructing the Signal ------------------------------

def signalReconstructing(time_Points, sampledTime, sampledAmplitude):

    TimeMatrix = np.resize(time_Points, (len(sampledTime), len(time_Points))) # Matrix containing all Timepoints

    # The following equations is according to White- Shannon interpoltion formula ((t- nT)/T)
    K = (TimeMatrix.T - sampledTime) / (sampledTime[1] - sampledTime[0]) # Transpose for TimeMatrix is a must for proper calculations (broadcasting)
    
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
        ysum += sig[i][0] * np.sin(2 * np.pi * sig[i][1] * t + sig[i][2])
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
        st.header("Signal Plotting")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeReadings, y=amplitudeReadings,
                                mode='lines', name='Signal Plot'))


        # Sampling points on signal
        fig.add_trace(go.Scatter(x=sampledTime, y=sampledAmplitude,
                                mode='markers', name='Sampling'))
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude (mV)")
        st.plotly_chart(fig, use_container_width=True)

    if samplingRate > 0:
        with right_column:
            st.header("Reconstructed Signal")
            reconstructedAmp = signalReconstructing(timeReadings, sampledTime, sampledAmplitude)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=timeReadings, y=reconstructedAmp, mode='lines', line=dict(color='firebrick')))
            fig2.update_xaxes(title_text="Time (s)")
            fig2.update_yaxes(title_text="Amplitude (mV)")
            st.plotly_chart(fig2, use_container_width=True)


# ----------------------- Function of reading data from file ------------------------------

def read_file(file):
    df = pd.read_csv(file)
    return df

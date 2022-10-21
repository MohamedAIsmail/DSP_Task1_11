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

    # The following equations is according to White- Shannon interpoltion formula ((t - nT)/T)
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

    st.write(GetMaximumFrequencyComponent(timeReadings, amplitudeReadings))

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


# ----------------------- Function of reading data from file and plotting ------------------------------

def read_file(file):
    df = pd.read_csv(file)
    return df


def Plotting():
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis)
    ax.set_xlabel(f'{x_axis_label}')
    ax.set_ylabel(f'{y_axis_label}')
    st.plotly_chart(fig, use_container_width=True)

# ----------------------- Function of adding noise ------------------------------

def addNoise(timeReadings, amplitudeReadings, snr_db):
    power_watt = amplitudeReadings**2
    power_avg_watt = np.mean(power_watt)
    power_avg_db = 10 * np.log10(power_avg_watt)
    noise_power_avg_db = power_avg_db - snr_db
    # convert P(dB) => P(watt)
    noise_power_avg_watts = 10 ** (noise_power_avg_db / 10)

    #     # # Generate an sample of white noise
    noise_mean = 0
    noise_volts = np.random.normal(
        noise_mean, np.sqrt(noise_power_avg_watts), len(power_watt))
        
    signal_with_noise = amplitudeReadings + noise_volts

    noiseFig = go.Figure()
    noiseFig.add_trace(go.Scatter(
        x=timeReadings, y=signal_with_noise, mode='lines', line=dict(color='firebrick')))
    noiseFig.update_xaxes(title_text="Time (s)")
    noiseFig.update_yaxes(title_text="Amplitude (mV)")
    st.plotly_chart(noiseFig, use_container_width=True)
    


# ----------------------- Function of converting any signal to CSV FILE ------------------------------

def convert_to_dataframe(timeReading, ampltiudeReading, par1_name, par2_name):
    signal = []
    for i in range(len(time)):
        signal.append([timeReading[i], ampltiudeReading[i]])
    return pd.DataFrame(signal, columns=[f'{par1_name}', f'{par2_name}'])


def download_csv_file(timeReading, ampltiudeReading, file_name, x_axis_label, y_axis_label):
    signal_analysis_table = convert_to_dataframe(
        timeReading, ampltiudeReading, x_axis_label, y_axis_label)
    signal_csv = signal_analysis_table.to_csv()
    st.download_button('Download CSV file', signal_csv,
                       f'signal_{file_name}.csv')



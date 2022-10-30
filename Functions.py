import plotly.graph_objects as go
import streamlit as st  # 🎈 data web app development
import pandas as pd  # read csv, df manipulation
import scipy.fft
import numpy as np  # np mean, np random
from scipy.signal import find_peaks


# Transforming Signal to Frequency Domain to Capture Value of Maximum Frequency


def GetMaximumFrequencyComponent(time, amplitudes):

    magnitudes = np.abs(scipy.fft.rfft(amplitudes)) / \
        np.max(np.abs(scipy.fft.rfft(amplitudes)))

    frequencies = scipy.fft.rfftfreq(
        len(time), (time[1] - time[0]))

    for index, frequency in enumerate(frequencies):
        if magnitudes[index] >= 0.05:
            maximumFrequency = frequency

    return round(maximumFrequency)

# ----------------------- Function of the Signal Resampling ------------------------------


def signalSampling(Amplitude, Time, sampleFreq, timeRange):

    # calculating the steps in points between a sampling point and another
    PointSteps = int((len(Time)/timeRange)/(sampleFreq))

    if PointSteps == 0:
        PointSteps = 1

    # sampledTime list contains the time points that we took samples at
    sampledTime = Time[::PointSteps]
    sampledAmplitude = Amplitude[::PointSteps]

    return sampledAmplitude, sampledTime

# ----------------------- Function of adding noise ------------------------------


def addNoise(signalAmplitude, snr_db):

    power_watt = signalAmplitude**2
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
    signal_with_noise = signalAmplitude + noise_amplitudes

    return signal_with_noise

# ----------------------- Function of Reconstructing the Signal ------------------------------


def signalReconstructing(time_Points, sampledTime, sampledAmplitude):

    # Matrix containing all Timepoints
    TimeMatrix = np.resize(time_Points, (len(sampledTime), len(time_Points)))

    # The following equations is according to White - Shannon interpoltion formula ((t - nT)/T)
    # Transpose for TimeMatrix is a must for proper calculations (broadcasting)
    interpolation_Formula = (TimeMatrix.T - sampledTime) / \
        (sampledTime[1] - sampledTime[0])

    # Reconstructed Amplitude = x[n] sinc(v) -- Whitetaker Shannon
    finalMatrix = sampledAmplitude * np.sinc(interpolation_Formula)

    # Summation of columns of the final matrix to get an array of reconstructed points
    reconstructedSignal = np.sum(finalMatrix, axis=1)

    return reconstructedSignal


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

def UploadedSignal(timeReadings, amplitudeReadings, samplingRate, AddNoiseCheckBox, showReconstructedSignal, showSamplingPoints, snr_db
                   ):

    timeRange_max = max(timeReadings)
    timeRange_min = min(timeReadings)
    timeRange = timeRange_max - timeRange_min

    fig = go.Figure()

    if (AddNoiseCheckBox):
        signal_with_Noise = addNoise(amplitudeReadings, snr_db)
        fig.add_trace(go.Scatter(x=timeReadings, y=signal_with_Noise,
                                 mode='lines', name='Noised', marker_color='#0784b5', line=dict(width=3)))
        sampledAmplitude, sampledTime = signalSampling(
            signal_with_Noise, timeReadings, samplingRate, timeRange)
    else:
        fig.add_trace(go.Scatter(x=timeReadings, y=amplitudeReadings,
                                 mode='lines', name='Original', marker_color='#0fb7bd', line=dict(width=3)))
        sampledAmplitude, sampledTime = signalSampling(
            amplitudeReadings, timeReadings, samplingRate, timeRange)

    # Reconstructing the signal then plotting it
    reconstructedAmp = signalReconstructing(
        timeReadings, sampledTime, sampledAmplitude)

    if (showReconstructedSignal):
        fig.add_trace(go.Scatter(x=timeReadings, y=reconstructedAmp,
                                 mode='lines', name='Reconstructed', marker_color='#FFF01F ', line=dict(width=2)))

    # Sampling points on signal
    if (showSamplingPoints):
        fig.add_trace(go.Scatter(x=sampledTime, y=sampledAmplitude,
                                 mode='markers', name='Sampled', marker_color='red', marker=dict(size=4)))

    fig.update_xaxes(range=[0, timeRange_max], showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(font=dict(size=13),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
                      showlegend=True,
                      autosize=False,
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

    reconstructdesignal_csv = download_csv_file(
        timeReadings, reconstructedAmp, 'Time (s)', 'Voltage (V)')

    return reconstructdesignal_csv

# ----------------------- Generating SIGNAL ------------------------------


def GeneratedSignal(ComposedT, samplingRate, AddNoiseCheckBox, showReconstructedSignal, showSamplingPoints, snr_db
                    ):
    timeRange_max = max(ComposedT)
    timeRange_min = min(ComposedT)
    timeRange = timeRange_max - timeRange_min

    composedSignal = summedsignal(ComposedT)

    fig = go.Figure()

    if (AddNoiseCheckBox):
        signal_with_Noise = addNoise(composedSignal, snr_db)
        fig.add_trace(go.Scatter(x=ComposedT, y=signal_with_Noise,
                                 mode='lines', name='Noised', marker_color='#0784b5', line=dict(width=3)))
        sampledAmplitude, sampledTime = signalSampling(
            signal_with_Noise, ComposedT, samplingRate, timeRange)
    else:
        fig.add_trace(go.Scatter(x=ComposedT, y=composedSignal,
                                 mode='lines', name='Composed', marker_color='#0fb7bd', line=dict(width=3)))
        sampledAmplitude, sampledTime = signalSampling(
            composedSignal, ComposedT, samplingRate, timeRange)

    # Reconstructing the signal then plotting it
    reconstructedAmp = signalReconstructing(
        ComposedT, sampledTime, sampledAmplitude)

    if (showReconstructedSignal):
        fig.add_trace(go.Scatter(x=ComposedT, y=reconstructedAmp,
                                 mode='lines', name='Reconstructed', marker_color='#FFF01F', line=dict(width=2)))

    # Sampling points on signal
    if (showSamplingPoints):
        fig.add_trace(go.Scatter(x=sampledTime, y=sampledAmplitude,
                                 mode='markers', name='Sampled', marker_color='red', marker=dict(size=4)))

    fig.update_xaxes(range=[0, timeRange_max], showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(width=800,
                      height=600,
                      showlegend=True,
                      autosize=False,
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

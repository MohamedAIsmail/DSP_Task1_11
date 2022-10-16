from fileinput import filename
from turtle import title
from scipy import signal, interpolate
from scipy.interpolate import interp1d,  BarycentricInterpolator
import scipy.fft

import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

import streamlit as st  # 🎈 data web app development
import scipy as sc
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.set_page_config(page_title="Sampling Studio",
                   page_icon=":bar_chart:", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Signal Studio")

fs = [
    lambda x: (x - 3) * (x + 3) * x,  # cubic
    lambda x: np.exp(-x**2 / 2),     # gaussian
    lambda x: np.sin(3*x) / (3*x),       # sinc function
    lambda x: 1 / (np.exp(-2*x) + 1)   # logistic
]


def chebyshev(k, scale=1):
    """
    return k Chebyshev interpolation points in the range [-scale, scale]
    """
    return scale*np.cos(np.arange(k) * np.pi / (k-1))


uploaded_Signal = st.sidebar.file_uploader('Upload your Signal here!')


# ----------------------- Function of Reading data and plotting it ------------------------------
def SignalPlotting(df):
    Time = df['Time (s)']
    Amplitude = df['Voltage (mV)']

    max_Time = df['Time (s)'].max()
    min_Time = df['Time (s)'].min()

    SamplingRate = st.slider('Choose desired Sampling points', 0, 200)

    left_column, right_column = st.columns(2)

    with left_column:
        f = fs[2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Time, y=Amplitude,
                                 mode='lines', name='Signal Plot'))

    # ----------------------- Calculating Sampled frequency according to the points chosen -----------------------------

        maximumFrequencyRatio = round(SamplingRate)

        global SampledFreq
        SampledFreq = np.linspace(min_Time, max_Time, SamplingRate)
        global SampledAmp
        SampledAmp = signal.resample(Amplitude, SamplingRate)

        fig.add_trace(go.Scatter(
            x=SampledFreq, y=SampledAmp, mode='markers', line=dict(color='firebrick')))

        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude (mV)")
        st.write(fig)

    if SamplingRate != 0:
        with right_column:
            interpolatedFreq = chebyshev(SamplingRate, scale=min_Time)
            interpolatedAmp = BarycentricInterpolator(interpolatedFreq, SampledAmp)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=Time, y=interpolatedAmp(Time), line=dict(color='firebrick')))
            st.write(fig2)


# Transforming Signal to Frequency Domain to Capture Value of Maximum Frequency

def GetMaximumFrequency(timeReadings, amplitudeReadings):
    magnitudes = np.abs(scipy.fft.rfft(amplitudeReadings)) / \
        np.max(np.abs(scipy.fft.rfft(amplitudeReadings)))
    frequencies = scipy.fft.rfftfreq(
        len(timeReadings), (timeReadings[1] - timeReadings[0]))
    for index, frequency in enumerate(frequencies):
        if magnitudes[index] >= 0.05:
            maximumFrequency = frequency
    return round(maximumFrequency)

# Mathematical Linear Interpolation


def InterpolateDataPoints(dataPointsToInterpolate, timestepToFindSampleValueAt):
    sampleValue = dataPointsToInterpolate[0][1] + (timestepToFindSampleValueAt - dataPointsToInterpolate[0][0]) * (
        (dataPointsToInterpolate[1][1] - dataPointsToInterpolate[0][1]) / (dataPointsToInterpolate[1][0] - dataPointsToInterpolate[0][0]))
    return sampleValue[0]

# Using Mathematical Linear Interpolation to Generate Samples According to Chosen Sampling Frequency


def ResampleSignal(self, timeReadings, amplitudeReadings, maximumFrequencyRatio):
    hi = 1


options = st.sidebar.radio(
    'Pages', options=['Signal Plotting', 'Sampling Signal'])


if uploaded_Signal:
    df = pd.read_csv(uploaded_Signal)
    if options == 'Signal Plotting':
        SignalPlotting(df)


# ----------------------- Plotting Sampled Signal ----------------------------------

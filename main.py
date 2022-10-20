""""
TODO
moshkla any lazm a3ml view abl mmsah
a3ml separate plots 
"""


from fileinput import filename
from turtle import title
from click import option
from pyparsing import col
from scipy import signal, interpolate
from scipy.interpolate import interp1d,  BarycentricInterpolator
import scipy.fft
import math


import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

import streamlit as st  # 🎈 data web app development
import scipy as sc
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


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

# ----------------------- plot any signal ------------------------------

# def plotanysig(x,y,z):
    
#     plot = px.line(x= x,y=y,title=z)
#     st.plotly_chart(plot)


# ----------------------- Function of plotting the composed signal ------------------------------

def signalplotter(sig,t,z):
    plot = px.line(x= t,y=sig,title=z)
    st.plotly_chart(plot)
   

# ----------------------- Function of plotting the summed signal ------------------------------

def summedsignal(t):
    ysum=0
    for i in range(len(st.session_state.sigparameters)):
        ysum += st.session_state.sigparameters[i][0] * np.sin(2 * np.pi * st.session_state.sigparameters[i][1] * t + st.session_state.sigparameters[i][2] )
    return ysum



#----------------------- find selected signal----------------------------------
def findsig(name):
    for i in range(len(st.session_state.sigparameters)):
        if name == st.session_state.sigparameters[i][3]:
            return st.session_state.sigparameters[i]

#----------------------- delete selected signal----------------------------------
def delsig(name):
    for i in range(len(st.session_state.sigparameters)):
        if name == st.session_state.sigparameters[i][3]:
            return i

#---------------------- handle click -------------------------------------------
def handle_click(name):  

    if name != None:

        indx =delsig(name)
        st.session_state.sigparameters.remove(st.session_state.sigparameters[indx])    
#---------------------- view selected signal-------------------------------------------

def view_selected_signal():

    if signalselect != None:
        sigpar = findsig(signalselect)
        sig = sigpar[0] * np.sin(2 * np.pi * sigpar[1]  * t + sigpar[2]  )
        # signalplotter(sig,t,'the selected signal')
        ax[0].plot(t, sig)

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
    'Pages', options=['Signal Plotting', 'Sampling Signal','sinusoidal composer'])


if uploaded_Signal:
    df = pd.read_csv(uploaded_Signal)
    if options == 'Signal Plotting':
        SignalPlotting(df)

    

  

if options == 'sinusoidal composer':
    
    # st.session_state
    if 'sigparameters' not in st.session_state:
         st.session_state['sigparameters']=[]
    if 'a_count' not in st.session_state:
        st.session_state['a_count']=0

    col1, col2 = st.columns([3, 1])

    fig, ax = plt.subplots(2)



    with col2:

        with st.form('sinusode composer'):
            t = np.linspace(-2, 2, 10000)
            freq = st.number_input('frequency', min_value=0.0, max_value=60.0, step=1.0)
            amp = st.number_input('amplitude' ,step=1.0)
            phi = st.number_input('phase', min_value = -2 * np.pi, max_value = 2 * np.pi, step=np.pi,value=0.0)
            view = st.form_submit_button('view')
            addsig = st.form_submit_button('add')
            
  


    with col1:
        
        if view:            
            sig=amp * np.sin(2 * np.pi * freq * t + phi )
            # signalplotter(sig,t,'the composed signal')
            ax[0].plot(t, sig)


        if addsig:
            st.session_state.a_count += 1
            name= 'signal ' + str(st.session_state.a_count)  

            signal=[amp,freq,phi,name]
            st.session_state.sigparameters.append(signal)
            print(st.session_state.sigparameters)
            # signalplotter(summedsignal(t),t,'the summed signal')
            ax[1].plot(t, summedsignal(t))



    with col2:
        # if addsig or (view and len(st.session_state.sigparameters)):
            # with st.form('sinusode selection'):
            slct=[]
            for i in range(len(st.session_state.sigparameters)):
                slct.append(st.session_state.sigparameters[i][3])  

            signalselect = st.selectbox('select a signal',slct, on_change =view_selected_signal)
            
            # viewslct = st.form_submit_button('view')


            deletesig = st.button('delete', on_click = handle_click, args = (signalselect,))

    with col1:
            
        view_selected_signal()
        # if viewslct:
        # if signalselect != None:
        #     sigpar = findsig(signalselect)
        #     sig = sigpar[0] * np.sin(2 * np.pi * sigpar[1]  * t + sigpar[2]  )
        #     # signalplotter(sig,t,'the selected signal')
        #     ax[0].plot(t, sig)


        if deletesig:
            if signalselect != None:

                
                if(len(st.session_state.sigparameters)):
                    # signalplotter(summedsignal(t),t,'the summed signal')
                    ax[1].plot(t, summedsignal(t))


    st.plotly_chart(fig)

# ----------------------- Plotting Sampled Signal ----------------------------------

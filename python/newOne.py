import matplotlib.pyplot as plt
import plotly
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sampling Studio",
                   page_icon="144316.png", layout="wide")

# set the page styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.image('sampling studio.png', width=200)
st.sidebar.title('Navigation')


def convert_to_dataframe(par1, par2, par1_name, par2_name):
    signal = []
    for i in range(len(time)):
        signal.append([par1[i], par2[i]])
    return pd.DataFrame(signal, columns=[f'{par1_name}', f'{par2_name}'])


def plot_two_arrays(x_axis, y_axis, x_axis_label, y_axis_label):
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis)
    ax.set_xlabel(f'{x_axis_label}')
    ax.set_ylabel(f'{y_axis_label}')
    st.plotly_chart(fig, True)


def download_csv_file(par1, par2, file_name, x_axis_label, y_axis_label):
    # ######################################
    signal_analysis_table = convert_to_dataframe(
        par1, par2, x_axis_label, y_axis_label)
    signal_csv = signal_analysis_table.to_csv()
    st.download_button('Download CSV file', signal_csv,
                       f'signal_{file_name}.csv')


st.image("144316.png", width=100)
options = st.sidebar.radio(
    'Select What You Want To Display',
    options=['Home',
             'Upload File'])

if options == 'Home':
    amplitude = st.sidebar.slider('Amplitude', 0.1, 100.0, 1.0)
    frequency = st.sidebar.slider('frequency', 0.1, 100.0, 1.0)
    phase_shift = st.sidebar.slider('Phase Shift', 0.0, 360.0, 0.0)
    time = np.linspace(0.000001, 1, 1000)
    voltage = amplitude*np.sin(2*np.pi*frequency*time+phase_shift*np.pi/360)
    additions = st.radio(
        "Addition To Display",
        options=['Default Signal',
                 'Add Nois',
                 'Power(watt)',
                 'Power(dBw)'])
    if additions == 'Default Signal':
        plot_two_arrays(time, voltage, "Time (s)", "Voltage (V)")
        download_csv_file(time, voltage, '', 'Time (s)', 'Voltage (V)')

    if additions == 'Power(watt)':
        power_watt = voltage**2
        plot_two_arrays(time, power_watt, "Time (s)", "Power (watt)")
        download_csv_file(time, power_watt, 'Power(watt)',
                          "Time (s)", "Power (watt)")

    if additions == 'Power(dBw)':
        power_dBw = 10*np.log10(voltage**2)
        plot_two_arrays(time, power_dBw, "Time (s)", "Power (dBw)")
        download_csv_file(time, power_dBw, 'Power(dBw)',
                          "Time (s)", "Power (dBw)")

    if additions == 'Add Nois':
        snr_db = st.slider('SNR (dBw)', 0.01, 100.0, 20.0)
        power_watt = voltage**2
        power_avg_watt = np.mean(power_watt)
        power_avg_db = 10 * np.log10(power_avg_watt)
        noise_power_avg_db = power_avg_db - snr_db
        # convert P(dB) => P(watt)
        noise_power_avg_watts = 10 ** (noise_power_avg_db / 10)
    #     # # Generate an sample of white noise
        noise_mean = 0
        noise_volts = np.random.normal(
            noise_mean, np.sqrt(noise_power_avg_watts), len(power_watt))
        signal_with_noise = voltage + noise_volts
        plot_two_arrays(time, signal_with_noise, "Time (s)", "Noise volts (V)")
        download_csv_file(time, signal_with_noise, 'Signal_with_Nois(V)',
                          "Time (s)", "Signal with Nois (dBw)")

if options == 'Upload File':
    # ________________ upload file
    signal_uploaded_file = st.sidebar.file_uploader('Upload Here')

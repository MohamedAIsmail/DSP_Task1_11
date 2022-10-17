import numpy as np
import streamlit as st


class Signal:
    def __init__(self, amplitude=1, frequency=1, phase_shift=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift*np.pi/360

    def set_amplitude(self, new_amplitude):
        self.amplitude = new_amplitude
        return self.amplitude

    def set_phase_shift(self, new_phase_shift):
        self.phase_shift = new_phase_shift
        return self.phase_shift

    def set_frequency(self, new_frequency):
        self.frequency = new_frequency
        return self.frequency

    def get_amplitude(self):
        return self.amplitude

    def generate_signal(self):
        '''retuns the voltage according to v=vmax sin(omegat+theta)'''
        time = np.linspace(0.000001, 1, 1000)
        return self.amplitude*np.sin(2*np.pi*self.frequency*time+self.phase_shift)

    def get_power(self):
        '''returns the power in watt of the signal according to P=V^2'''
        return self.generate_signal()**2

    def get_power_dBw(self):
        '''returs the power in dBw according to P(dBw)=10log10(P(watt))'''
        return 10*np.log10(self.get_power())

    # Noise
    def add_noise(self, snr_db=20):
        '''returns an array contanis a signal with noise in volts
            SNR=Psignal/Pnoise,  SNR = P(dB)-nP(db)
        '''
        self.power_avg_watt = np.mean(self.get_power())
        self.power_avg_db = 10 * np.log10(self.power_avg_watt)
        self.noise_power_avg_db = self.power_avg_db - snr_db
        # convert P(dB) => P(watt)
        self.noise_power_avg_watts = 10 ** (self.noise_power_avg_db / 10)
        # # Generate an sample of white noise
        self.noise_mean = 0
        self.noise_volts = np.random.normal(
            self.noise_mean, np.sqrt(self.noise_power_avg_watts), len(self.get_power()))
        # # Noise up the original signal
        return self.generate_signal() + self.noise_volts


# signal1 = Signal()
# signal2 = Signal(5, 20, 3.14)
# print(signal1.amplitude)
# print(signal2.get_amplitude())
signal1 = Signal()
st.header('Signal')
st.line_chart(signal1.generate_signal())
st.header('Signal Power in watt')
st.line_chart(signal1.get_power())
st.header('Signal Power in dBw')
st.line_chart(signal1.get_power_dBw())
st.header('Signal With Noise')
st.line_chart(signal1.add_noise())

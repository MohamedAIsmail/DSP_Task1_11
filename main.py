import streamlit.components.v1 as components
import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development
import matplotlib.pyplot as plt
import mpld3

# create your figure and get the figure object returned
fig = plt.figure()
plt.plot([1, 2, 3, 4, 5])

fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)

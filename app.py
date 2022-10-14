from __future__ import division
import wave
from flask import Flask, render_template, url_for, request, redirect, make_response
import json
import io

import numpy as np
import matplotlib.pyplot as plot
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math

from bokeh.io import output_file, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.layouts import layout
from bokeh.plotting import figure, show, output_file
from bokeh.models import Div, RangeSlider, Spinner
from bokeh.models.widgets import Tabs, Panel

plot.rcParams["figure.figsize"] = [7.50, 3.50]
plot.rcParams["figure.autolayout"] = True

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():

    return render_template('index.html')


@app.route('/data', methods=["GET", "POST"])
def data():

    length = 0.05
    sampleRate = 48000.0
    f1 = 1000.0

    waveform = [math.sin(2.0 * math.pi * f1 * i / sampleRate)
                for i in range(int/length * sampleRate)]
    timeBins = [1000 * i / sampleRate for i in range(len(waveform))]

    data = [timeBins, waveform]

    response = make_response(json.dumps(data))
    response.content_type = 'application/json'

    return response


if __name__ == "__main__":
    app.run(debug=True)

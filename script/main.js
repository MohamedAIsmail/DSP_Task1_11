var x_max = 50;

let chart = new Highcharts.Chart({
    chart:
    {
        renderTo: 'container',
        type: 'line',
        marginRight: 130,
        marginBottom: 25,
        zoomType: "x",
    },
    title:
    {
        text: 'Signal Data',
        x: -20
    },
    xAxis: {
        title: {
            text: 'Time (ms)',
            margin: 80
        }
    },
    yAxis: {
        title: {
            text: 'Amplitude',
            margin: 80
        }
    },
    series: [{
        name: 'Sine Wave Values',
        color: 'rgba(40, 120, 120, 0.5)',
        data: calculate(),
        turboThreshold: 0
    },
    {
        name: 'Sampled Wave Values',
        color: 'rgba(255, 0, 0, 1)',
        data: calculateSampled(),
        turboThreshold: 0
    }]
})

function calculateY(x) {
    return Math.sin(x);
}
function calculateSampled() {
    var v = [];
    for (var xt = 0; xt <= x_max; xt += 1 / getRate()) {
        v.push({ x: xt, y: calculateY(xt) });
    }
    return v;
}

function calculate() {
    var v = [];
    for (var xt = 0; xt <= x_max; xt += .01) {
        v.push({ x: xt, y: calculateY(xt) });
    }
    return v;
}

function getRate() {
    return (+$("#range").val());
}

function plot() {
    chart.series[1].setData(calculateSampled());
}

$('#range').change(update);

function update() {
    $("#fsval").val(getRate());
    plot();
}

update();

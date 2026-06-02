let chartLatency = null;
let chartServerChoice = null;
const chartDataLatency = { labels: [], datasets: [] };
const chartDataServer = { labels: [], datasets: [] };
const SERVER_NAMES = Object.keys(CACHE_COORDS);
const SERVER_PALETTE = [
  ["rgba(40,167,69,1)", "rgba(40,167,69,0.15)"],
  ["rgba(255,153,0,1)", "rgba(255,153,0,0.15)"],
  ["rgba(0,123,255,1)", "rgba(0,123,255,0.15)"],
];
const SERVER_COLOR_MAP = SERVER_NAMES.reduce((acc, serverName, index) => {
  acc[serverName] = SERVER_PALETTE[index % SERVER_PALETTE.length][0];
  return acc;
}, {});
const SERVER_COLOR_MAP_BG = SERVER_NAMES.reduce((acc, serverName, index) => {
  acc[serverName] = SERVER_PALETTE[index % SERVER_PALETTE.length][1];
  return acc;
}, {});
const SERVER_NUMERIC_MAP = SERVER_NAMES.reduce((acc, serverName, index) => {
  acc[serverName] = index + 1;
  return acc;
}, {});
const MAX_CHART_POINTS = 300;
function _initCharts() {
  const ctxLat = document.getElementById("chartLatency");
  const ctxSrv = document.getElementById("chartServerChoice");
  if (!ctxLat || !ctxSrv) return;
  if (chartLatency) {
    chartLatency.destroy();
    chartLatency = null;
  }
  if (chartServerChoice) {
    chartServerChoice.destroy();
    chartServerChoice = null;
  }
  chartDataLatency.labels = [];
  chartDataLatency.datasets = [];
  chartDataServer.labels = [];
  chartDataServer.datasets = [];
  for (const cacheName in CACHE_COORDS) {
    chartDataLatency.datasets.push({
      label: CACHE_COORDS[cacheName].label,
      data: [],
      borderColor: SERVER_COLOR_MAP[cacheName] || "grey",
      backgroundColor:
        SERVER_COLOR_MAP_BG[cacheName] || "rgba(128,128,128,0.1)",
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    });
  }
  chartDataServer.datasets.push({
    label: "Chosen Server",
    data: [],
    borderColor: "rgba(0,0,0,0.8)",
    backgroundColor: "rgba(0,123,255,0.15)",
    borderWidth: 2,
    pointRadius: 3,
    pointBackgroundColor: [],
    stepped: "before",
    fill: false,
  });
  chartLatency = new Chart(ctxLat, {
    type: "line",
    data: chartDataLatency,
    options: {
      responsive: true,
      animation: { duration: 0 },
      scales: {
        x: {
          title: { display: true, text: "Sim Time (s)" },
          ticks: { maxTicksLimit: 20 },
        },
        y: {
          title: { display: true, text: "Latency (ms)" },
          beginAtZero: true,
        },
      },
      plugins: {
        legend: {
          position: "top",
          labels: { boxWidth: 12, font: { size: 11 } },
        },
      },
      interaction: { mode: "index", intersect: false },
    },
  });
  chartServerChoice = new Chart(ctxSrv, {
    type: "line",
    data: chartDataServer,
    options: {
      responsive: true,
      animation: { duration: 0 },
      scales: {
        x: {
          title: { display: true, text: "Sim Time (s)" },
          ticks: { maxTicksLimit: 20 },
        },
        y: {
          title: { display: true, text: "Server" },
          min: 0.5,
          max: 3.5,
          ticks: {
            stepSize: 1,
            callback: function (value) {
              const serverName = SERVER_NAMES[value - 1];
              return serverName ? CACHE_COORDS[serverName].label : "";
            },
          },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}
function _updateCharts(simTime, latencies, decision) {
  if (!chartLatency || !chartServerChoice) return;
  if (chartDataLatency.labels.length >= MAX_CHART_POINTS) {
    chartDataLatency.labels.shift();
    chartDataLatency.datasets.forEach((ds) => ds.data.shift());
    chartDataServer.labels.shift();
    chartDataServer.datasets[0].data.shift();
    chartDataServer.datasets[0].pointBackgroundColor.shift();
  }
  chartDataLatency.labels.push(simTime);
  const serverNames = Object.keys(CACHE_COORDS);
  for (let i = 0; i < serverNames.length; i++) {
    const lat = latencies[serverNames[i]];
    chartDataLatency.datasets[i].data.push(
      lat != null ? parseFloat(lat.toFixed(1)) : null,
    );
  }
  chartDataServer.labels.push(simTime);
  const srvNum = SERVER_NUMERIC_MAP[decision] || null;
  chartDataServer.datasets[0].data.push(srvNum);
  chartDataServer.datasets[0].pointBackgroundColor.push(
    SERVER_COLOR_MAP[decision] || "rgba(128,128,128,1)",
  );
  chartLatency.update("none");
  chartServerChoice.update("none");
}

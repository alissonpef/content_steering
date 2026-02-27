let player;
let currentSegmentServiceLocation = { audio: null, video: null };
let cdnIconDomElements = {};
let simTimer = null;
let simElapsedTime = 0;
let simMovementActive = false;
let simIntervalID_movement = null;
let movementStarted = false;
let simSpamActive_1 = false,
  simSpamActive_2 = false;
let simSpamEventSent_1 = false;
let simSpamEventSent_2 = false;
let simCurrentLat, simCurrentLon;
let isSimulationRunning = false;
let manifestSuccessfullyLoaded = false;
let currentRunIndex = 0;
let totalRunsToExecute = 1;
let onManifestLoadedCallback = null;
let onManifestErrorCallback = null;
let onStreamInitForPlay = null;
let onStreamInitForAutomaticPlay = null;
let fragmentLoadStarts = {};
let chartLatency = null;
let chartServerChoice = null;
const chartDataLatency = { labels: [], datasets: [] };
const chartDataServer = { labels: [], datasets: [] };
const SERVER_COLOR_MAP = {
  "video-streaming-cache-1": "rgba(40,167,69,1)",
  "video-streaming-cache-2": "rgba(255,153,0,1)",
  "video-streaming-cache-3": "rgba(0,123,255,1)",
};
const SERVER_COLOR_MAP_BG = {
  "video-streaming-cache-1": "rgba(40,167,69,0.15)",
  "video-streaming-cache-2": "rgba(255,153,0,0.15)",
  "video-streaming-cache-3": "rgba(0,123,255,0.15)",
};
const SERVER_NUMERIC_MAP = {
  "video-streaming-cache-1": 1,
  "video-streaming-cache-2": 2,
  "video-streaming-cache-3": 3,
};
const MAX_CHART_POINTS = 300;
function stopInterval(intervalId) {
  if (intervalId != null) clearInterval(intervalId);
}
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
              const names = {
                1: "Cache 1 (BR)",
                2: "Cache 2 (CL)",
                3: "Cache 3 (CO)",
              };
              return names[value] || "";
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
function setupPlayer() {
  const videoElement = document.querySelector("video");
  if (player) {
    if (onManifestLoadedCallback)
      player.off(
        dashjs.MediaPlayer.events.MANIFEST_LOADED,
        onManifestLoadedCallback,
      );
    if (onManifestErrorCallback)
      player.off(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback);
    if (onStreamInitForPlay)
      player.off(
        dashjs.MediaPlayer.events.STREAM_INITIALIZED,
        onStreamInitForPlay,
      );
    if (onStreamInitForAutomaticPlay)
      player.off(
        dashjs.MediaPlayer.events.STREAM_INITIALIZED,
        onStreamInitForAutomaticPlay,
      );
    player.reset();
  }
  player = dashjs.MediaPlayer().create();
  player.initialize(videoElement, null, false);
  player.on(
    dashjs.MediaPlayer.events.FRAGMENT_LOADING_STARTED,
    _onFragmentLoadingStarted,
  );
  player.on(
    dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED,
    _onFragmentLoadingCompleted,
  );
  player.on(
    dashjs.MediaPlayer.events.CONTENT_STEERING_REQUEST_COMPLETED,
    _onContentSteeringRequestCompleted,
  );
  player.on(dashjs.MediaPlayer.events.ERROR, (e) => {
    if (
      e.error &&
      e.error.code &&
      (e.error.code ===
        dashjs.MediaPlayer.errors.MANIFEST_LOADER_PARSING_FAILURE_ERROR_CODE ||
        e.error.code ===
          dashjs.MediaPlayer.errors
            .MANIFEST_LOADER_LOADING_FAILURE_ERROR_CODE ||
        e.error.code === dashjs.MediaPlayer.errors.DOWNLOAD_ERROR_ID_MANIFEST)
    ) {
      manifestSuccessfullyLoaded = false;
      document.getElementById("button_StartControlledSim").disabled = true;
    }
  });
}
function init() {
  console.log("Initializing simulation...");
  if (typeof CACHE_COORDS === "undefined") {
    console.error("CACHE_COORDS is not defined!");
    alert(
      "Configuration error: CACHE_COORDS missing. Check config.js loading.",
    );
    return;
  }
  if (typeof dashjs === "undefined") {
    console.error("dashjs is not defined!");
    alert("Error: dash.js library not loaded.");
    return;
  }
  setupPlayer();
  setupEventListeners();
  populateSelect("simMovementTarget", "Stay Still");
  populateSelect("simSpamTarget_1", "No Spam");
  populateSelect("simSpamTarget_2", "No Spam");
  const cdnContainer = document.getElementById("cdn-selection-container");
  cdnContainer.innerHTML = "";
  cdnIconDomElements = {};
  for (const cacheName in CACHE_COORDS) {
    _createIcon(cdnContainer, cacheName, cdnIconDomElements, "cdn");
  }
  _resetUIOnly();
  document.getElementById("button_StartControlledSim").disabled = true;
  document.getElementById("button_StopSim").disabled = true;
}
function setupEventListeners() {
  document.getElementById("load-button").addEventListener("click", _load);
  document
    .getElementById("button_StartControlledSim")
    .addEventListener("click", startControlledSimulation);
  document
    .getElementById("button_StopSim")
    .addEventListener("click", stopCurrentSimulation);
  document
    .getElementById("button_ResetSimUI")
    .addEventListener("click", resetSimulationUIAndState);
  document
    .getElementById("simLoops")
    .addEventListener("input", updateCalculatedDuration);
  const runModeRadios = document.querySelectorAll('input[name="runMode"]');
  runModeRadios.forEach((radio) => {
    radio.addEventListener("change", (e) => {
      if (e.target.value === "duration") {
        document.getElementById("durationInputGroup").style.display = "block";
        document.getElementById("loopsInputGroup").style.display = "none";
      } else {
        document.getElementById("durationInputGroup").style.display = "none";
        document.getElementById("loopsInputGroup").style.display = "block";
        updateCalculatedDuration();
      }
    });
  });
}
function updateCalculatedDuration() {
  const loops = parseInt(document.getElementById("simLoops").value) || 1;
  const infoSpan = document.getElementById("calculatedDurationInfo");
  if (
    player &&
    player.isReady() &&
    player.duration() &&
    player.duration() > 0
  ) {
    const videoDuration = player.duration();
    const totalDuration = Math.ceil(videoDuration * loops);
    if (infoSpan) infoSpan.innerText = `≈ ${totalDuration}s`;
  } else {
    if (infoSpan) infoSpan.innerText = "(Load video first)";
  }
}
function populateSelect(selectId, noneOptionText) {
  const selectElement = document.getElementById(selectId);
  selectElement.innerHTML = "";
  if (noneOptionText) {
    const noneOpt = document.createElement("option");
    noneOpt.value = "none";
    noneOpt.textContent = noneOptionText;
    selectElement.appendChild(noneOpt);
  }
  for (const cacheName in CACHE_COORDS) {
    const option = document.createElement("option");
    option.value = cacheName;
    option.textContent = CACHE_COORDS[cacheName].label;
    selectElement.appendChild(option);
  }
}
function stopCurrentSimulation(
  pausePlayerAndSeek = true,
  finishedNaturally = false,
) {
  if (simTimer) {
    clearInterval(simTimer);
    simTimer = null;
  }
  stopInterval(simIntervalID_movement);
  simIntervalID_movement = null;
  simMovementActive = false;
  movementStarted = false;
  simSpamActive_1 = false;
  simSpamActive_2 = false;
  if (finishedNaturally && currentRunIndex < totalRunsToExecute) {
    _prepareNextRun();
    return;
  }
  setTimeout(() => {
    isSimulationRunning = false;
    if (pausePlayerAndSeek && player && manifestSuccessfullyLoaded) {
      if (player.isReady() && !player.isPaused()) player.pause();
      if (player.isReady()) player.seek(0);
    }
    if (currentRunIndex > 0 && currentRunIndex >= totalRunsToExecute) {
      console.log("All runs completed.");
      const ri = document.getElementById("runIndicator");
      if (ri) {
        ri.textContent = "Completed";
        ri.classList.remove("bg-info");
        ri.classList.add("bg-success");
      }
    }
    currentRunIndex = 0;
  }, 500);
  document.getElementById("button_StartControlledSim").disabled =
    !manifestSuccessfullyLoaded;
  document.getElementById("button_StopSim").disabled = true;
  document.getElementById("simMovementStatus").textContent = "Inactive";
  document.getElementById("simSpamStatus_1").textContent = "Inactive";
  document.getElementById("simSpamStatus_2").textContent = "Inactive";
}
function _resetUIOnly() {
  simElapsedTime = 0;
  document.getElementById("simCurrentTimeDisplay").textContent = "0";
  const initialLatVal = parseFloat(
    document.getElementById("initialSimLat").value,
  );
  const initialLonVal = parseFloat(
    document.getElementById("initialSimLon").value,
  );
  simCurrentLat = isNaN(initialLatVal) ? -23.0 : initialLatVal;
  simCurrentLon = isNaN(initialLonVal) ? -47.0 : initialLonVal;
  document.getElementById("current-latitude").value = simCurrentLat.toFixed(5);
  document.getElementById("current-longitude").value = simCurrentLon.toFixed(5);
  document.getElementById("steering-decision-display").textContent = "N/A";
  document.getElementById("steering-request-timestamp").textContent = "N/A";
  document.getElementById("steering-request-url").textContent = "N/A";
  document.getElementById("steering-pathway-cloning").textContent = "N/A";
  currentSegmentServiceLocation = { audio: null, video: null };
  _updateActiveServerIcons();
  fragmentLoadStarts = {};
  document.getElementById("simMovementStatus").textContent = "Inactive";
  document.getElementById("simSpamStatus_1").textContent = "Inactive";
  document.getElementById("simSpamStatus_2").textContent = "Inactive";
  movementStarted = false;
  simSpamEventSent_1 = false;
  simSpamEventSent_2 = false;
}
function resetSimulationUIAndState() {
  stopCurrentSimulation(true);
  _resetUIOnly();
  document.getElementById("button_StartControlledSim").disabled = true;
}
function startControlledSimulation() {
  if (!manifestSuccessfullyLoaded) {
    alert("Manifest not loaded. Please load an MPD first.");
    return;
  }
  if (isSimulationRunning) {
    return;
  }
  if (currentRunIndex === 0) {
    const runsInput = document.getElementById("simRuns");
    totalRunsToExecute = runsInput ? parseInt(runsInput.value) || 1 : 1;
    currentRunIndex = 1;
    fetch("https://steering-service:30500/reset_simulation", { method: "POST" })
      .then((response) => response.json())
      .then((data) => console.log("Initial backend reset:", data))
      .catch((err) => console.error("Failed initial reset:", err));
  }
  console.log(`Starting Run ${currentRunIndex} of ${totalRunsToExecute}`);
  const runIndicator = document.getElementById("runIndicator");
  if (runIndicator) {
    runIndicator.style.display = "inline-block";
    runIndicator.textContent = `Run ${currentRunIndex}/${totalRunsToExecute}`;
    runIndicator.classList.remove("bg-warning", "bg-success");
    runIndicator.classList.add("bg-info");
  }
  simElapsedTime = 0;
  simMovementActive = false;
  movementStarted = false;
  simSpamActive_1 = false;
  simSpamActive_2 = false;
  simSpamEventSent_1 = false;
  simSpamEventSent_2 = false;
  stopInterval(simIntervalID_movement);
  simIntervalID_movement = null;
  if (simTimer) {
    clearInterval(simTimer);
    simTimer = null;
  }
  document.getElementById("simCurrentTimeDisplay").textContent = "0";
  isSimulationRunning = true;
  fragmentLoadStarts = {};
  _initCharts();
  _ensurePlayerReady();
  document.getElementById("button_StartControlledSim").disabled = false;
  document.getElementById("button_StopSim").disabled = false;
  const simConfig = _readSimulationConfig();
  player.play();
  console.log("Simulation started. Duration:", simConfig.duration);
  simTimer = setInterval(() => _onSimulationTick(simConfig), 1000);
}
function _prepareNextRun() {
  console.log(
    `Run ${currentRunIndex} finished. Preparing for Run ${currentRunIndex + 1}...`,
  );
  document.getElementById("button_StartControlledSim").disabled = true;
  setTimeout(() => {
    isSimulationRunning = false;
    if (player && player.isReady()) player.seek(0);
    setTimeout(() => {
      currentRunIndex++;
      const ri = document.getElementById("runIndicator");
      if (ri) {
        ri.textContent = `Preparing Run ${currentRunIndex}/${totalRunsToExecute}...`;
        ri.classList.remove("bg-info");
        ri.classList.add("bg-warning");
      }
      fetch("https://steering-service:30500/reset_simulation", {
        method: "POST",
      })
        .then((r) => r.json())
        .then((data) => {
          console.log("Backend reset:", data);
          if (player) {
            player.reset();
            setupPlayer();
            player.updateSettings({
              streaming: {
                buffer: { stableBufferTime: 4, bufferTimeAtTopQuality: 4 },
              },
            });
            const mpdUrl = document.getElementById("manifest").value;
            if (mpdUrl) {
              const cb = () => {
                startControlledSimulation();
                player.off(dashjs.MediaPlayer.events.MANIFEST_LOADED, cb);
              };
              player.on(dashjs.MediaPlayer.events.MANIFEST_LOADED, cb);
              player.attachSource(mpdUrl);
            } else {
              startControlledSimulation();
            }
          } else {
            startControlledSimulation();
          }
        })
        .catch((err) => {
          console.error("Failed to reset backend:", err);
          alert("Failed to reset backend. Stopping sequence.");
          currentRunIndex = 0;
          isSimulationRunning = false;
          document.getElementById("button_StartControlledSim").disabled =
            !manifestSuccessfullyLoaded;
        });
    }, 1500);
  }, 500);
}
function _ensurePlayerReady() {
  function attemptSeek() {
    if (
      player.getActiveStream() &&
      player.isReady() &&
      manifestSuccessfullyLoaded
    ) {
      player.seek(0);
    }
  }
  if (player.getActiveStream() && player.isReady()) {
    attemptSeek();
  } else if (player.isReady()) {
    if (onStreamInitForPlay && player)
      player.off(
        dashjs.MediaPlayer.events.STREAM_INITIALIZED,
        onStreamInitForPlay,
      );
    onStreamInitForPlay = function () {
      if (!isSimulationRunning) return;
      attemptSeek();
      if (player)
        player.off(
          dashjs.MediaPlayer.events.STREAM_INITIALIZED,
          onStreamInitForPlay,
        );
    };
    player.on(
      dashjs.MediaPlayer.events.STREAM_INITIALIZED,
      onStreamInitForPlay,
      null,
      { once: true },
    );
  } else {
    isSimulationRunning = false;
  }
}
function _readSimulationConfig() {
  let duration = 180;
  const runMode = document.querySelector('input[name="runMode"]:checked').value;
  if (runMode === "duration") {
    duration = parseInt(document.getElementById("simDuration").value) || 180;
  } else {
    const loops = parseInt(document.getElementById("simLoops").value) || 1;
    const videoDuration = player.duration();
    duration =
      videoDuration && videoDuration > 0
        ? Math.ceil(videoDuration * loops)
        : 180;
  }
  simCurrentLat = parseFloat(document.getElementById("initialSimLat").value);
  simCurrentLon = parseFloat(document.getElementById("initialSimLon").value);
  if (isNaN(simCurrentLat)) simCurrentLat = -23.0;
  if (isNaN(simCurrentLon)) simCurrentLon = -47.0;
  document.getElementById("current-latitude").value = simCurrentLat.toFixed(5);
  document.getElementById("current-longitude").value = simCurrentLon.toFixed(5);
  document.getElementById("simMovementStatus").textContent = "Inactive";
  document.getElementById("simSpamStatus_1").textContent = "Inactive";
  document.getElementById("simSpamStatus_2").textContent = "Inactive";
  return {
    duration,
    movementTarget: document.getElementById("simMovementTarget").value,
    movementStartTime:
      parseInt(document.getElementById("simMovementStartTime").value) || 60,
    movementDuration:
      parseInt(document.getElementById("simMovementDuration").value) || 60,
    spamTarget1: document.getElementById("simSpamTarget_1").value,
    spamStartTime1:
      parseInt(document.getElementById("simSpamStartTime_1").value) || 60,
    spamDuration1:
      parseInt(document.getElementById("simSpamDuration_1").value) || 120,
    spamTarget2: document.getElementById("simSpamTarget_2").value,
    spamStartTime2:
      parseInt(document.getElementById("simSpamStartTime_2").value) || 120,
    spamDuration2:
      parseInt(document.getElementById("simSpamDuration_2").value) || 60,
  };
}
function _onSimulationTick(cfg) {
  if (!isSimulationRunning) {
    clearInterval(simTimer);
    simTimer = null;
    return;
  }
  simElapsedTime++;
  document.getElementById("simCurrentTimeDisplay").textContent = simElapsedTime;
  if (
    cfg.movementTarget !== "none" &&
    simElapsedTime >= cfg.movementStartTime &&
    !movementStarted
  ) {
    movementStarted = true;
    simMovementActive = true;
    const effectiveDur = Math.max(
      1,
      Math.min(cfg.movementDuration, cfg.duration - simElapsedTime),
    );
    startSimulatedMovement(
      cfg.movementTarget,
      simCurrentLat,
      simCurrentLon,
      effectiveDur,
    );
    document.getElementById("simMovementStatus").textContent =
      `Moving (to ${CACHE_COORDS[cfg.movementTarget]?.label || cfg.movementTarget})`;
  }
  if (
    cfg.spamTarget1 !== "none" &&
    simElapsedTime >= cfg.spamStartTime1 &&
    !simSpamEventSent_1
  ) {
    simSpamActive_1 = true;
    simSpamEventSent_1 = true;
    startSimulatedCacheSpam(cfg.spamTarget1, 1);
    document.getElementById("simSpamStatus_1").textContent =
      `Spamming (${CACHE_COORDS[cfg.spamTarget1]?.label || cfg.spamTarget1})`;
  }
  if (
    simSpamActive_1 &&
    simElapsedTime >= cfg.spamStartTime1 + cfg.spamDuration1
  ) {
    simSpamActive_1 = false;
    document.getElementById("simSpamStatus_1").textContent = "Inactive";
  }
  if (
    cfg.spamTarget2 !== "none" &&
    simElapsedTime >= cfg.spamStartTime2 &&
    !simSpamEventSent_2
  ) {
    simSpamActive_2 = true;
    simSpamEventSent_2 = true;
    startSimulatedCacheSpam(cfg.spamTarget2, 2);
    document.getElementById("simSpamStatus_2").textContent =
      `Spamming (${CACHE_COORDS[cfg.spamTarget2]?.label || cfg.spamTarget2})`;
  }
  if (
    simSpamActive_2 &&
    simElapsedTime >= cfg.spamStartTime2 + cfg.spamDuration2
  ) {
    simSpamActive_2 = false;
    document.getElementById("simSpamStatus_2").textContent = "Inactive";
  }
  reportLocationToSteering(simCurrentLat, simCurrentLon);
  fetch("https://steering-service:30500/sim_state")
    .then((r) => r.json())
    .then((state) => {
      _updateCharts(
        simElapsedTime,
        state.latencies || {},
        state.decision || "N/A",
      );
    })
    .catch(() => {});
  if (simElapsedTime >= cfg.duration) {
    stopCurrentSimulation(true, true);
  }
}
function startSimulatedMovement(
  targetCacheName,
  initialClientLat,
  initialClientLon,
  moveDurationSec,
) {
  if (!CACHE_COORDS[targetCacheName]) {
    simMovementActive = false;
    document.getElementById("simMovementStatus").textContent = "Error";
    return;
  }
  const targetCoord = CACHE_COORDS[targetCacheName];
  const totalSteps =
    moveDurationSec > 0 ? Math.max(1, Math.floor(moveDurationSec)) : 1;
  const stepLat = (targetCoord.lat - initialClientLat) / totalSteps;
  const stepLon = (targetCoord.lon - initialClientLon) / totalSteps;
  let stepsTaken = 0;
  if (simIntervalID_movement) clearInterval(simIntervalID_movement);
  simIntervalID_movement = null;
  const intervalFunc = () => {
    if (!isSimulationRunning || !simMovementActive || !simIntervalID_movement) {
      stopInterval(simIntervalID_movement);
      simIntervalID_movement = null;
      return;
    }
    if (stepsTaken < totalSteps) {
      simCurrentLat += stepLat;
      simCurrentLon += stepLon;
      stepsTaken++;
      document.getElementById("current-latitude").value =
        simCurrentLat.toFixed(5);
      document.getElementById("current-longitude").value =
        simCurrentLon.toFixed(5);
    } else {
      simCurrentLat = targetCoord.lat;
      simCurrentLon = targetCoord.lon;
      document.getElementById("current-latitude").value =
        simCurrentLat.toFixed(5);
      document.getElementById("current-longitude").value =
        simCurrentLon.toFixed(5);
      stopInterval(simIntervalID_movement);
      simIntervalID_movement = null;
      simMovementActive = false;
      if (isSimulationRunning)
        document.getElementById("simMovementStatus").textContent =
          "Reached Target";
    }
  };
  simIntervalID_movement = setInterval(intervalFunc, 1000);
}
function startSimulatedCacheSpam(targetCacheName, phaseId) {
  if (!CACHE_COORDS[targetCacheName]) {
    if (phaseId === 1) {
      simSpamActive_1 = false;
      document.getElementById("simSpamStatus_1").textContent = "Error";
    } else {
      simSpamActive_2 = false;
      document.getElementById("simSpamStatus_2").textContent = "Error";
    }
    return;
  }
  const spamDurationElementId = `simSpamDuration_${phaseId}`;
  const spamDurationValue = parseInt(
    document.getElementById(spamDurationElementId)?.value,
  );
  if (isNaN(spamDurationValue) || spamDurationValue <= 0) {
    if (phaseId === 1) {
      simSpamActive_1 = false;
      document.getElementById("simSpamStatus_1").textContent =
        "Error: Invalid Duration";
    } else {
      simSpamActive_2 = false;
      document.getElementById("simSpamStatus_2").textContent =
        "Error: Invalid Duration";
    }
    return;
  }
  const payload = {
    server_name: targetCacheName,
    factor: 10.0,
    duration_seconds: spamDurationValue,
  };
  fetch("https://steering-service:30500/latency_event", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-type": "application/json; charset=UTF-8" },
  })
    .then((response) =>
      response
        .text()
        .then((text) => ({ ok: response.ok, text, status: response.status })),
    )
    .then((data) => {})
    .catch((err) => {});
}
function reportLocationToSteering(lat, lon) {
  if (!isSimulationRunning || lat === undefined || lon === undefined) return;
  const payload = { time: simElapsedTime, lat: lat, long: lon };
  fetch("https://steering-service:30500/coords", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-type": "application/json; charset=UTF-8" },
  }).catch((error) => {});
}
function reportLatencyToSteering(lat, lon, clientMeasuredLatency, serverUsed) {
  if (!isSimulationRunning || lat === undefined || lon === undefined) return;
  if (clientMeasuredLatency === undefined || serverUsed === undefined) return;
  const payload = {
    time: simElapsedTime,
    lat: lat,
    long: lon,
    rt: clientMeasuredLatency,
    server_used: serverUsed,
  };
  fetch("https://steering-service:30500/coords", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-type": "application/json; charset=UTF-8" },
  })
    .then((response) =>
      response
        .text()
        .then((text) => ({ ok: response.ok, status: response.status, text })),
    )
    .then((data) => {})
    .catch((error) => {});
}
function _load() {
  let newMpdUrl = document.getElementById("manifest").value;
  if (!newMpdUrl) {
    alert("Please enter an MPD URL.");
    return;
  }
  manifestSuccessfullyLoaded = false;
  document.getElementById("button_StartControlledSim").disabled = true;
  if (isSimulationRunning) stopCurrentSimulation(true);
  setupPlayer();
  player.updateSettings({
    streaming: {
      buffer: {
        stableBufferTime: 4,
        bufferTimeAtTopQuality: 4,
      },
    },
  });
  _resetUIOnly();
  try {
    player.attachSource(newMpdUrl);
    onManifestLoadedCallback = function (e) {
      if (e.error) {
        alert("Error loading manifest: " + (e.error.message || e.error));
        manifestSuccessfullyLoaded = false;
        document.getElementById("button_StartControlledSim").disabled = true;
      } else {
        manifestSuccessfullyLoaded = true;
        const autoStartEnabled =
          document.getElementById("autoStartCheckbox").checked;
        if (autoStartEnabled) {
          if (onStreamInitForAutomaticPlay && player)
            player.off(
              dashjs.MediaPlayer.events.STREAM_INITIALIZED,
              onStreamInitForAutomaticPlay,
            );
          onStreamInitForAutomaticPlay = function () {
            if (player.getActiveStream() && player.isReady()) {
              if (manifestSuccessfullyLoaded && !isSimulationRunning) {
                startControlledSimulation();
              }
            }
            if (player)
              player.off(
                dashjs.MediaPlayer.events.STREAM_INITIALIZED,
                onStreamInitForAutomaticPlay,
              );
          };
          if (player.getActiveStream() && player.isReady()) {
            onStreamInitForAutomaticPlay();
          } else if (player.isReady()) {
            player.on(
              dashjs.MediaPlayer.events.STREAM_INITIALIZED,
              onStreamInitForAutomaticPlay,
              null,
              { once: true },
            );
          } else {
            document.getElementById("button_StartControlledSim").disabled =
              false;
          }
        } else {
          document.getElementById("button_StartControlledSim").disabled = false;
        }
        updateCalculatedDuration();
      }
      if (player)
        player.off(
          dashjs.MediaPlayer.events.MANIFEST_LOADED,
          onManifestLoadedCallback,
        );
    };
    player.on(
      dashjs.MediaPlayer.events.MANIFEST_LOADED,
      onManifestLoadedCallback,
      null,
      { once: true },
    );
    if (onManifestErrorCallback && player)
      player.off(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback);
    onManifestErrorCallback = function (e) {
      if (
        e.error &&
        e.error.code &&
        (e.error.code ===
          dashjs.MediaPlayer.errors
            .MANIFEST_LOADER_PARSING_FAILURE_ERROR_CODE ||
          e.error.code ===
            dashjs.MediaPlayer.errors
              .MANIFEST_LOADER_LOADING_FAILURE_ERROR_CODE ||
          e.error.code === dashjs.MediaPlayer.errors.DOWNLOAD_ERROR_ID_MANIFEST)
      ) {
        alert(
          "Failed to load or parse manifest: " + (e.error.message || e.error),
        );
        manifestSuccessfullyLoaded = false;
        document.getElementById("button_StartControlledSim").disabled = true;
      }
      if (player)
        player.off(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback);
    };
    player.on(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback, null, {
      once: true,
    });
  } catch (error) {
    alert("Error setting up player with MPD.");
    manifestSuccessfullyLoaded = false;
    document.getElementById("button_StartControlledSim").disabled = true;
  }
}
function _onFragmentLoadingStarted(e) {
  try {
    if (
      e &&
      e.mediaType &&
      (e.mediaType === "video" || e.mediaType === "audio") &&
      e.request
    ) {
      const key = e.mediaType + "_" + e.request.index;
      if (e.request.serviceLocation) {
        fragmentLoadStarts[key] = {
          startTime: performance.now(),
          serviceLocation: e.request.serviceLocation,
          url: e.request.url,
        };
        currentSegmentServiceLocation[e.mediaType] = e.request.serviceLocation;
        _updateActiveServerIcons();
      }
    }
  } catch (err) {}
}
function _onFragmentLoadingCompleted(e) {
  try {
    const key = e.mediaType + "_" + e.request.index;
    if (e && e.request && fragmentLoadStarts[key]) {
      const loadInfo = fragmentLoadStarts[key];
      const endTime = performance.now();
      let clientMeasuredLatencyMs = Math.round(endTime - loadInfo.startTime);
      const serverUsed = loadInfo.serviceLocation;
      delete fragmentLoadStarts[key];
      if (isSimulationRunning) {
        if (simCurrentLat !== undefined && simCurrentLon !== undefined) {
          reportLatencyToSteering(
            simCurrentLat,
            simCurrentLon,
            clientMeasuredLatencyMs,
            serverUsed,
          );
        }
      }
    }
  } catch (err) {}
}
function _onContentSteeringRequestCompleted(e) {
  try {
    if (!e) return;
    document.getElementById(`steering-request-timestamp`).innerText =
      new Date().toLocaleTimeString();
    if (e.url)
      document.getElementById(`steering-request-url`).innerText =
        decodeURIComponent(e.url);
    if (e.currentSteeringResponseData) {
      const data = e.currentSteeringResponseData;
      const priority = data["PATHWAY-PRIORITY"] || data.pathwayPriority || [];
      document.getElementById(`steering-decision-display`).textContent =
        priority.map((p) => CACHE_COORDS[p]?.label || p).join(" > ");
      document.getElementById(`steering-pathway-cloning`).innerText =
        JSON.stringify(
          data["PATHWAY-CLONES"] || data.pathwayClones || [],
          null,
          2,
        );
    } else {
      document.getElementById(`steering-decision-display`).textContent =
        "N/A (No response data)";
      document.getElementById(`steering-pathway-cloning`).innerText = "N/A";
    }
  } catch (err) {}
}
function _createIcon(container, serviceLoc, domMap, prefix) {
  const span = document.createElement("span");
  span.id = `${prefix}-icon-${serviceLoc}`;
  const figure = document.createElement("figure");
  figure.className = "cdn-selection";
  const img = document.createElement("img");
  img.src = "assets/img/server.svg";
  img.alt = serviceLoc;
  img.className = "figure-img img-fluid cdn-selection";
  const figCaption = document.createElement("figcaption");
  figCaption.className = "figure-caption";
  figCaption.textContent = CACHE_COORDS[serviceLoc]?.label || serviceLoc;
  figure.append(img, figCaption);
  span.appendChild(figure);
  container.appendChild(span);
  domMap[serviceLoc] = img;
}
function _updateActiveServerIcons() {
  const activeServers = {};
  if (currentSegmentServiceLocation.audio)
    activeServers[currentSegmentServiceLocation.audio] = true;
  if (currentSegmentServiceLocation.video)
    activeServers[currentSegmentServiceLocation.video] = true;
  for (const serverName in cdnIconDomElements) {
    if (cdnIconDomElements.hasOwnProperty(serverName)) {
      cdnIconDomElements[serverName].src = activeServers[serverName]
        ? "assets/img/server-active.svg"
        : "assets/img/server.svg";
    }
  }
}
document.addEventListener("DOMContentLoaded", init);

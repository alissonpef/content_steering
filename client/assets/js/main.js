let player;
let currentSegmentServiceLocation = { audio: null, video: null };
let cdnIconDomElements = {};
let simTimer = null;
let simElapsedTime = 0;
let simMovementActive = false;
let simIntervalID_movement = null;
let movementStarted = false;
let simSpamActive_1 = false;
let simSpamActive_2 = false;
let simIntervalID_spam_1 = null;
let simIntervalID_spam_2 = null;
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
let currentDecisionId = null;
let isStalled = false;
let lastStallStart = 0;
let totalStallTime = 0;
let selectedStrategy = "";

const STRATEGY_LABELS = {
  epsilon_greedy: "Epsilon-Greedy",
  ucb1: "UCB1",
  linucb: "LinUCB",
  thompson_sampling: "Thompson Sampling",
  ppo_hybrid: "PPO Hybrid",
  sac_hybrid: "SAC Hybrid",
  random: "Random",
  best: "Best",
};
function stopInterval(intervalId) {
  if (intervalId != null) clearInterval(intervalId);
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

  player.extend(
    "RequestModifier",
    function () {
      return {
        modifyRequest: function (request) {
          return Promise.resolve(request);
        },
      };
    },
    true,
  );

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
  player.on(dashjs.MediaPlayer.events.BUFFER_EMPTY, () => {
    if (!isStalled) {
      isStalled = true;
      lastStallStart = performance.now();
    }
  });
  player.on(dashjs.MediaPlayer.events.BUFFER_LOADED, () => {
    if (isStalled) {
      isStalled = false;
      totalStallTime += performance.now() - lastStallStart;
    }
  });
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
  if (typeof DEFAULT_MANIFEST_URL !== "undefined") {
    document.getElementById("manifest").value = DEFAULT_MANIFEST_URL;
  }
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
  loadStrategiesFromBackend();
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
function loadStrategiesFromBackend() {
  fetch(`${STEERING_SERVER_URL}/strategies`)
    .then((r) => r.json())
    .then((data) => {
      const select = document.getElementById("simStrategy");
      select.innerHTML = "";
      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.disabled = true;
      defaultOpt.selected = true;
      defaultOpt.textContent = "— Select a strategy —";
      select.appendChild(defaultOpt);
      (data.strategies || []).forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = STRATEGY_LABELS[s] || s;
        select.appendChild(opt);
      });
      console.log("Strategies loaded from backend:", data.strategies);
    })
    .catch((err) => {
      console.warn("Could not load strategies from backend:", err);
      const select = document.getElementById("simStrategy");
      select.innerHTML = "";
      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.disabled = true;
      defaultOpt.selected = true;
      defaultOpt.textContent = "— Select a strategy —";
      select.appendChild(defaultOpt);
      for (const key in STRATEGY_LABELS) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = STRATEGY_LABELS[key];
        select.appendChild(opt);
      }
    });
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
  document.getElementById("simStrategy").addEventListener("change", (e) => {
    selectedStrategy = e.target.value;
    const badge = document.getElementById("strategyStatusBadge");
    if (badge) {
      badge.textContent = STRATEGY_LABELS[selectedStrategy] || selectedStrategy;
      badge.classList.remove("bg-secondary");
      badge.classList.add("bg-primary");
    }
    document.getElementById("button_StartControlledSim").disabled =
      !manifestSuccessfullyLoaded || !selectedStrategy;
  });
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
  stopInterval(simIntervalID_spam_1);
  simIntervalID_spam_1 = null;
  simSpamActive_1 = false;
  stopInterval(simIntervalID_spam_2);
  simIntervalID_spam_2 = null;
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
    !manifestSuccessfullyLoaded || !selectedStrategy;
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
  currentDecisionId = null;
  isStalled = false;
  totalStallTime = 0;
  document.getElementById("simMovementStatus").textContent = "Inactive";
  document.getElementById("simSpamStatus_1").textContent = "Inactive";
  document.getElementById("simSpamStatus_2").textContent = "Inactive";
  movementStarted = false;
  simSpamActive_1 = false;
  simSpamActive_2 = false;
}
function resetSimulationUIAndState() {
  stopCurrentSimulation(true);
  _resetUIOnly();
  document.getElementById("button_StartControlledSim").disabled =
    !manifestSuccessfullyLoaded || !selectedStrategy;
}
function _runSimulation() {
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
  stopInterval(simIntervalID_movement);
  simIntervalID_movement = null;
  stopInterval(simIntervalID_spam_1);
  simIntervalID_spam_1 = null;
  stopInterval(simIntervalID_spam_2);
  simIntervalID_spam_2 = null;
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
function startControlledSimulation() {
  if (!manifestSuccessfullyLoaded) {
    alert("Manifest not loaded. Please load an MPD first.");
    return;
  }
  if (!selectedStrategy) {
    alert("Please select an RL strategy before starting.");
    return;
  }
  if (isSimulationRunning) {
    return;
  }
  if (currentRunIndex === 0) {
    const runsInput = document.getElementById("simRuns");
    totalRunsToExecute = runsInput ? parseInt(runsInput.value) || 1 : 1;
    currentRunIndex = 1;
    document.getElementById("button_StartControlledSim").disabled = true;
    fetch(`${STEERING_SERVER_URL}/reset_simulation`, {
      method: "POST",
      body: JSON.stringify({ strategy: selectedStrategy }),
      headers: { "Content-type": "application/json; charset=UTF-8" },
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Initial backend reset:", data);
        if (player) {
          player.reset();
          setupPlayer();
          player.updateSettings({
            streaming: {
              buffer: { stableBufferTime: 4, bufferTimeAtTopQuality: 4 },
              abr: { autoSwitchBitrate: { video: false } },
            },
          });
          const mpdUrl = document.getElementById("manifest").value;
          if (mpdUrl) {
            const cb = () => {
              manifestSuccessfullyLoaded = true;
              _runSimulation();
              player.off(dashjs.MediaPlayer.events.MANIFEST_LOADED, cb);
            };
            player.on(dashjs.MediaPlayer.events.MANIFEST_LOADED, cb);
            player.attachSource(mpdUrl);
          } else {
            _runSimulation();
          }
        } else {
          _runSimulation();
        }
      })
      .catch((err) => {
        console.error("Failed initial reset:", err);
        document.getElementById("button_StartControlledSim").disabled = false;
        currentRunIndex = 0;
      });
  } else {
    _runSimulation();
  }
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
      fetch(`${STEERING_SERVER_URL}/reset_simulation`, {
        method: "POST",
        body: JSON.stringify({ strategy: selectedStrategy }),
        headers: { "Content-type": "application/json; charset=UTF-8" },
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
                abr: { autoSwitchBitrate: { video: false } },
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
            !manifestSuccessfullyLoaded || !selectedStrategy;
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
    spamTarget_1: document.getElementById("simSpamTarget_1").value,
    spamStartTime_1:
      parseInt(document.getElementById("simSpamStartTime_1").value) || 20,
    spamDuration_1:
      parseInt(document.getElementById("simSpamDuration_1").value) || 20,
    spamTarget_2: document.getElementById("simSpamTarget_2").value,
    spamStartTime_2:
      parseInt(document.getElementById("simSpamStartTime_2").value) || 80,
    spamDuration_2:
      parseInt(document.getElementById("simSpamDuration_2").value) || 20,
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
    cfg.spamTarget_1 !== "none" &&
    simElapsedTime >= cfg.spamStartTime_1 &&
    !simSpamActive_1 &&
    simElapsedTime < cfg.spamStartTime_1 + cfg.spamDuration_1
  ) {
    simSpamActive_1 = true;
    startSimulatedCacheSpam(cfg.spamTarget_1, 1);
    document.getElementById("simSpamStatus_1").textContent =
      `Spamming (${CACHE_COORDS[cfg.spamTarget_1]?.label || cfg.spamTarget_1})`;
  }
  if (
    simSpamActive_1 &&
    simElapsedTime >= cfg.spamStartTime_1 + cfg.spamDuration_1
  ) {
    stopInterval(simIntervalID_spam_1);
    simIntervalID_spam_1 = null;
    simSpamActive_1 = false;
    document.getElementById("simSpamStatus_1").textContent = "Inactive";
  }

  if (
    cfg.spamTarget_2 !== "none" &&
    simElapsedTime >= cfg.spamStartTime_2 &&
    !simSpamActive_2 &&
    simElapsedTime < cfg.spamStartTime_2 + cfg.spamDuration_2
  ) {
    simSpamActive_2 = true;
    startSimulatedCacheSpam(cfg.spamTarget_2, 2);
    document.getElementById("simSpamStatus_2").textContent =
      `Spamming (${CACHE_COORDS[cfg.spamTarget_2]?.label || cfg.spamTarget_2})`;
  }
  if (
    simSpamActive_2 &&
    simElapsedTime >= cfg.spamStartTime_2 + cfg.spamDuration_2
  ) {
    stopInterval(simIntervalID_spam_2);
    simIntervalID_spam_2 = null;
    simSpamActive_2 = false;
    document.getElementById("simSpamStatus_2").textContent = "Inactive";
  }

  let activeSpamTargets = [];
  if (simSpamActive_1) activeSpamTargets.push(cfg.spamTarget_1);
  if (simSpamActive_2) activeSpamTargets.push(cfg.spamTarget_2);

  reportLocationToSteering(
    simCurrentLat,
    simCurrentLon,
    activeSpamTargets.length > 0 ? activeSpamTargets : null,
  );
  fetch(`${STEERING_SERVER_URL}/sim_state`)
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
    console.warn(
      `[SPAM ${phaseId}] Target cache ${targetCacheName} not found.`,
    );
    if (phaseId === 1) {
      simSpamActive_1 = false;
      document.getElementById("simSpamStatus_1").textContent = "Error";
    } else {
      simSpamActive_2 = false;
      document.getElementById("simSpamStatus_2").textContent = "Error";
    }
    return;
  }
  const nodeMap = {
    "delivery-node-1": "node1",
    "delivery-node-2": "node2",
    "delivery-node-3": "node3",
  };
  const nodePath = nodeMap[targetCacheName] || "node1";
  const hostUrl = `${window.location.origin}/${nodePath}/Eldorado/4sec/avc/750000/seg-1.m4s?spam_ts=${Date.now()}&phase=${phaseId}`;

  const currentPhaseActiveFlag =
    phaseId === 1 ? () => simSpamActive_1 : () => simSpamActive_2;
  const setSpamIntervalId = (val) => {
    if (phaseId === 1) simIntervalID_spam_1 = val;
    else simIntervalID_spam_2 = val;
  };
  const currentSpamIntervalId =
    phaseId === 1 ? simIntervalID_spam_1 : simIntervalID_spam_2;
  const myIntervalClearer = () => {
    if (phaseId === 1) {
      stopInterval(simIntervalID_spam_1);
      simIntervalID_spam_1 = null;
    } else {
      stopInterval(simIntervalID_spam_2);
      simIntervalID_spam_2 = null;
    }
  };

  function sendSpamRequest() {
    if (isSimulationRunning && currentPhaseActiveFlag()) {
      fetch(hostUrl).catch(() => {});
    } else {
      myIntervalClearer();
    }
  }

  if (currentSpamIntervalId) clearInterval(currentSpamIntervalId);
  setSpamIntervalId(setInterval(sendSpamRequest, 100));
}

function reportLocationToSteering(lat, lon, spamTarget) {
  if (!isSimulationRunning || lat === undefined || lon === undefined) return;
  const payload = {
    time: simElapsedTime,
    lat: lat,
    long: lon,
    spam_target: spamTarget || null,
  };
  fetch(`${STEERING_SERVER_URL}/coords`, {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-type": "application/json; charset=UTF-8" },
  }).catch((error) => {});
}
function reportLatencyToSteering(
  lat,
  lon,
  clientMeasuredLatency,
  serverUsed,
  decisionId,
  stallTime,
  qualityLevel,
) {
  if (!isSimulationRunning || lat === undefined || lon === undefined) return;
  if (clientMeasuredLatency === undefined || serverUsed === undefined) return;
  const payload = {
    time: simElapsedTime,
    lat: lat,
    long: lon,
    rt: clientMeasuredLatency,
    server_used: serverUsed,
    decision_id: decisionId,
    stall_time: stallTime,
    quality_level: qualityLevel,
  };
  fetch(`${STEERING_SERVER_URL}/coords`, {
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
      abr: { autoSwitchBitrate: { video: false } },
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
              if (
                manifestSuccessfullyLoaded &&
                !isSimulationRunning &&
                selectedStrategy
              ) {
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
              !selectedStrategy;
          }
        } else {
          document.getElementById("button_StartControlledSim").disabled =
            !selectedStrategy;
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
          const stallToReport = totalStallTime;
          totalStallTime = 0;
          reportLatencyToSteering(
            simCurrentLat,
            simCurrentLon,
            clientMeasuredLatencyMs,
            serverUsed,
            currentDecisionId,
            stallToReport,
            player.getQualityFor("video"),
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
      if (data["DECISION-ID"]) {
        currentDecisionId = data["DECISION-ID"];
      }
      if (data["RL-QUALITY-LEVEL"] !== undefined) {
        player.setQualityFor("video", data["RL-QUALITY-LEVEL"]);
      }
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

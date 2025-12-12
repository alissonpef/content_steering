let player;
let currentSegmentServiceLocation = { audio: null, video: null };
let cdnIconDomElements = {};

let simTimer = null;
let simElapsedTime = 0;

let simMovementActive = false;
let simIntervalID_movement = null;
let movementStarted = false;

let simSpamActive_1 = false, simSpamActive_2 = false;
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

function stopInterval(intervalId) {
    if (intervalId != null) clearInterval(intervalId);
}

function setupPlayer() {
    const videoElement = document.querySelector("video");
    if (player) {
        if (onManifestLoadedCallback) player.off(dashjs.MediaPlayer.events.MANIFEST_LOADED, onManifestLoadedCallback);
        if (onManifestErrorCallback) player.off(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback);
        if (onStreamInitForPlay) player.off(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForPlay);
        if (onStreamInitForAutomaticPlay) player.off(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForAutomaticPlay);
        player.reset();
    }
    player = dashjs.MediaPlayer().create();
    player.initialize(videoElement, null, false);
    player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_STARTED, _onFragmentLoadingStarted);
    player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, _onFragmentLoadingCompleted);
    player.on(dashjs.MediaPlayer.events.CONTENT_STEERING_REQUEST_COMPLETED, _onContentSteeringRequestCompleted);
    player.on(dashjs.MediaPlayer.events.ERROR, (e) => {
        if (e.error && e.error.code &&
            (e.error.code === dashjs.MediaPlayer.errors.MANIFEST_LOADER_PARSING_FAILURE_ERROR_CODE ||
             e.error.code === dashjs.MediaPlayer.errors.MANIFEST_LOADER_LOADING_FAILURE_ERROR_CODE ||
             e.error.code === dashjs.MediaPlayer.errors.DOWNLOAD_ERROR_ID_MANIFEST)
        ) {
             manifestSuccessfullyLoaded = false;
             document.getElementById("button_StartControlledSim").disabled = true;
        }
    });
}

function init() {
    console.log("Initializing simulation...");
    if (typeof CACHE_COORDS === 'undefined') {
        console.error("CACHE_COORDS is not defined!");
        alert("Configuration error: CACHE_COORDS missing. Check config.js loading.");
        return;
    }
    if (typeof dashjs === 'undefined') {
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
    cdnContainer.innerHTML = ''; cdnIconDomElements = {};
    for (const cacheName in CACHE_COORDS) {
        _createIcon(cdnContainer, cacheName, cdnIconDomElements, "cdn");
    }
    _resetUIOnly();
    document.getElementById("button_StartControlledSim").disabled = true;
    document.getElementById("button_StopSim").disabled = true;
}

function setupEventListeners() {
    document.getElementById("load-button").addEventListener("click", _load);
    document.getElementById("button_StartControlledSim").addEventListener("click", startControlledSimulation);
    document.getElementById("button_StopSim").addEventListener("click", stopCurrentSimulation);
    document.getElementById("button_ResetSimUI").addEventListener("click", resetSimulationUIAndState);
    
    document.getElementById("simLoops").addEventListener("input", updateCalculatedDuration);

    const runModeRadios = document.querySelectorAll('input[name="runMode"]');
    runModeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'duration') {
                document.getElementById('durationInputGroup').style.display = 'block';
                document.getElementById('loopsInputGroup').style.display = 'none';
            } else {
                document.getElementById('durationInputGroup').style.display = 'none';
                document.getElementById('loopsInputGroup').style.display = 'block';
                updateCalculatedDuration();
            }
        });
    });
}



function updateCalculatedDuration() {
    const loops = parseInt(document.getElementById("simLoops").value) || 1;
    const infoSpan = document.getElementById("calculatedDurationInfo");
    
    if (player && player.isReady() && player.duration() && player.duration() > 0) {
        const videoDuration = player.duration();
        const totalDuration = Math.ceil(videoDuration * loops);
        if(infoSpan) infoSpan.innerText = `≈ ${totalDuration}s`;
    } else {
        if(infoSpan) infoSpan.innerText = "(Load video first)";
    }
}



function populateSelect(selectId, noneOptionText) {
    const selectElement = document.getElementById(selectId);
    selectElement.innerHTML = '';
    if (noneOptionText) {
        const noneOpt = document.createElement("option"); noneOpt.value = "none";
        noneOpt.textContent = noneOptionText; selectElement.appendChild(noneOpt);
    }
    for (const cacheName in CACHE_COORDS) {
        const option = document.createElement("option"); option.value = cacheName;
        option.textContent = CACHE_COORDS[cacheName].label; selectElement.appendChild(option);
    }
}

function stopCurrentSimulation(pausePlayerAndSeek = true, finishedNaturally = false) {
    // Don't immediately set isSimulationRunning = false - let final events process
    if (simTimer) { clearInterval(simTimer); simTimer = null; }

    stopInterval(simIntervalID_movement); simIntervalID_movement = null;

    simMovementActive = false;
    movementStarted = false;
    simSpamActive_1 = false; simSpamActive_2 = false;

    if (finishedNaturally && currentRunIndex < totalRunsToExecute) {
        // Intermediate run finished - prepare for next run WITHOUT pausing
        console.log(`Run ${currentRunIndex} finished. Preparing for Run ${currentRunIndex + 1}...`);
        document.getElementById("button_StartControlledSim").disabled = true;
        
        // Give time for last fragments to finish loading (500ms)
        setTimeout(() => {
            isSimulationRunning = false;
            
            // Only seek to beginning, don't pause yet
            if (player && player.isReady()) {
                player.seek(0);
            }

            setTimeout(() => {
                // Increment BEFORE calling reset to ensure proper log numbering
                currentRunIndex++;
                
                // Update UI to show which log file is being written
                const runIndicator = document.getElementById("runIndicator");
                if (runIndicator) {
                    runIndicator.textContent = `Preparing Run ${currentRunIndex}/${totalRunsToExecute}...`;
                    runIndicator.classList.remove("bg-info");
                    runIndicator.classList.add("bg-warning");
                }
                
                fetch("https://steering-service:30500/reset_simulation", { method: "POST" })
                .then(response => response.json())
                .then(data => {
                    console.log("Backend reset:", data);
                    console.log(`✓ Log file created: ${data.new_log || 'unknown'}`);
                    console.log(`Starting data collection for Run ${currentRunIndex}/${totalRunsToExecute}...`);
                    
                    // Force player reset to clear buffer and ensure fresh download
                    if (player) {
                        player.reset();
                        setupPlayer();
                        // Re-apply buffer settings
                        player.updateSettings({
                            'streaming': {
                                'buffer': {
                                    'stableBufferTime': 4,
                                    'bufferTimeAtTopQuality': 4
                                }
                            }
                        });
                        
                        // Re-attach source
                        const mpdUrl = document.getElementById("manifest").value;
                        if (mpdUrl) {
                            // Re-attach the manifest loaded callback to ensure we wait for it
                            const onManifestLoaded = () => {
                                console.log("Run " + currentRunIndex + ": Manifest loaded. Starting simulation.");
                                startControlledSimulation();
                                player.off(dashjs.MediaPlayer.events.MANIFEST_LOADED, onManifestLoaded);
                            };
                            player.on(dashjs.MediaPlayer.events.MANIFEST_LOADED, onManifestLoaded);
                            
                            player.attachSource(mpdUrl);
                        } else {
                            startControlledSimulation(); // Fallback
                        }
                    } else {
                        startControlledSimulation();
                    }
                })
                .catch(err => {
                    console.error("Failed to reset backend:", err);
                    alert("Failed to reset backend. Stopping sequence.");
                    currentRunIndex = 0;
                    isSimulationRunning = false;
                    document.getElementById("button_StartControlledSim").disabled = !manifestSuccessfullyLoaded;
                });
            }, 1500);
        }, 500); // Wait 500ms for final events to process
        return;
    } else {
        // Final completion or manual stop - give time for final events, then stop
        setTimeout(() => {
            isSimulationRunning = false;
            
            // Final completion or manual stop - NOW we pause
            if (pausePlayerAndSeek && player && manifestSuccessfullyLoaded) {
                if (player.isReady() && !player.isPaused()) {
                    player.pause();
                }
                if (player.isReady()){
                    player.seek(0);
                }
            }
            
            // This is the final completion (all runs done)
            if (currentRunIndex > 0 && currentRunIndex >= totalRunsToExecute) {
                 console.log("All runs completed.");
                 const runIndicator = document.getElementById("runIndicator");
                 if (runIndicator) {
                     runIndicator.textContent = "Completed";
                     runIndicator.classList.remove("bg-info");
                     runIndicator.classList.add("bg-success");
                 }
            }
            currentRunIndex = 0;
        }, 500); // Wait 500ms for final events
    }

    document.getElementById("button_StartControlledSim").disabled = !manifestSuccessfullyLoaded;
    document.getElementById("button_StopSim").disabled = true;
    document.getElementById("simMovementStatus").textContent = "Inactive";
    document.getElementById("simSpamStatus_1").textContent = "Inactive";
    document.getElementById("simSpamStatus_2").textContent = "Inactive";
}

function _resetUIOnly() {
    simElapsedTime = 0;
    document.getElementById("simCurrentTimeDisplay").textContent = "0";
    const initialLatVal = parseFloat(document.getElementById("initialSimLat").value);
    const initialLonVal = parseFloat(document.getElementById("initialSimLon").value);
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

    // Initialize runs if this is a fresh start (not a recursive call)
    if (currentRunIndex === 0) {
        const runsInput = document.getElementById("simRuns");
        totalRunsToExecute = runsInput ? (parseInt(runsInput.value) || 1) : 1;
        currentRunIndex = 1;
        
        // Reset backend ONCE at the start to create the first log
        fetch("https://steering-service:30500/reset_simulation", { method: "POST" })
            .then(response => response.json())
            .then(data => console.log("Initial backend reset:", data))
            .catch(err => console.error("Failed initial reset:", err));
    }
    console.log(`Starting Run ${currentRunIndex} of ${totalRunsToExecute}`);

    // Update Run Indicator
    const runIndicator = document.getElementById("runIndicator");
    if (runIndicator) {
        runIndicator.style.display = "inline-block";
        runIndicator.textContent = `Run ${currentRunIndex}/${totalRunsToExecute}`;
        runIndicator.classList.remove("bg-warning", "bg-success");
        runIndicator.classList.add("bg-info");
    }

    simElapsedTime = 0;
    simMovementActive = false; movementStarted = false;
    simSpamActive_1 = false; simSpamActive_2 = false;
    simSpamEventSent_1 = false;
    simSpamEventSent_2 = false;
    stopInterval(simIntervalID_movement); simIntervalID_movement = null;
    if (simTimer) { clearInterval(simTimer); simTimer = null; }
    document.getElementById("simCurrentTimeDisplay").textContent = "0";
    isSimulationRunning = true;
    fragmentLoadStarts = {};

    function attemptPlayAfterStreamReady() {
         if (player.getActiveStream() && player.isReady() && manifestSuccessfullyLoaded) {
            player.seek(0);
            // Do not play yet, wait for button click
            // player.play(); 
        }
    }

    if (player.getActiveStream() && player.isReady()) {
        attemptPlayAfterStreamReady();
    } else if (player.isReady()) {
        if (onStreamInitForPlay && player) player.off(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForPlay);
        onStreamInitForPlay = function() {
            if (!isSimulationRunning) return;
            attemptPlayAfterStreamReady();
            if (player) player.off(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForPlay);
        };
        player.on(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForPlay, null, { once: true });
    } else {
        isSimulationRunning = false;
        return;
    }
    document.getElementById("button_StartControlledSim").disabled = false;
    document.getElementById("button_StopSim").disabled = false;

    // Determine duration based on mode
    let duration = 180;
    const runMode = document.querySelector('input[name="runMode"]:checked').value;
    if (runMode === 'duration') {
        duration = parseInt(document.getElementById("simDuration").value) || 180;
    } else {
        const loops = parseInt(document.getElementById("simLoops").value) || 1;
        const videoDuration = player.duration();
        if (videoDuration && videoDuration > 0) {
            duration = Math.ceil(videoDuration * loops);
        } else {
            // Fallback if duration is not available yet
            duration = 180; 
            // Try to update it if it becomes available later? 
            // For now, just warn or accept it might be inaccurate if not loaded.
        }
    }

    // Start playback
    player.play();
    console.log("Simulation started. Duration:", duration);

    const movementTarget = document.getElementById("simMovementTarget").value;
    const movementStartTime = parseInt(document.getElementById("simMovementStartTime").value) || 60;
    const desiredMovementDuration = parseInt(document.getElementById("simMovementDuration").value) || 60;
    const spamTarget_1 = document.getElementById("simSpamTarget_1").value;
    const spamStartTime_1 = parseInt(document.getElementById("simSpamStartTime_1").value) || 60;
    const spamDuration_1_val = parseInt(document.getElementById("simSpamDuration_1").value) || 120;
    const spamTarget_2 = document.getElementById("simSpamTarget_2").value;
    const spamStartTime_2 = parseInt(document.getElementById("simSpamStartTime_2").value) || 120;
    const spamDuration_2_val = parseInt(document.getElementById("simSpamDuration_2").value) || 60;

    simCurrentLat = parseFloat(document.getElementById("initialSimLat").value);
    simCurrentLon = parseFloat(document.getElementById("initialSimLon").value);
    if(isNaN(simCurrentLat)) simCurrentLat = -23.0;
    if(isNaN(simCurrentLon)) simCurrentLon = -47.0;
    document.getElementById("current-latitude").value = simCurrentLat.toFixed(5);
    document.getElementById("current-longitude").value = simCurrentLon.toFixed(5);
    document.getElementById("simMovementStatus").textContent = "Inactive";
    document.getElementById("simSpamStatus_1").textContent = "Inactive";
    document.getElementById("simSpamStatus_2").textContent = "Inactive";

    simTimer = setInterval(() => {
        if (!isSimulationRunning) { clearInterval(simTimer); simTimer = null; return; }
        simElapsedTime++;
        document.getElementById("simCurrentTimeDisplay").textContent = simElapsedTime;

        if (movementTarget !== "none" && simElapsedTime >= movementStartTime && !movementStarted) {
            movementStarted = true;
            simMovementActive = true;
            let timeRemainingInSim = duration - simElapsedTime;
            let effectiveMoveDuration = Math.max(1, Math.min(desiredMovementDuration, timeRemainingInSim));
            startSimulatedMovement(movementTarget, simCurrentLat, simCurrentLon, effectiveMoveDuration);
            document.getElementById("simMovementStatus").textContent = `Moving (to ${CACHE_COORDS[movementTarget]?.label || movementTarget})`;
        }
        if (spamTarget_1 !== "none" && simElapsedTime >= spamStartTime_1 && !simSpamEventSent_1) {
            simSpamActive_1 = true;
            simSpamEventSent_1 = true;
            startSimulatedCacheSpam(spamTarget_1, 1);
            document.getElementById("simSpamStatus_1").textContent = `Spamming (${CACHE_COORDS[spamTarget_1]?.label || spamTarget_1})`;
        }
        if (simSpamActive_1 && simElapsedTime >= (spamStartTime_1 + spamDuration_1_val)) {
            simSpamActive_1 = false;
            document.getElementById("simSpamStatus_1").textContent = "Inactive";
        }
        if (spamTarget_2 !== "none" && simElapsedTime >= spamStartTime_2 && !simSpamEventSent_2) {
            simSpamActive_2 = true;
            simSpamEventSent_2 = true;
            startSimulatedCacheSpam(spamTarget_2, 2);
            document.getElementById("simSpamStatus_2").textContent = `Spamming (${CACHE_COORDS[spamTarget_2]?.label || spamTarget_2})`;
        }
        if (simSpamActive_2 && simElapsedTime >= (spamStartTime_2 + spamDuration_2_val)) {
            simSpamActive_2 = false;
            document.getElementById("simSpamStatus_2").textContent = "Inactive";
        }

        reportLocationToSteering(simCurrentLat, simCurrentLon);
        if (simElapsedTime >= duration) {
            stopCurrentSimulation(true, true);
        }
    }, 1000);
}

function startSimulatedMovement(targetCacheName, initialClientLat, initialClientLon, moveDurationSec) {
    if (!CACHE_COORDS[targetCacheName]) {
        simMovementActive = false; document.getElementById("simMovementStatus").textContent = "Error";
        return;
    }
    const targetCoord = CACHE_COORDS[targetCacheName];
    const totalSteps = moveDurationSec > 0 ? Math.max(1, Math.floor(moveDurationSec)) : 1;

    const stepLat = (targetCoord.lat - initialClientLat) / totalSteps;
    const stepLon = (targetCoord.lon - initialClientLon) / totalSteps;
    let stepsTaken = 0;

    if (simIntervalID_movement) clearInterval(simIntervalID_movement);
    simIntervalID_movement = null;

    const intervalFunc = () => {
        if (!isSimulationRunning || !simMovementActive || !simIntervalID_movement) {
            stopInterval(simIntervalID_movement); simIntervalID_movement = null;
            return;
        }
        if (stepsTaken < totalSteps) {
            simCurrentLat += stepLat;
            simCurrentLon += stepLon;
            stepsTaken++;
            document.getElementById("current-latitude").value = simCurrentLat.toFixed(5);
            document.getElementById("current-longitude").value = simCurrentLon.toFixed(5);
        } else {
            simCurrentLat = targetCoord.lat;
            simCurrentLon = targetCoord.lon;
            document.getElementById("current-latitude").value = simCurrentLat.toFixed(5);
            document.getElementById("current-longitude").value = simCurrentLon.toFixed(5);
            stopInterval(simIntervalID_movement); simIntervalID_movement = null;
            simMovementActive = false;
            if(isSimulationRunning) document.getElementById("simMovementStatus").textContent = "Reached Target";
        }
    };
    simIntervalID_movement = setInterval(intervalFunc, 1000);
}

function startSimulatedCacheSpam(targetCacheName, phaseId) {
    if (!CACHE_COORDS[targetCacheName]) {
        if (phaseId === 1) { simSpamActive_1 = false; document.getElementById("simSpamStatus_1").textContent = "Error"; }
        else { simSpamActive_2 = false; document.getElementById("simSpamStatus_2").textContent = "Error"; }
        return;
    }
    const spamDurationElementId = `simSpamDuration_${phaseId}`;
    const spamDurationValue = parseInt(document.getElementById(spamDurationElementId)?.value);
     if (isNaN(spamDurationValue) || spamDurationValue <=0 ) {
         if (phaseId === 1) { simSpamActive_1 = false; document.getElementById("simSpamStatus_1").textContent = "Error: Invalid Duration"; }
         else { simSpamActive_2 = false; document.getElementById("simSpamStatus_2").textContent = "Error: Invalid Duration"; }
        return;
    }

    const payload = {
        server_name: targetCacheName,
        factor: 15.0,
        duration_seconds: spamDurationValue
    };
    fetch("https://steering-service:30500/latency_event", {
        method: "POST",
        body: JSON.stringify(payload),
        headers: { "Content-type": "application/json; charset=UTF-8" }
    })
    .then(response => response.text().then(text => ({ok: response.ok, text, status: response.status})))
    .then(data => {
    })
    .catch(err => {});
}

function reportLocationToSteering(lat, lon) {
    if (!isSimulationRunning || lat === undefined || lon === undefined) return;
    const payload = { time: simElapsedTime, lat: lat, long: lon };
    fetch("https://steering-service:30500/coords", {
        method: "POST", body: JSON.stringify(payload),
        headers: { "Content-type": "application/json; charset=UTF-8" }
    })
    .catch(error => {});
}

function reportLatencyToSteering(lat, lon, clientMeasuredLatency, serverUsed) {
    if (!isSimulationRunning || lat === undefined || lon === undefined) return;
    if (clientMeasuredLatency === undefined || serverUsed === undefined) return;

    const payload = {
        time: simElapsedTime,
        lat: lat,
        long: lon,
        rt: clientMeasuredLatency,
        server_used: serverUsed
    };
    fetch("https://steering-service:30500/coords", {
        method: "POST", body: JSON.stringify(payload),
        headers: { "Content-type": "application/json; charset=UTF-8" }
    })
    .then(response => response.text().then(text => ({ok: response.ok, status: response.status, text})))
    .then(data => {
    })
    .catch(error => {});
}

function _load() {
    let newMpdUrl = document.getElementById("manifest").value;
    if (!newMpdUrl) { alert("Please enter an MPD URL."); return; }

    manifestSuccessfullyLoaded = false;
    document.getElementById("button_StartControlledSim").disabled = true;
    if (isSimulationRunning) stopCurrentSimulation(true);

    setupPlayer();
    
    // Limit buffer to prevent "loading ahead of time" and ensure consistent runs
    player.updateSettings({
        'streaming': {
            'buffer': {
                'stableBufferTime': 4,
                'bufferTimeAtTopQuality': 4
            }
        }
    });

    _resetUIOnly();

    try {
        player.attachSource(newMpdUrl);

        onManifestLoadedCallback = function(e) {
            if(e.error) {
                alert("Error loading manifest: " + (e.error.message || e.error));
                manifestSuccessfullyLoaded = false;
                document.getElementById("button_StartControlledSim").disabled = true;
            } else {
                manifestSuccessfullyLoaded = true;
                const autoStartEnabled = document.getElementById("autoStartCheckbox").checked;

                if (autoStartEnabled) {
                     if (onStreamInitForAutomaticPlay && player) player.off(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForAutomaticPlay);
                    onStreamInitForAutomaticPlay = function() {
                        if (player.getActiveStream() && player.isReady()) {
                            if (manifestSuccessfullyLoaded && !isSimulationRunning) {
                                startControlledSimulation();
                            }
                        }
                        if (player) player.off(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForAutomaticPlay);
                    };
                    if (player.getActiveStream() && player.isReady()) {
                        onStreamInitForAutomaticPlay();
                    } else if (player.isReady()){
                        player.on(dashjs.MediaPlayer.events.STREAM_INITIALIZED, onStreamInitForAutomaticPlay, null, { once: true });
                    } else {
                        document.getElementById("button_StartControlledSim").disabled = false;
                    }
                } else {
                    document.getElementById("button_StartControlledSim").disabled = false;
                }
                // Update duration info if in loops mode
                updateCalculatedDuration();
            }
             if (player) player.off(dashjs.MediaPlayer.events.MANIFEST_LOADED, onManifestLoadedCallback);
        };
        player.on(dashjs.MediaPlayer.events.MANIFEST_LOADED, onManifestLoadedCallback, null, {once: true});

        if (onManifestErrorCallback && player) player.off(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback);
        onManifestErrorCallback = function(e) {
            if (e.error && e.error.code && (
                e.error.code === dashjs.MediaPlayer.errors.MANIFEST_LOADER_PARSING_FAILURE_ERROR_CODE ||
                e.error.code === dashjs.MediaPlayer.errors.MANIFEST_LOADER_LOADING_FAILURE_ERROR_CODE ||
                e.error.code === dashjs.MediaPlayer.errors.DOWNLOAD_ERROR_ID_MANIFEST
                )) {
                alert("Failed to load or parse manifest: " + (e.error.message || e.error));
                manifestSuccessfullyLoaded = false;
                document.getElementById("button_StartControlledSim").disabled = true;
            }
            if (player) player.off(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback);
        };
        player.on(dashjs.MediaPlayer.events.ERROR, onManifestErrorCallback, null, {once: true});

    } catch (error) {
         alert("Error setting up player with MPD.");
         manifestSuccessfullyLoaded = false;
         document.getElementById("button_StartControlledSim").disabled = true;
    }
}

function _onFragmentLoadingStarted(e) {
    try {
        if (e && e.mediaType && (e.mediaType === "video" || e.mediaType === "audio") && e.request) {
            const key = e.mediaType + "_" + e.request.index;
            if (e.request.serviceLocation) {
                fragmentLoadStarts[key] = {
                    startTime: performance.now(),
                    serviceLocation: e.request.serviceLocation,
                    url: e.request.url
                };
                currentSegmentServiceLocation[e.mediaType] = e.request.serviceLocation;
                _updateActiveServerIcons();
            }
        }
    } catch (err) { }
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
                    reportLatencyToSteering(simCurrentLat, simCurrentLon, clientMeasuredLatencyMs, serverUsed);
                }
            }
        }
    } catch (err) { }
}

function _onContentSteeringRequestCompleted(e) {
    try {
        if (!e) return;
        document.getElementById(`steering-request-timestamp`).innerText = new Date().toLocaleTimeString();
        if (e.url) document.getElementById(`steering-request-url`).innerText = decodeURIComponent(e.url);
        if (e.currentSteeringResponseData) {
            const data = e.currentSteeringResponseData;
            const priority = data["PATHWAY-PRIORITY"] || data.pathwayPriority || [];
            document.getElementById(`steering-decision-display`).textContent = priority.map(p => CACHE_COORDS[p]?.label || p).join(' > ');
            document.getElementById(`steering-pathway-cloning`).innerText = JSON.stringify(data["PATHWAY-CLONES"] || data.pathwayClones || [], null, 2);
        } else {
             document.getElementById(`steering-decision-display`).textContent = "N/A (No response data)";
             document.getElementById(`steering-pathway-cloning`).innerText = "N/A";
        }
    } catch (err) { }
}

function _createIcon(container, serviceLoc, domMap, prefix) {
    const span = document.createElement("span");
    span.id = `${prefix}-icon-${serviceLoc}`;
    const figure = document.createElement("figure"); figure.className = "cdn-selection";
    const img = document.createElement("img");
    img.src = "assets/img/server.svg"; img.alt = serviceLoc;
    img.className = "figure-img img-fluid cdn-selection";
    const figCaption = document.createElement("figcaption"); figCaption.className = "figure-caption";
    figCaption.textContent = CACHE_COORDS[serviceLoc]?.label || serviceLoc;
    figure.append(img, figCaption); span.appendChild(figure); container.appendChild(span);
    domMap[serviceLoc] = img;
}

function _updateActiveServerIcons() {
    const activeServers = {};
    if (currentSegmentServiceLocation.audio) activeServers[currentSegmentServiceLocation.audio] = true;
    if (currentSegmentServiceLocation.video) activeServers[currentSegmentServiceLocation.video] = true;
    for (const serverName in cdnIconDomElements) {
        if (cdnIconDomElements.hasOwnProperty(serverName)) {
            cdnIconDomElements[serverName].src = activeServers[serverName] ? "assets/img/server-active.svg" : "assets/img/server.svg";
        }
    }
}

document.addEventListener("DOMContentLoaded", init);

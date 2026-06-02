const queryParams = new URLSearchParams(window.location.search);

var CACHE_COORDS = {
  "delivery-node-1": { lat: -23.0, lon: -47.0, label: "Cache 1 (BR)" },
  "delivery-node-2": { lat: -33.0, lon: -71.0, label: "Cache 2 (CL)" },
  "delivery-node-3": { lat: 5.0, lon: -74.0, label: "Cache 3 (CO)" },
};

var STEERING_SERVER_URL = `${window.location.origin}/steering`;
var DEFAULT_MANIFEST_URL = `${window.location.origin}/node1/Eldorado/4sec/avc/manifest.mpd`;

console.log("Config loaded (kubernetes mode).");

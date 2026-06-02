#!/bin/bash

set -e

KIND_CLUSTER_NAME="kind"

log() {
  echo -e "\033[1;34m[STOP]\033[0m $1"
}

log "Deleting Kind cluster '$KIND_CLUSTER_NAME'..."
kind delete cluster --name "$KIND_CLUSTER_NAME"

log "Cleanup completed successfully!"

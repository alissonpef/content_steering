#!/bin/bash

COMPOSE_FILE="./streaming-service/docker-compose.yml"

echo "--- Stopping Cache Server Containers ---"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: docker-compose.yml not found at $COMPOSE_FILE"
    exit 1
fi

sudo docker compose -f "$COMPOSE_FILE" down

echo "--- All streaming services have been stopped and removed. ---"
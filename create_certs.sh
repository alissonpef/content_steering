#!/bin/bash

set -e

SERVICES=(
    "video-streaming-cache-1"
    "video-streaming-cache-2"
    "video-streaming-cache-3"
    "steering-service"
)

echo "=================================================="
echo " Verifying the local mkcert CA installation... "
echo "=================================================="
mkcert -install
echo ""

for SERVICE_NAME in "${SERVICES[@]}"; do
    DEST_DIR=""

    if [[ "$SERVICE_NAME" == "steering-service" ]]; then
        DEST_DIR="./steering-service/certs"
    elif [[ "$SERVICE_NAME" == "video-streaming-cache-"* ]]; then
        DEST_DIR="./streaming-service/certs"
    else
        echo "WARNING: Unknown service name: '$SERVICE_NAME'. Skipping."
        continue
    fi

    echo "--------------------------------------------------"
    echo "  Creating certificate for: $SERVICE_NAME"
    echo "  Destination directory:   $DEST_DIR"
    echo "--------------------------------------------------"

    mkdir -p "$DEST_DIR"

    echo "--> Generating certificate with mkcert..."
    mkcert "$SERVICE_NAME"

    echo "--> Moving certificate files..."
    mv "./${SERVICE_NAME}.pem" "$DEST_DIR/"
    mv "./${SERVICE_NAME}-key.pem" "$DEST_DIR/"

    echo "--> Success! Certificate for '$SERVICE_NAME' created in $DEST_DIR"
    echo ""
done

echo "=================================================="
echo "   ALL CERTIFICATES WERE CREATED SUCCESSFULLY!   "
echo "=================================================="
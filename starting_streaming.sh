#!/bin/bash

echo "--- Starting/Verifying Cache Servers (Docker Compose) ---"
docker compose -f ./streaming-service/docker-compose.yml up -d
echo "Docker Compose command executed. Verifying container status..."

if ! docker ps --filter "name=video-streaming-cache-1" --filter "status=running" --quiet &>/dev/null || \
   ! docker ps --filter "name=video-streaming-cache-2" --filter "status=running" --quiet &>/dev/null || \
   ! docker ps --filter "name=video-streaming-cache-3" --filter "status=running" --quiet &>/dev/null; then
    echo "Error: Not all cache containers are running. Please check docker logs."
    exit 1
fi
echo "All expected cache containers are running."

echo ""
echo "--- Obtaining IP Addresses of Cache Containers ---"
IP_CACHE1=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-1)
IP_CACHE2=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-2)
IP_CACHE3=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-3)

if [ -z "$IP_CACHE1" ] || [ -z "$IP_CACHE2" ] || [ -z "$IP_CACHE3" ]; then
    echo "Error obtaining IP addresses for one or more containers."
    exit 1
fi
echo "IPs obtained: Cache1=$IP_CACHE1, Cache2=$IP_CACHE2, Cache3=$IP_CACHE3"

echo ""
echo "--- Updating /etc/hosts File ---"
HOSTS_FILE="/etc/hosts"
START_MARKER="# START Content Steering entries"
END_MARKER="# END Content Steering entries"
BACKUP_FILE="/etc/hosts.bak_content_steering_$(date +%Y%m%d_%H%M%S)"

cp "$HOSTS_FILE" "$BACKUP_FILE"
echo "Backed up /etc/hosts to $BACKUP_FILE"

sed -i "/$START_MARKER/,/$END_MARKER/d" "$HOSTS_FILE"

sed -i "/video-streaming-cache-1/d" "$HOSTS_FILE"
sed -i "/video-streaming-cache-2/d" "$HOSTS_FILE"
sed -i "/video-streaming-cache-3/d" "$HOSTS_FILE"
sed -i "/steering-service/d" "$HOSTS_FILE"
echo "Cleaned up any old content-steering entries from $HOSTS_FILE."

echo "Adding new entries..."
cat << EOF >> "$HOSTS_FILE"

$START_MARKER
$IP_CACHE1    video-streaming-cache-1
$IP_CACHE2    video-streaming-cache-2
$IP_CACHE3    video-streaming-cache-3
127.0.0.1    steering-service
$END_MARKER
EOF

echo "Successfully updated $HOSTS_FILE."
echo ""
echo "--- Initial Setup Completed Successfully ---"
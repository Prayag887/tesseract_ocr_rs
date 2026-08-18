#!/bin/bash
# Resource usage for the whole OCR stack (gateway + nid + passport + citizenship).
export DOCKER_CONFIG=${DOCKER_CONFIG:-/tmp/dockercfg}
SVCS="ocr-gateway nid-scan-service passport-scan-service citizenship-scan-service"

echo "=== LIVE USAGE ==="
docker stats --no-stream \
  --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}" $SVCS

echo
echo "=== TOTAL ==="
docker stats --no-stream --format "{{.MemUsage}}" $SVCS \
  | awk '{v=$1; u=$1;
          sub(/[A-Za-z]+$/,"",v);
          if (u ~ /GiB/) v*=1024; else if (u ~ /KiB/) v/=1024;
          t+=v}
         END {printf "  memory: %.0f MiB (%.2f GiB)\n", t, t/1024}'

echo
echo "=== IMAGE / DISK ==="
docker images --format "table {{.Repository}}\t{{.Size}}" \
  | grep -E "ocr-|REPOSITORY"
echo
docker system df | head -5

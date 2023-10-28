#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
rm -rf /workspace && \
  ln -s /runpod-volume /workspace

echo "Starting Clothes Segment API"
source /workspace/venv/bin/activate
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
export HF_HOME="/workspace"
python /workspace/webserver_handler/server.py > /workspace/logs/segmentclothes.log 2>&1 &
deactivate

echo "Starting RunPod Handler"
python3 -u /rp_handler.py
#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="snap-harvester.service"
LOG_TAG="HEARTBEAT"

if ! journalctl -u "${SERVICE_NAME}" --since "5 minutes ago" | grep -q "${LOG_TAG}"; then
  systemctl restart "${SERVICE_NAME}"
fi


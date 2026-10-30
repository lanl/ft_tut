#!/usr/bin/env bash
set -euo pipefail

supervisord -c /etc/supervisor/supervisord.conf

for _ in $(seq 1 50); do
  if [[ -S /var/run/supervisor.sock ]]; then
    break
  fi
  sleep 0.2
done

if [[ $# -gt 0 ]]; then
  "$@"
fi

exec tail -f /dev/null

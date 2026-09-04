#!/bin/bash
# Full teardown for the Goodway kiosk demo. Called from clear-demo.desktop
# or manually by the demo operator. Ordering matters: kill the restart
# loop first so it cannot respawn the browser we are about to close.
set -uo pipefail

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

APP_DIR="$USER_HOME/aries-vision-vlm-demo"
STATE_DIR="$USER_HOME/.local/state/aries-vision-vlm-demo"
BROWSER_PROFILE_DIR="$STATE_DIR/browser-profile"
PID_FILE="$STATE_DIR/run.pid"

# 1) Stop the browser restart loop first so it cannot respawn after step 2.
# Prefer the PID file that run.sh writes on startup; command-line pattern
# matches are unreliable because autostart launches run.sh as `./run.sh`
# under bash and the path string never appears in ps.
if [ -f "$PID_FILE" ]; then
  RUN_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$RUN_PID" ] && kill -0 "$RUN_PID" 2>/dev/null; then
    kill "$RUN_PID" 2>/dev/null || true
    # Wait briefly for graceful exit, then force.
    for _ in 1 2 3 4 5; do
      kill -0 "$RUN_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "$RUN_PID" 2>/dev/null && kill -KILL "$RUN_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

# 2) Kill the kiosk browser. Match on the private profile path so an
# unrelated chromium the operator may have open is left alone.
pkill -f "user-data-dir=$BROWSER_PROFILE_DIR" 2>/dev/null || true
pkill -f "profile $BROWSER_PROFILE_DIR" 2>/dev/null || true

# 3) Bring the docker stack down. `docker compose down` cleans only this
# project's containers/network; fall back to stopping every container if
# the compose file is unreachable for some reason.
if [ -d "$APP_DIR" ] && [ -f "$APP_DIR/docker-compose.yml" ]; then
  (cd "$APP_DIR" && docker compose down --remove-orphans) || true
else
  RUNNING_IDS="$(docker ps -q 2>/dev/null || true)"
  if [ -n "$RUNNING_IDS" ]; then
    docker stop $RUNNING_IDS || true
  fi
fi

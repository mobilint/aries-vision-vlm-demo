#!/bin/bash
# Kiosk entry point for the Goodway demo. Brings the docker stack up in the
# background, waits for the frontend to serve, then keeps a kiosk browser
# open in a restart loop so a crashed or force-closed browser recovers on
# its own. Docker teardown is handled separately by stop.sh (invoked via
# clear-demo.desktop) so operator-controlled shutdown is the only path.
set -euo pipefail

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

APP_DIR="$USER_HOME/aries-vision-vlm-demo"

cd "$APP_DIR"

STATE_DIR="$USER_HOME/.local/state/aries-vision-vlm-demo"
mkdir -p "$STATE_DIR"
LOG_FILE="$STATE_DIR/kiosk.log"
BROWSER_PROFILE_DIR="$STATE_DIR/browser-profile"
PID_FILE="$STATE_DIR/run.pid"
mkdir -p "$BROWSER_PROFILE_DIR"
exec >>"$LOG_FILE" 2>&1
echo "===== $(date -Is) run.sh starting (pid $$) ====="

# Record our PID so stop.sh can kill exactly this loop instead of
# guessing via pkill -f (autostart launches us as `./run.sh` under bash,
# so command-line pattern matches are fragile).
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

# Pick the core-allocation profile that matches how many MLA100 cards
# mbltml sees at boot. If detection fails (venv missing, mbltml import
# fails, etc.) we fall back to the 2-card profile because that is the
# competition target - a 1-card rig will surface a clear error at load
# time, whereas silently falling back to 1-card on a 2-card rig would
# just halve NPU utilization without a visible cause.
detect_card_count() {
  local venv_python="$APP_DIR/backend_vlm/.venv/bin/python"
  if [ ! -x "$venv_python" ]; then
    return 1
  fi
  "$venv_python" - <<'PY' 2>/dev/null
try:
    import mbltml
    try:
        mbltml.mbltmlInitDevices({mbltml.MBLTML_DEVICE_ARIES})
    except AttributeError:
        mbltml.mbltmlInit()
    print(int(mbltml.mbltmlGetDeviceCount()))
except Exception:
    pass
PY
}

CARD_COUNT="$(detect_card_count || true)"
case "$CARD_COUNT" in
  1) PROFILE=1card ;;
  2) PROFILE=2card ;;
  *)
    echo "$(date -Is) unexpected MLA100 card count '$CARD_COUNT', falling back to 2card profile" >&2
    PROFILE=2card
    ;;
esac

PROFILE_FILE="$APP_DIR/core_allocation.$PROFILE.yaml"
if [ ! -f "$PROFILE_FILE" ]; then
  echo "$(date -Is) ERROR: profile file $PROFILE_FILE not found" >&2
  exit 1
fi
cp "$PROFILE_FILE" "$APP_DIR/core_allocation.yaml"
echo "$(date -Is) using core allocation profile: $PROFILE (detected cards: $CARD_COUNT)"

docker compose up -d --remove-orphans

FRONTEND_URL="http://localhost:3000"
for _ in $(seq 1 180); do
  if curl -sf --max-time 2 "$FRONTEND_URL" >/dev/null; then
    break
  fi
  sleep 1
done

BROWSER=""
for candidate in chromium chromium-browser google-chrome google-chrome-stable; do
  if command -v "$candidate" >/dev/null 2>&1; then
    BROWSER="$candidate"
    break
  fi
done

if [ -z "$BROWSER" ] && command -v firefox >/dev/null 2>&1; then
  BROWSER="firefox"
fi

if [ -z "$BROWSER" ]; then
  echo "ERROR: no kiosk-capable browser found (chromium / google-chrome / firefox)" >&2
  exit 1
fi

# Restart loop: if the browser crashes or is force-closed, respawn after a
# short cooldown. The only intended shutdown path is stop.sh (via the Clear
# Demo desktop shortcut) or reboot, so we never exit voluntarily.
while true; do
  echo "===== $(date -Is) launching $BROWSER ====="
  if [ "$BROWSER" = "firefox" ]; then
    firefox --kiosk --profile "$BROWSER_PROFILE_DIR" "$FRONTEND_URL" || true
  else
    # --user-data-dir isolates this kiosk instance from any chromium the
    # operator may already have open on the same machine; without it, a
    # running chromium receives the URL as a new tab and the launcher
    # returns immediately, sending the restart loop into a tab-spam storm.
    "$BROWSER" \
      --kiosk \
      --user-data-dir="$BROWSER_PROFILE_DIR" \
      --noerrdialogs \
      --disable-infobars \
      --disable-session-crashed-bubble \
      --disable-translate \
      --no-first-run \
      --check-for-update-interval=31536000 \
      "$FRONTEND_URL" || true
  fi
  echo "===== $(date -Is) browser exited, respawning ====="
  sleep 2
done

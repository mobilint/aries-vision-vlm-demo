#!/bin/bash
# Kiosk entry point for the Goodway demo. Brings the docker stack up in the
# background, waits for every backend the dashboard needs, then keeps a kiosk
# browser open in a restart loop so a crashed or force-closed browser recovers
# on its own. Docker teardown is handled separately by stop.sh (invoked via
# clear-demo.desktop) so operator-controlled shutdown is the only path.
#
# Must be run from the machine's own desktop session. Over SSH there is no
# display, every browser start fails instantly, and the restart loop would
# spin forever.
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
# Snap-confined Firefox/Chromium cannot access hidden top-level dirs such as
# ~/.local, so keep the kiosk profile under the (visible) app directory.
BROWSER_PROFILE_DIR="$APP_DIR/.kiosk-browser-profile"
PID_FILE="$STATE_DIR/run.pid"
mkdir -p "$BROWSER_PROFILE_DIR"

# Refuse to start a second loop. Autostart plus a manual icon click would
# otherwise leave two loops launching browsers into the same profile, which on
# screen looks like the dashboard flickering in and out every few seconds.
if [ -f "$PID_FILE" ]; then
  OTHER_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$OTHER_PID" ] && [ "$OTHER_PID" != "$$" ] && kill -0 "$OTHER_PID" 2>/dev/null &&
     tr '\0' ' ' < "/proc/$OTHER_PID/cmdline" 2>/dev/null | grep -q "run.sh"; then
    MSG="$(date -Is) run.sh is already running (pid $OTHER_PID); nothing to do."
    echo "$MSG" | tee -a "$LOG_FILE" >&2
    exit 0
  fi
  rm -f "$PID_FILE"
fi

# A kiosk browser needs a graphical session.
if [ -z "${WAYLAND_DISPLAY-}" ] && [ -z "${DISPLAY-}" ]; then
  {
    echo "$(date -Is) ERROR: no graphical display (WAYLAND_DISPLAY and DISPLAY are both unset)."
    echo "$(date -Is)        Start the demo from the machine's own desktop session, not over SSH."
  } | tee -a "$LOG_FILE" >&2
  exit 1
fi

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

# Wait for every service the dashboard talks to, not just the frontend. The two
# VLM servers each load a vision and a text model onto the NPUs and need a
# minute or two; a browser opened before they answer shows nothing but
# "Connecting to backend...". A service that never answers is logged and the
# browser opens anyway, so the operator sees a partly working dashboard
# instead of a blank screen.
wait_for() {
  local name="$1" url="$2" tries="$3"
  for _ in $(seq 1 "$tries"); do
    if curl -sf --max-time 2 -o /dev/null "$url"; then
      echo "$(date -Is) $name is up"
      return 0
    fi
    sleep 1
  done
  echo "$(date -Is) WARNING: $name did not answer within ${tries}s ($url)" >&2
  return 1
}

wait_for "frontend"       "$FRONTEND_URL"                                             180 || true
wait_for "vision backend" "http://localhost:8081/healthz"                             180 || true
wait_for "VLM (weapon)"   "http://localhost:5000/socket.io/?EIO=4&transport=polling"  300 || true
wait_for "VLM (fall)"     "http://localhost:5001/socket.io/?EIO=4&transport=polling"  300 || true

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
# short cooldown. The intended shutdown path is stop.sh (via the Clear Demo
# desktop shortcut) or reboot, so a healthy browser is always restarted.
#
# A browser that dies within seconds of starting cannot be fixed by retrying -
# it is a display, profile or packaging problem. Back off and then give up, so
# the failure is visible in the log instead of buried under thousands of
# launch attempts.
FAST_EXIT_LIMIT=5
fast_exits=0
while true; do
  echo "===== $(date -Is) launching $BROWSER ====="
  started_at=$SECONDS
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
  ran_for=$(( SECONDS - started_at ))
  if [ "$ran_for" -lt 5 ]; then
    fast_exits=$(( fast_exits + 1 ))
    if [ "$fast_exits" -ge "$FAST_EXIT_LIMIT" ]; then
      echo "$(date -Is) ERROR: $BROWSER exited immediately $fast_exits times in a row; giving up." >&2
      echo "$(date -Is)        Read the browser errors above - a missing display or an" >&2
      echo "$(date -Is)        inaccessible profile directory are the usual causes." >&2
      exit 1
    fi
    backoff=$(( fast_exits * 5 ))
    echo "===== $(date -Is) browser exited after ${ran_for}s (attempt $fast_exits/$FAST_EXIT_LIMIT), retrying in ${backoff}s ====="
    sleep "$backoff"
  else
    fast_exits=0
    echo "===== $(date -Is) browser exited after ${ran_for}s, respawning ====="
    sleep 2
  fi
done

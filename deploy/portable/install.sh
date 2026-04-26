#!/usr/bin/env bash
# install.sh -- one-shot dependency install for the portable web decoder.
#
# Run once on the Pi while it has internet access (before bringing up
# the AP, or temporarily uplinking via Ethernet). After this finishes
# you can run setup_hotspot.sh to put the Pi on its own AP.
#
# Steps:
#   1. apt install runtime dependencies (chrony, sound libs, curl)
#   2. pip install Python deps into the user environment
#   3. download Socket.IO client JS so the SPA works without internet
#   4. add the current user to the gpio + audio groups
#
# Usage:
#   ./install.sh             # interactive (asks for sudo when needed)
#   ./install.sh --no-apt    # skip apt step
#   ./install.sh --no-pip    # skip pip step

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOCKETIO_VERSION="4.7.5"
SOCKETIO_URL="https://cdn.socket.io/${SOCKETIO_VERSION}/socket.io.min.js"
SOCKETIO_DEST="$SCRIPT_DIR/static/socket.io.min.js"

APT_PKGS=(
  chrony
  python3-pip
  python3-venv
  libportaudio2
  libsndfile1
  curl
  iw
  rfkill
)

DO_APT=1
DO_PIP=1
for arg in "$@"; do
  case "$arg" in
    --no-apt) DO_APT=0 ;;
    --no-pip) DO_PIP=0 ;;
    -h|--help)
      sed -n '1,/^set -euo/p' "$0" | sed -n '2,$p' | sed 's/^# *//; /^set -/d'
      exit 0
      ;;
  esac
done

# ---- 1. apt ----
if [[ $DO_APT -eq 1 ]]; then
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "==> apt install (will sudo)"
    sudo apt update
    sudo apt install -y "${APT_PKGS[@]}"
  else
    apt update
    apt install -y "${APT_PKGS[@]}"
  fi
else
  echo "==> skipping apt"
fi

# ---- 2. pip ----
if [[ $DO_PIP -eq 1 ]]; then
  echo "==> pip install"
  REQ="$SCRIPT_DIR/requirements.txt"
  if [[ ! -f "$REQ" ]]; then
    echo "ERROR: $REQ not found" >&2
    exit 1
  fi
  # On Bookworm pip refuses to install system-wide without --break-system-packages.
  # Prefer --user; fall back to --break-system-packages with a warning.
  if pip3 install --user -r "$REQ"; then
    :
  else
    echo "==> --user failed, retrying with --break-system-packages"
    pip3 install --break-system-packages -r "$REQ"
  fi
else
  echo "==> skipping pip"
fi

# ---- 3. socket.io.min.js ----
echo "==> fetching Socket.IO client JS for offline use"
mkdir -p "$SCRIPT_DIR/static"
if [[ ! -s "$SOCKETIO_DEST" ]]; then
  if curl -fsSL "$SOCKETIO_URL" -o "$SOCKETIO_DEST"; then
    echo "    saved $SOCKETIO_DEST"
  else
    echo "WARNING: could not download $SOCKETIO_URL" >&2
    echo "         The SPA will not function without it. Either run this" >&2
    echo "         script while the Pi has internet, or copy a local" >&2
    echo "         socket.io.min.js into deploy/portable/static/." >&2
  fi
else
  echo "    already present at $SOCKETIO_DEST"
fi

# ---- 4. groups ----
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  TARGET_USER="${SUDO_USER:-$USER}"
else
  TARGET_USER="${SUDO_USER:-$USER}"
fi

for grp in gpio audio; do
  if getent group "$grp" >/dev/null 2>&1; then
    if id -nG "$TARGET_USER" 2>/dev/null | grep -qw "$grp"; then
      echo "==> $TARGET_USER already in $grp"
    else
      echo "==> adding $TARGET_USER to $grp (effective after re-login)"
      sudo usermod -aG "$grp" "$TARGET_USER"
    fi
  fi
done

cat <<EOF

  Install complete.

  Next steps:
    1. (Optional) ./setup_hotspot.sh    # bring up the Pi's AP
    2. python3 deploy/portable/serve.py # start the decoder server

  Notes:
    - chrony is installed but not configured for GPS yet. To pull time
      from a USB GPS dongle, install gpsd and add a refclock line to
      /etc/chrony/chrony.conf -- the time-sync indicator in the web UI
      will start showing 'GPS' once it's working.
    - Audio/GPIO group membership only takes effect after you log out
      and back in (or reboot).

EOF

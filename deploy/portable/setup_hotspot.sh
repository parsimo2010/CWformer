#!/usr/bin/env bash
# setup_hotspot.sh -- one-shot AP configuration for the portable decoder.
#
# Targets Raspberry Pi OS (Bookworm) on a Pi 5, where NetworkManager is
# the default network stack and `nmcli` can stand up a WPA2-protected AP
# in a few lines. After this runs once, the AP autostarts at boot.
#
# Usage:
#   sudo ./setup_hotspot.sh                # use defaults
#   sudo CWFORMER_PASSWORD=mypass ./setup_hotspot.sh
#   sudo CWFORMER_SSID=MyPi CWFORMER_ADDR=192.168.42.1/24 ./setup_hotspot.sh
#
# Defaults:
#   SSID=CWformer-Pi  password=cwformerpi  iface=wlan0  addr=192.168.50.1/24
#
# To stop or remove later:
#   sudo nmcli con down  CWformer-AP
#   sudo nmcli con up    CWformer-AP
#   sudo nmcli con delete CWformer-AP

set -euo pipefail

SSID="${CWFORMER_SSID:-CWformer-Pi}"
PASSWORD="${CWFORMER_PASSWORD:-cwformerpi}"
IFACE="${CWFORMER_IFACE:-wlan0}"
CON_NAME="${CWFORMER_CON_NAME:-CWformer-AP}"
ADDR="${CWFORMER_ADDR:-192.168.50.1/24}"
COUNTRY="${CWFORMER_COUNTRY:-US}"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "ERROR: run with sudo (needs to modify NetworkManager connections)." >&2
  exit 1
fi

if ! command -v nmcli >/dev/null 2>&1; then
  echo "ERROR: nmcli not found. NetworkManager is required." >&2
  echo "       On Pi OS Bookworm it is installed by default; on older" >&2
  echo "       images run: sudo apt install -y network-manager" >&2
  exit 1
fi

if ! nmcli device status | awk '{print $1}' | grep -qx "$IFACE"; then
  echo "ERROR: NetworkManager doesn't manage interface '$IFACE'." >&2
  echo "       Available interfaces:" >&2
  nmcli device status >&2
  exit 1
fi

if [[ ${#PASSWORD} -lt 8 ]]; then
  echo "ERROR: WPA2 password must be at least 8 characters." >&2
  exit 1
fi

# Set Wi-Fi regulatory domain so the AP picks a legal channel.
if command -v iw >/dev/null 2>&1; then
  iw reg set "$COUNTRY" 2>/dev/null || true
fi
if command -v rfkill >/dev/null 2>&1; then
  rfkill unblock wifi 2>/dev/null || true
fi

echo "Configuring AP:"
echo "  SSID:       $SSID"
echo "  Password:   ********"
echo "  Interface:  $IFACE"
echo "  Address:    $ADDR"
echo "  Connection: $CON_NAME"

nmcli con delete "$CON_NAME" >/dev/null 2>&1 || true

nmcli con add type wifi ifname "$IFACE" con-name "$CON_NAME" \
  autoconnect yes ssid "$SSID"

nmcli con modify "$CON_NAME" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  ipv4.method shared \
  ipv4.addresses "$ADDR" \
  ipv6.method disabled

nmcli con modify "$CON_NAME" \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.proto rsn \
  wifi-sec.pairwise ccmp \
  wifi-sec.group ccmp \
  wifi-sec.psk "$PASSWORD"

nmcli con up "$CON_NAME"

cat <<EOF

  AP is up.

  SSID:      $SSID
  Password:  $PASSWORD
  IP:        ${ADDR%/*}

  Connect a phone/tablet to "$SSID" then browse to:
      http://${ADDR%/*}:8080/

  Start the decoder server:
      python3 deploy/portable/serve.py

  Note: this AP shares no internet uplink. The Pi is the only host on
  the network. Devices connecting see only this Pi.

EOF

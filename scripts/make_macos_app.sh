#!/bin/bash
# Build "Prusa PA Tuner.app" -- a small macOS launcher for an existing install.
#
#   scripts/make_macos_app.sh                 # -> ~/Applications/Prusa PA Tuner.app
#   scripts/make_macos_app.sh --desktop       # ...plus a shortcut on the Desktop
#   scripts/make_macos_app.sh --dest /Applications
#
# The app is a thin wrapper, not a standalone bundle: it starts the
# `prusa-pa-tuner` from this checkout's virtualenv and opens the web UI in the
# default browser. Quitting the app stops the server. If you move the checkout
# or recreate the venv somewhere else, just run this script again.
#
# Uses only tools that ship with macOS (osacompile, sips, iconutil, codesign).
set -euo pipefail

APP_NAME="Prusa PA Tuner"
DEST="$HOME/Applications"
DESKTOP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dest) DEST="${2:?--dest needs a directory}"; shift 2 ;;
        --desktop) DESKTOP=1; shift ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This script only works on macOS." >&2
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/packaging/macos"

# Prefer the checkout's own venv; fall back to whatever is on PATH.
if [ -x "$REPO/.venv/bin/prusa-pa-tuner" ]; then
    TUNER_EXE="$REPO/.venv/bin/prusa-pa-tuner"
elif command -v prusa-pa-tuner >/dev/null 2>&1; then
    TUNER_EXE="$(command -v prusa-pa-tuner)"
else
    echo "prusa-pa-tuner not found. Install first (see README: python -m venv .venv && pip install -e .)." >&2
    exit 1
fi

APP="$DEST/$APP_NAME.app"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Bake the local paths into the launcher. Escape for an AppleScript string
# literal (\ and ") and for the sed replacement (\, & and the | delimiter).
esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | sed -e 's/[\\&|]/\\&/g'; }
sed -e "s|@@TUNER_EXE@@|$(esc "$TUNER_EXE")|" \
    -e "s|@@WORK_DIR@@|$(esc "$REPO")|" \
    "$SRC/launcher.applescript" > "$WORK/launcher.applescript"

# Icon: icon.png (1024 px, rendered from icon.svg) -> .icns
ICONSET="$WORK/app.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
    sips -z "$s" "$s" "$SRC/icon.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z "$d" "$d" "$SRC/icon.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$WORK/applet.icns"

# Quit a running copy before replacing it (this also stops its server).
osascript -e "if application \"$APP_NAME\" is running then tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true

mkdir -p "$DEST"
rm -rf "$APP"
osacompile -s -o "$APP" "$WORK/launcher.applescript"   # -s = stay open (keeps the Dock icon)
cp "$WORK/applet.icns" "$APP/Contents/Resources/applet.icns"
rm -f "$APP/Contents/Resources/Assets.car"             # would override applet.icns
PLIST="$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName $APP_NAME" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string org.prusa-pa-tuner.launcher" "$PLIST" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier org.prusa-pa-tuner.launcher" "$PLIST"
/usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$PLIST" 2>/dev/null || true
codesign --force -s - "$APP" >/dev/null 2>&1           # ad-hoc re-sign after editing the bundle
touch "$APP"

echo "Built: $APP"
echo "  starts: $TUNER_EXE"

if [ "$DESKTOP" = 1 ]; then
    ln -sfn "$APP" "$HOME/Desktop/$APP_NAME"
    echo "Shortcut: $HOME/Desktop/$APP_NAME"
fi

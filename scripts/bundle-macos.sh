#!/usr/bin/env bash
# Build dist/AI Model Viewer.app from the ai-model-viewer-mac binary.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cargo build --release --bin ai-model-viewer-mac --features app

app="$root/dist/AI Model Viewer.app"
rm -rf "$root/dist/Netron RS.app" "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$root/target/release/ai-model-viewer-mac" "$app/Contents/MacOS/ai-model-viewer-mac"
cp "$root/macos/Info.plist" "$app/Contents/Info.plist"
cp "$root/macos/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
chmod +x "$app/Contents/MacOS/ai-model-viewer-mac"
codesign --force --sign - "$app"

echo "built $app"

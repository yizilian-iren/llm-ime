#!/bin/sh
# Builds shell/build/LLMIME.app with the Command Line Tools' swiftc (no Xcode), ad-hoc signed.
#   shell/build.sh            the input method
#   shell/build.sh mock DIR   also the candidate window mock, rendered light / dark into DIR
set -e
cd "$(dirname "$0")"
B=build
APP=$B/LLMIME.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" $B/tools
VERSION="$(git rev-list --count HEAD).$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo .dirty)"
swiftc -O -target arm64-apple-macos13 -framework AppKit -framework InputMethodKit \
  Sources/*.swift -o "$APP/Contents/MacOS/LLMIME"
sed "s/@VERSION@/$VERSION/" Info.plist > "$APP/Contents/Info.plist"
cp -R Resources/*.lproj "$APP/Contents/Resources/"
swiftc -O -framework AppKit tools/icon.swift -o $B/tools/icon
$B/tools/icon "$APP/Contents/Resources/icon.tiff"
codesign --force --sign - "$APP"
echo "built $APP $VERSION"
if [ "$1" = mock ]; then
  swiftc -O -framework AppKit -framework CoreImage Sources/Theme.swift Sources/CandidateView.swift tools/mock/main.swift -o $B/tools/mock
  $B/tools/mock "${2:-$B/mock}" "$APP/Contents/Resources/icon.tiff"
fi

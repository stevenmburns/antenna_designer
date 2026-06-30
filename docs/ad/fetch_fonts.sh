#!/usr/bin/env bash
# Fetch the IBM Plex fonts the banner generators use (same family the app loads).
# IBM Plex is licensed OFL 1.1 (github.com/IBM/plex); these files are NOT
# committed to keep the repo lean — run this once before regenerating the ads.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts && cd fonts
BASE="https://cdn.jsdelivr.net/gh/IBM/plex@v6.4.0"
for w in Regular SemiBold Bold; do
  curl -fsSL "$BASE/IBM-Plex-Sans/fonts/complete/ttf/IBMPlexSans-$w.ttf" -o "IBMPlexSans-$w.ttf"
done
for w in Medium SemiBold; do
  curl -fsSL "$BASE/IBM-Plex-Mono/fonts/complete/ttf/IBMPlexMono-$w.ttf" -o "IBMPlexMono-$w.ttf"
done
echo "Fetched IBM Plex TTFs into $(pwd)"

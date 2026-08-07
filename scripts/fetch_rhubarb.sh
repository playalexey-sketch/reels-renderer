#!/usr/bin/env bash
# Rhubarb Lip Sync (open source) из npm-реестра (разрешённый эгресс)
set -e
mkdir -p "${THIRDPARTY:-$HOME/third_party}/rhubarb"
cd /tmp
TARBALL=$(curl -s https://registry.npmjs.org/rhubarb-lip-sync | python3 -c "import json,sys; d=json.load(sys.stdin); v=d['dist-tags']['latest']; print(d['versions'][v]['dist']['tarball'])")
curl -s "$TARBALL" -o rh.tgz
tar -xzf rh.tgz -C "${THIRDPARTY:-$HOME/third_party}/rhubarb" --strip-components=1
"${THIRDPARTY:-$HOME/third_party}/rhubarb/.tools/rhubarb-Lip-Sync-1.13.0-Linux/rhubarb" --version

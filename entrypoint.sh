#!/usr/bin/env bash
set -e

# Ensure PORT env exists
: "${PORT:=5000}"

# Choose which app module to run:
if [ -f /app/server/app.py ]; then
  APP_MODULE="server.app:app"
elif [ -f /app/app.py ]; then
  APP_MODULE="app:app"
else
  echo "Error: Could not find app.py in /app or /app/server"
  ls -al /app
  exit 1
fi

# Run with gunicorn
exec gunicorn -b "0.0.0.0:${PORT}" "$APP_MODULE" --workers 1 --threads 4

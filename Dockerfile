# Dockerfile - robust copy of repo and auto-detect app location
FROM python:3.11-slim

# Install ffmpeg and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire repo into image
COPY . /app

# Install Python deps from either /app/server/requirements.txt or /app/requirements.txt
RUN if [ -f /app/server/requirements.txt ] ; then \
      pip install --no-cache-dir -r /app/server/requirements.txt ; \
    elif [ -f /app/requirements.txt ] ; then \
      pip install --no-cache-dir -r /app/requirements.txt ; \
    else \
      echo "No requirements.txt found - continuing" ; \
    fi

# Make entrypoint executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000
ENV PORT=5000

# Entrypoint will pick correct app module (server.app or app)
ENTRYPOINT ["/app/entrypoint.sh"]

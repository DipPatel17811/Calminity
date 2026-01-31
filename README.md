# CalmiAI minimal server (Gemini + SpeechRecognition + gTTS)

This server:
- Accepts WAV audio (POST /api/voice as raw bytes)
- Transcribes with SpeechRecognition (Google Web Speech API) — English-only
- Generates text reply using Google Gemini (requires GEMINI_API_KEY)
- Synthesizes audio via gTTS (MP3) and converts to WAV (16kHz PCM16) via pydub (ffmpeg)
- Returns JSON: `{ "text": "<AI reply>", "audio_base64": "<base64 WAV bytes>" }`

Requirements
- Docker (recommended) or a Linux host with python3 and ffmpeg
- Environment variable: `GEMINI_API_KEY` (your Gemini API key)

Deploy with Docker (recommended)
1. Build locally:
   docker build -t calmiai-server .

2. Run locally for quick test:
   docker run --rm -p 5000:5000 -e GEMINI_API_KEY=your_key calmiai-server

3. Visit `http://localhost:5000/_ffmpeg_test` to confirm ffmpeg is present.

Render deployment (Docker)
1. Push this repo to your Git remote.
2. Create a new Render Web Service and select "Docker".
3. Connect your repository and branch. Render will use the Dockerfile in repo root.
4. Add environment variable `GEMINI_API_KEY` in Render service settings.
5. Deploy.

Render deployment (non-Docker) - NOT RECOMMENDED
- You can try using Render's build command to apt-get install ffmpeg during build, but Docker is more reliable.

API usage
POST /api/voice
- Headers: Content-Type: audio/wav
- Body: raw WAV bytes (recommend LINEAR16 @ 16000 Hz mono)
- Response: JSON {"text": "...", "audio_base64": "..."}

Notes & caveats
- SpeechRecognition's `recognize_google` uses Google's free web endpoint and may be rate-limited. For production reliability, replace STT with a paid or on-prem STT (e.g. Google Cloud STT, Whisper, Vosk).
- gTTS uses Google Translate TTS endpoint (works well for English for prototyping).
- ffmpeg must be present for pydub audio conversion. Dockerfile installs ffmpeg.

Security
- Keep GEMINI_API_KEY secret (Render environment variables).
- Consider adding authentication between device and server (API key, token).
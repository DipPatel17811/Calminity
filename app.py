#!/usr/bin/env python3
"""
CalmiAI Flask server (English-only)

Features:
- /api/voice  : Accepts raw WAV bytes -> STT -> Gemini -> TTS -> returns {"text","audio_base64"}
- /api/vitals : Accepts JSON vitals from device and stores to SQLite
- /api/vitals/latest : Returns latest stored vitals (convenience)
- /_ffmpeg_test: Verify ffmpeg is available in runtime

Notes:
- Requires GEMINI_API_KEY env var
- Default Gemini model: models/gemini-1.5-flash (override with GEMINI_MODEL)
- ffmpeg is required for MP3->WAV conversion (pydub)
"""

import os
import time
import sqlite3
import base64
import tempfile
import traceback
import subprocess
from typing import Optional, Tuple

from flask import Flask, request, jsonify
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

# ----------------------------
# Configuration
# ----------------------------
DB_PATH = os.environ.get("CALMI_DB_PATH", "calmi.db")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-1.5-flash")
# optional: max tokens cap
GEMINI_MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "400"))

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)
# Instantiate model object (SDK pattern)
try:
    model = genai.GenerativeModel(GEMINI_MODEL)
except Exception:
    # allow server to start but fail gracefully later
    model = None

# ----------------------------
# Database helpers (SQLite)
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT DEFAULT '',
            heart_rate INTEGER,
            spo2 INTEGER,
            stress REAL,
            ts INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def insert_vitals(heart_rate: int, spo2: int, stress: float, device_id: str = "") -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vitals (device_id, heart_rate, spo2, stress, ts) VALUES (?, ?, ?, ?, ?)",
        (device_id, heart_rate, spo2, stress, int(time.time())),
    )
    conn.commit()
    conn.close()


def fetch_latest_vitals() -> Optional[Tuple[int, int, float, int, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT heart_rate, spo2, stress, ts, device_id FROM vitals ORDER BY ts DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    return row  # (heart_rate, spo2, stress, ts, device_id) or None


# Initialize DB at startup
init_db()

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)


def safe_extract_gemini_text(response) -> str:
    """
    Robustly extract text from a Gemini SDK response object.
    The SDK shape can vary across versions; try a few places.
    """
    try:
        if not response:
            return ""
        # Common convenient attribute
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        # Some SDKs return candidates or output items
        if hasattr(response, "candidates") and response.candidates:
            first = response.candidates[0]
            if hasattr(first, "content"):
                return getattr(first, "content").strip()
            if hasattr(first, "text"):
                return getattr(first, "text").strip()
        # Some return 'output' with 'content' keys
        if hasattr(response, "output") and response.output:
            out0 = response.output[0]
            if isinstance(out0, dict):
                # might be {"content": "..." }
                content = out0.get("content") or out0.get("text")
                if content:
                    return content.strip()
        # Fallback to str()
        return str(response).strip()
    except Exception:
        return ""


@app.route("/api/vitals", methods=["POST"])
def receive_vitals():
    """
    Accept JSON payload like:
    {
      "device_id": "esp32-123",
      "heart_rate": 78,
      "spo2": 97,
      "stress_index": 0.63
    }
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        heart_rate = int(data.get("heart_rate", 0))
        spo2 = int(data.get("spo2", 0))
        stress = float(data.get("stress_index", data.get("stress", 0.0)))
        device_id = str(data.get("device_id", ""))

        if heart_rate <= 0 or spo2 <= 0:
            return jsonify({"error": "Invalid vitals values"}), 400

        insert_vitals(heart_rate, spo2, stress, device_id)
        app.logger.info(f"Inserted vitals: hr={heart_rate} spo2={spo2} stress={stress} device={device_id}")
        return jsonify({"status": "ok"}), 200

    except Exception as e:
        app.logger.exception("Error in /api/vitals")
        return jsonify({"error": str(e)}), 500


@app.route("/api/vitals/latest", methods=["GET"])
def latest_vitals():
    try:
        row = fetch_latest_vitals()
        if not row:
            return jsonify({"vitals": None}), 200
        hr, spo2, stress, ts, device_id = row
        return jsonify({
            "heart_rate": hr,
            "spo2": spo2,
            "stress_index": stress,
            "timestamp": ts,
            "device_id": device_id
        }), 200
    except Exception:
        app.logger.exception("Error in /api/vitals/latest")
        return jsonify({"error": "Failed to fetch latest vitals"}), 500


@app.route("/api/voice", methods=["POST"])
def voice():
    """
    Accepts raw WAV bytes in request body (Content-Type: audio/wav or application/octet-stream)
    Returns:
      {
        "text": "<AI reply>",
        "audio_base64": "<base64 WAV (LINEAR16 16kHz)>"
      }
    """
    in_wav_path = None
    mp3_path = None
    wav_path = None
    try:
        # allow octet-stream or absent content-type
        audio_bytes = request.data or request.get_data()
        if not audio_bytes:
            return jsonify({"error": "No audio provided"}), 400

        # Save incoming audio to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_tmp:
            in_tmp.write(audio_bytes)
            in_tmp.flush()
            in_wav_path = in_tmp.name

        # STT using SpeechRecognition (Google Web Speech API)
        recognizer = sr.Recognizer()
        transcript = ""
        try:
            with sr.AudioFile(in_wav_path) as source:
                audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language="en-US")
        except sr.UnknownValueError:
            transcript = ""
        except sr.RequestError as e:
            app.logger.exception("STT request error")
            return jsonify({"error": f"STT request failed: {str(e)}"}), 502
        except Exception:
            app.logger.exception("STT processing error")
            return jsonify({"error": "STT processing error"}), 500

        # Get latest vitals and format context
        vitals_row = fetch_latest_vitals()
        vitals_context = ""
        if vitals_row:
            hr, spo2, stress, ts, device_id = vitals_row
            vitals_context = (
                f"\nPhysiological data (latest):\n"
                f"- Heart rate: {hr} bpm\n"
                f"- SpO2: {spo2}%\n"
                f"- Stress index (0.0–1.0): {stress}\n"
                f"- Device: {device_id}\n"
            )

        # Build prompt with a medical-safety disclaimer
        prompt = (
            "You are CalmiAI, a calm, empathetic assistant for general wellness.\n"
            f"The user said: \"{transcript}\"\n"
            f"{vitals_context}\n"
            "Do NOT provide medical diagnoses or emergency instructions. "
            "Provide concise, non-judgmental wellness advice and practical grounding exercises "
            "appropriate for teens. If vitals indicate a safety risk (e.g. very low SpO2 or dangerously high heart rate), advise seeking immediate medical attention.\n"
            "Respond in plain English, concisely."
        )

        # Generate reply with Gemini
        ai_text = ""
        try:
            if model is None:
                raise RuntimeError("Gemini model is not initialized on the server")

            # Use model.generate_content (SDK wrapper)
            gen_response = model.generate_content(prompt)
            ai_text = safe_extract_gemini_text(gen_response)
            if not ai_text:
                ai_text = "Sorry, I couldn't generate a response right now."
        except Exception:
            app.logger.exception("Gemini generation error")
            ai_text = "Sorry, I couldn't generate a response at this time."

        # TTS with gTTS -> MP3 -> WAV (16kHz PCM16)
        wav_out_b64 = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
                mp3_path = mp3_tmp.name

            tts = gTTS(text=ai_text, lang="en")
            tts.save(mp3_path)

            audio_seg = AudioSegment.from_file(mp3_path, format="mp3")
            audio_seg = audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_path = wav_tmp.name

            audio_seg.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            wav_out_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        except Exception:
            app.logger.exception("TTS / audio conversion error")
            wav_out_b64 = ""  # fallback to text-only

        return jsonify({"text": ai_text, "audio_base64": wav_out_b64}), 200

    finally:
        # Always try to cleanup temp files
        for p in (in_wav_path, mp3_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


@app.route("/_ffmpeg_test", methods=["GET"])
def ffmpeg_test():
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT, timeout=5)
        first_line = out.decode("utf-8").splitlines()[0]
        return jsonify({"ffmpeg": first_line})
    except Exception as e:
        app.logger.exception("ffmpeg test failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use environment PORT if provided (Render sets PORT)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

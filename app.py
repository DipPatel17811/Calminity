#!/usr/bin/env python3
"""
CalmiAI Flask server (English-only)

Pipeline:
- Input: raw WAV bytes (LINEAR16 PCM, 16 kHz mono recommended)
- STT: SpeechRecognition (Google Web Speech API)
- LLM: Google Gemini
- TTS: gTTS → MP3 → WAV (16 kHz PCM16)
- Output: JSON { text, audio_base64 }
"""

import os
import base64
import tempfile
import traceback
import subprocess

from flask import Flask, request, jsonify
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

# ---------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "models/gemini-2.5-flash"
)

model = genai.GenerativeModel(GEMINI_MODEL)

# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------
app = Flask(__name__)

@app.route("/api/voice", methods=["POST"])
def voice():
    try:
        # ESP32 safety: handle octet-stream or missing content-type
        audio_bytes = request.data or request.get_data()
        if not audio_bytes:
            return jsonify({"error": "No audio provided"}), 400

        # Save input WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            in_wav_path = f.name

        # -------------------------------------------------------------
        # 1) Speech-to-Text
        # -------------------------------------------------------------
        recognizer = sr.Recognizer()
        transcript = ""

        try:
            with sr.AudioFile(in_wav_path) as source:
                audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(
                audio_data,
                language="en-US"
            )
        except sr.UnknownValueError:
            transcript = ""
        except sr.RequestError as e:
            return jsonify({"error": f"STT failed: {str(e)}"}), 502
        except Exception:
            traceback.print_exc()
            return jsonify({"error": "STT processing error"}), 500

        # -------------------------------------------------------------
        # 2) Gemini response
        # -------------------------------------------------------------
        prompt = (
            "You are CalmiAI, a calm, empathetic AI assistant.\n"
            f'The user said: "{transcript}".\n'
            "Reply concisely in English with practical, supportive advice."
        )

        try:
            response = model.generate_content(prompt)
            ai_text = response.text.strip() if response and response.text else \
                "I'm here with you. How can I help further?"
        except Exception:
            traceback.print_exc()
            ai_text = "Sorry, I couldn't generate a response right now."

        # -------------------------------------------------------------
        # 3) Text-to-Speech (gTTS → WAV)
        # -------------------------------------------------------------
        wav_out_b64 = ""

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
                mp3_path = mp3_f.name

            gTTS(text=ai_text, lang="en").save(mp3_path)

            audio = AudioSegment.from_file(mp3_path, format="mp3")
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
                wav_path = wav_f.name

            audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

            with open(wav_path, "rb") as f:
                wav_out_b64 = base64.b64encode(f.read()).decode("utf-8")

            os.unlink(mp3_path)
            os.unlink(wav_path)

        except Exception:
            traceback.print_exc()
            wav_out_b64 = ""

        # Cleanup input
        try:
            os.unlink(in_wav_path)
        except:
            pass

        return jsonify({
            "text": ai_text,
            "audio_base64": wav_out_b64
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/_ffmpeg_test", methods=["GET"])
def ffmpeg_test():
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-version"],
            stderr=subprocess.STDOUT,
            timeout=5
        )
        return jsonify({"ffmpeg": out.decode().splitlines()[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

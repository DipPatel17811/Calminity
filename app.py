#!/usr/bin/env python3
"""
CalmiAI minimal Flask server (English-only)

Pipeline:
- Accepts: raw WAV in request body (Content-Type: audio/wav), LINEAR16 PCM 16 kHz mono recommended.
- STT: SpeechRecognition recognize_google (English).
- LLM: Gemini via google.generativeai (requires GEMINI_API_KEY env var).
- TTS: gTTS (generates MP3), convert to WAV (LINEAR16 16kHz) via pydub/ffmpeg.
- Returns JSON: {"text": "<AI reply>", "audio_base64": "<base64 WAV bytes>"}
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

# Required env var
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

# Configure google.generativeai
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gpt-4o-mini")

app = Flask(__name__)

@app.route("/api/voice", methods=["POST"])
def voice():
    """
    Expects: raw audio bytes (Content-Type: audio/wav)
    Returns: JSON { "text": "<AI reply>", "audio_base64": "<BASE64 WAV (LINEAR16 16kHz)>" }
    """
    try:
        audio_bytes = request.data
        if not audio_bytes:
            return jsonify({"error": "No audio provided"}), 400

        # Save incoming audio to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_tmp:
            in_tmp.write(audio_bytes)
            in_tmp.flush()
            in_wav_path = in_tmp.name

        # 1) STT using SpeechRecognition (Google Web Speech API) - English
        recognizer = sr.Recognizer()
        transcript = ""
        try:
            with sr.AudioFile(in_wav_path) as source:
                audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language="en-US")
        except sr.UnknownValueError:
            transcript = ""
        except sr.RequestError as e:
            return jsonify({"error": f"STT request failed: {str(e)}"}), 502
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"STT processing error: {str(e)}"}), 500

        # 2) Generate reply with Gemini (English)
        prompt = (
            f"You are CalmiAI, an empathetic and helpful assistant. The user said: \"{transcript}\".\n"
            "Reply concisely in English with practical suggestions and a calm tone."
        )
        try:
            gen_resp = genai.generate_text(model=GEMINI_MODEL, prompt=prompt, max_output_tokens=400)
            ai_text = gen_resp.text if gen_resp and hasattr(gen_resp, "text") else str(gen_resp)
            if not ai_text:
                ai_text = "Sorry, I couldn't generate a response right now."
        except Exception as e:
            traceback.print_exc()
            ai_text = "Sorry, I couldn't generate a response at this time."

        # 3) TTS with gTTS -> MP3, then convert to WAV (16kHz, 16-bit PCM)
        wav_out_b64 = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
                mp3_path = mp3_tmp.name
            tts = gTTS(text=ai_text, lang="en")
            tts.save(mp3_path)

            # Convert MP3 -> WAV 16kHz mono PCM16 using pydub (ffmpeg required)
            audio_seg = AudioSegment.from_file(mp3_path, format="mp3")
            audio_seg = audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_path = wav_tmp.name
            audio_seg.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            wav_out_b64 = base64.b64encode(wav_bytes).decode("utf-8")

            # cleanup
            try:
                os.unlink(mp3_path)
            except:
                pass
            try:
                os.unlink(wav_path)
            except:
                pass

        except Exception as e:
            traceback.print_exc()
            wav_out_b64 = ""  # fallback to text-only

        # cleanup input file
        try:
            os.unlink(in_wav_path)
        except:
            pass

        return jsonify({"text": ai_text, "audio_base64": wav_out_b64})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/_ffmpeg_test", methods=["GET"])
def ffmpeg_test():
    """
    Simple endpoint to verify ffmpeg exists in runtime image.
    """
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT, timeout=5)
        first_line = out.decode("utf-8").splitlines()[0]
        return jsonify({"ffmpeg": first_line})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
"""
speech_to_text.py

Transcribes an uploaded audio clip (someone saying a drug name) into
text, using Google's free Web Speech API via the SpeechRecognition
library -- a pretrained service, no training/model needed on your end.

Accepts WAV, AIFF, or FLAC audio (SpeechRecognition reads these
natively, no extra audio-conversion tools needed). If your partner's
frontend records audio in a different format (e.g. browser-recorded
webm/mp3), it will need to convert to WAV before sending -- worth
agreeing with your partner up front, same as the SMILES/image contract.
"""

import io

import speech_recognition as sr

_recognizer = sr.Recognizer()


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Transcribe spoken audio (WAV/AIFF/FLAC bytes) into text.
    Returns None if the audio couldn't be understood or the
    recognition service is unreachable, rather than raising --
    callers should treat None as "ask the user to try again."
    """
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = _recognizer.record(source)
    except Exception as e:
        print(f"[speech_to_text] Could not read audio file: {e}")
        return None

    try:
        return _recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("[speech_to_text] Could not understand the audio (unclear speech)")
        return None
    except sr.RequestError as e:
        print(f"[speech_to_text] Speech recognition service error: {e}")
        return None
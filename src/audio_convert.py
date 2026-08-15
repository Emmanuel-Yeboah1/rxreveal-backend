"""
audio_convert.py

Converts browser-recorded audio (typically webm/opus from live
microphone capture) into WAV, which is what speech_to_text.py
actually needs.

Uses PyAV, which bundles the ffmpeg decoding libraries INSIDE the
Python package itself as compiled code -- not as a separate .exe
that gets called via subprocess. This avoids the classic ffmpeg/
ffprobe "not found" errors entirely, since there's no external
binary or PATH configuration involved at all.
"""

import io

import av
import numpy as np
from scipy.io import wavfile


def convert_to_wav(audio_bytes: bytes, source_format: str = "webm") -> bytes:
    """
    Convert audio bytes (e.g. webm from a browser's live microphone
    recording) into WAV bytes.
    """
    input_buffer = io.BytesIO(audio_bytes)
    container = av.open(input_buffer, format=source_format if source_format != "webm" else None)

    audio_stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    samples = []
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for resampled_frame in resampled_frames:
            samples.append(resampled_frame.to_ndarray())

    if not samples:
        raise ValueError("No audio samples decoded from input")

    audio_array = np.concatenate(samples, axis=1).flatten()

    output_buffer = io.BytesIO()
    wavfile.write(output_buffer, 16000, audio_array)
    return output_buffer.getvalue()
import logging
import os
from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
DEFAULT_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "3"))
DEFAULT_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "3"))
DEFAULT_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
DEFAULT_SILENCE_THRESHOLD = float(
    os.getenv("WHISPER_SILENCE_THRESHOLD", "1e-6")
)
DEFAULT_VAD_FILTER = os.getenv("WHISPER_VAD", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
DEFAULT_VAD_MIN_SILENCE_MS = int(
    os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "350")
)

logger = logging.getLogger(__name__)


class ASR:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio = decode_audio(audio_path, sampling_rate=16000)
        except Exception:
            logger.exception("Failed to decode audio: %s", audio_path)
            raise

        if audio is None or len(audio) == 0:
            return ""

        max_amplitude = float(np.max(np.abs(audio)))
        if max_amplitude < DEFAULT_SILENCE_THRESHOLD:
            return ""

        text = self._transcribe_with_vad(
            audio_path,
            vad_filter=DEFAULT_VAD_FILTER,
            min_silence_ms=DEFAULT_VAD_MIN_SILENCE_MS,
        )
        if not text and DEFAULT_VAD_FILTER:
            text = self._transcribe_with_vad(
                audio_path,
                vad_filter=False,
                min_silence_ms=DEFAULT_VAD_MIN_SILENCE_MS,
            )

        return text

    def _transcribe_with_vad(
        self,
        audio_path: str,
        vad_filter: bool,
        min_silence_ms: int,
    ) -> str:
        language = DEFAULT_LANGUAGE.strip().lower()
        language = None if language in ("", "auto", "none") else DEFAULT_LANGUAGE
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=DEFAULT_BEAM_SIZE,
                best_of=DEFAULT_BEST_OF,
                temperature=DEFAULT_TEMPERATURE,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": min_silence_ms},
            )
        except Exception:
            logger.exception("Transcription failed for %s", audio_path)
            raise

        text_parts = []
        for segment in segments:
            if segment.text:
                text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()


@lru_cache(maxsize=1)
def _get_asr() -> ASR:
    return ASR()


def transcribe_audio(audio_path: str) -> str:
    return _get_asr().transcribe(audio_path)

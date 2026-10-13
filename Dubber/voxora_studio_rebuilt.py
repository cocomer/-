"""
Voxora Studio — a fresh, single-file Windows dubbing editor.

Required:
    python -m pip install PySide6 numpy soundfile
Optional microphone recording:
    python -m pip install sounddevice
Optional subtitle translation and neural speech:
    python -m pip install deep-translator edge-tts
Optional local Russian voice cloning:
    llama-tts.exe and the Qwen3-TTS GGUF files in D:\\Dub Studio\\models\\voxora-qwen3-tts-1.7b-q4
    In the app, select "Qwen3 · локально" and assign each character a WAV/MP3 voice sample.
    Imported samples are copied to the local Voxora voice library and are not uploaded.
    Samples are trimmed locally to 8 seconds when used for synthesis.
Optional expressive local Russian voice cloning:
    CrispASR and the Chatterbox GGUF files in D:\\Dub Studio\\models\\chatterbox
    In the app, select "Chatterbox · локально" and assign each character a WAV/MP3 sample.
    Its expression control is a per-line intensity scalar, not a named emotion selector.
Optional local dialogue/music/effects separation (large PyTorch dependency and model):
    python -m pip install demucs
Optional AI text assistance:
    Local Ollama, or a DeepSeek API key (saved using Windows DPAPI encryption).
FFmpeg is needed to import compressed audio and export MP4.
"""
from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, QSize, QSettings, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QAction, QColor, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QProgressBar, QSlider, QSplitter,
    QSpinBox, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
    QScrollArea, QInputDialog,
)

APP_TITLE = "Voxora Studio"
APP_VERSION = 1
AUDIO_RATE = 44100
TRACKS = 6
TRACK_HEIGHT = 48
TRACK_COLORS = (
    "#4b82c3", "#8863ad", "#3d9871", "#c48b3c", "#bd5a69", "#438d94",
)
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv",
}
AUDIO_FILTER = "Audio (*.wav *.mp3 *.m4a *.aac *.ogg *.flac *.wma)"
DEFAULT_VOICES = [
    {"name": "uk-UA-PolinaNeural", "locale": "uk-UA", "gender": "Female"},
    {"name": "uk-UA-OstapNeural", "locale": "uk-UA", "gender": "Male"},
]
QWEN_TTS_DIR = Path(os.environ.get(
    "VOXORA_QWEN_TTS_DIR", r"D:\Dub Studio\models\voxora-qwen3-tts-1.7b-q4"))
QWEN_TTS_MODEL = QWEN_TTS_DIR / "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf"
QWEN_TTS_MMPROJ = QWEN_TTS_DIR / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
VOICE_SAMPLES_DIR = Path(os.environ.get("VOXORA_VOICE_SAMPLES_DIR", r"D:\Dub Studio\voices"))
VOICE_LIBRARY_DIR = Path(os.environ.get(
    "VOXORA_VOICE_LIBRARY_DIR",
    str(Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        / APP_TITLE / "voices"),
))
VOICE_LIBRARY_FILE = VOICE_LIBRARY_DIR.parent / "voice_library.json"
LLAMA_TTS_EXE = Path(os.environ.get(
    "VOXORA_LLAMA_TTS", r"D:\Dub Studio\tools\llama\llama-tts.exe"))
CRISPASR_EXE = Path(os.environ.get(
    "VOXORA_CRISPASR_EXE",
    r"D:\Dub Studio\tools\crispasr-windows-x86_64-cpu\crispasr.exe",
))
CHATTERBOX_MODEL_DIR = Path(os.environ.get(
    "VOXORA_CHATTERBOX_MODEL_DIR", r"D:\Dub Studio\models\chatterbox"))


@dataclass
class AudioClip:
    path: str
    name: str
    start_ms: int = 0
    track: int = 0
    gain: float = 1.0
    muted: bool = False
    trim_start_ms: int = 0
    trim_end_ms: int = 0
    kind: str = "audio"
    cue_index: int = -1
    cue_id: str = ""


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str
    dub_text: str = ""
    character: str = ""
    cue_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    exaggeration: float = 0.5


def run_ffmpeg(*args: str) -> None:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("Не знайдено FFmpeg. Встановіть FFmpeg і додайте його до PATH.")
    result = subprocess.run(
        [executable, "-y", "-hide_banner", "-loglevel", "error", *map(str, args)],
        capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=60 * 60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1600:] or "FFmpeg завершився з помилкою.")


def probe_video_codec(path: str) -> str:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError(
            "Не знайдено ffprobe. Перевірте, що FFmpeg встановлено й додано до PATH.")
    result = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
         path],
        capture_output=True, text=True, errors="replace", timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(
            "Не вдалося визначити відеокодек:\n" +
            (result.stderr.strip()[-1600:] or "ffprobe завершився з помилкою."))
    codec = result.stdout.strip().splitlines()
    if not codec:
        raise RuntimeError("У вибраному файлі не знайдено відеодоріжку.")
    return codec[0].lower()


def prepare_video_playback(path: str, media_folder: str) -> tuple[str, str]:
    source = os.path.abspath(path)
    if not os.path.isfile(source):
        raise RuntimeError(f"Не знайдено відеофайл:\n{source}")
    codec = probe_video_codec(source)
    if codec not in {"hevc", "h265"}:
        return source, source

    source_path = Path(source)
    sibling_proxy = source_path.with_name(source_path.stem + "_h264" + source_path.suffix)
    if sibling_proxy.is_file() and probe_video_codec(str(sibling_proxy)) == "h264":
        return source, str(sibling_proxy.resolve())

    key = hashlib.sha256(os.path.normcase(source).encode("utf-8")).hexdigest()[:12]
    proxy_dir = Path(media_folder) / "playback_proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy = proxy_dir / f"{source_path.stem}_{key}_h264.mp4"
    if proxy.is_file() and probe_video_codec(str(proxy)) == "h264":
        return source, str(proxy)

    temporary = proxy.with_name(proxy.stem + ".partial.mp4")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "Для створення сумісної копії HEVC-відео потрібен FFmpeg у PATH.")
    try:
        result = subprocess.run(
            [executable, "-y", "-hide_banner", "-loglevel", "error",
             "-i", source, "-map", "0:v:0", "-map", "0:a?",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", str(temporary)],
            capture_output=True, text=True, errors="replace", timeout=60 * 60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError(
                "Не вдалося підготувати сумісну копію H.264:\n" +
                (result.stderr.strip()[-1800:] or f"FFmpeg код {result.returncode}."))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg не створив сумісну копію відео.")
        if probe_video_codec(str(temporary)) != "h264":
            raise RuntimeError("Створена копія відео не має кодека H.264.")
        os.replace(temporary, proxy)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return source, str(proxy)


def load_voice_library() -> list[dict]:
    if not VOICE_LIBRARY_FILE.is_file():
        return []
    try:
        with VOICE_LIBRARY_FILE.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Не вдалося прочитати бібліотеку голосів:\n{exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise RuntimeError("Файл бібліотеки голосів має непідтримуваний формат.")
    voices = data.get("voices")
    if not isinstance(voices, list):
        raise RuntimeError("У файлі бібліотеки голосів немає коректного списку голосів.")
    result = []
    for index, voice in enumerate(voices, 1):
        if (
            not isinstance(voice, dict)
            or not isinstance(voice.get("name"), str)
            or not voice["name"].strip()
            or not isinstance(voice.get("path"), str)
            or not voice["path"].strip()
        ):
            raise RuntimeError(f"Некоректний запис голосу №{index} у бібліотеці.")
        result.append({"name": voice["name"], "path": voice["path"]})
    return result


def save_voice_library(voices: list[dict]) -> None:
    VOICE_LIBRARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = VOICE_LIBRARY_FILE.with_name(VOICE_LIBRARY_FILE.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump({"version": 1, "voices": voices}, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, VOICE_LIBRARY_FILE)
    except OSError as exc:
        cleanup_error = ""
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            cleanup_error = f"\nТакож не вдалося прибрати тимчасовий файл: {cleanup_exc}"
        raise RuntimeError(
            f"Не вдалося зберегти бібліотеку голосів:\n{exc}{cleanup_error}") from exc


def local_tts_configuration() -> tuple[str, str, str]:
    executable = str(LLAMA_TTS_EXE)
    if not os.path.isfile(executable):
        executable = shutil.which("llama-tts") or ""
    missing = [
        label for label, path in (
            ("llama-tts.exe", executable),
            ("модель Qwen3-TTS", str(QWEN_TTS_MODEL)),
            ("аудіомодуль Qwen3-TTS", str(QWEN_TTS_MMPROJ)),
        ) if not path or not os.path.isfile(path)
    ]
    if missing:
        raise RuntimeError(
            "Не знайдено локальні компоненти синтезу: " + ", ".join(missing) +
            ". Перевірте шляхи VOXORA_LLAMA_TTS і VOXORA_QWEN_TTS_DIR.")
    return executable, str(QWEN_TTS_MODEL), str(QWEN_TTS_MMPROJ)


def generate_local_speech(text: str, reference: str, output_path: str) -> None:
    if not os.path.isfile(reference):
        raise RuntimeError(f"Не знайдено аудіозразок голосу:\n{reference}")
    executable, model, mmproj = local_tts_configuration()
    reference_wav = os.path.join(tempfile.gettempdir(), f"voxora_reference_{time.time_ns()}.wav")
    try:
        run_ffmpeg("-i", reference, "-t", "8", "-ar", "24000", "-ac", "1",
                   "-c:a", "pcm_s16le", reference_wav)
        result = subprocess.run(
            [executable, "-m", model, "-mm", mmproj, "-p", text, "--tts-lang", "ru",
             "--tts-speaker-file", reference_wav, "--output", output_path,
             "-t", "2", "-ngl", "0"],
            capture_output=True, text=True, errors="replace", timeout=30 * 60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            details = (result.stderr or result.stdout).strip()[-1800:]
            raise RuntimeError(details or f"llama-tts завершився з кодом {result.returncode}.")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("llama-tts не створив WAV-файл.")
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    finally:
        try:
            os.remove(reference_wav)
        except OSError:
            pass


def chatterbox_tts_configuration() -> tuple[str, Path]:
    executable = str(CRISPASR_EXE)
    if not os.path.isfile(executable):
        executable = shutil.which("crispasr") or ""
    if not executable or not os.path.isfile(executable):
        raise RuntimeError(
            "Не знайдено CrispASR. Перевірте VOXORA_CRISPASR_EXE або встановіть "
            "CrispASR у D:\\Dub Studio\\tools\\crispasr-windows-x86_64-cpu.")
    try:
        CHATTERBOX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Не вдалося створити папку локальних моделей Chatterbox:\n"
            f"{CHATTERBOX_MODEL_DIR}\n{exc}") from exc
    return executable, CHATTERBOX_MODEL_DIR


class ChatterboxServer:
    def __init__(self, status_callback):
        self.status_callback = status_callback
        self.process: subprocess.Popen | None = None
        self.port = 0
        self.log_path = ""
        self.voice_dir = ""

    def _log_tail(self) -> str:
        if not self.log_path:
            return ""
        try:
            with open(self.log_path, "rb") as stream:
                stream.seek(max(0, os.path.getsize(self.log_path) - 6000))
                return stream.read().decode("utf-8", errors="replace").strip()
        except OSError as exc:
            return f"Не вдалося прочитати журнал CrispASR: {exc}"

    def start(self) -> None:
        executable, model_dir = chatterbox_tts_configuration()
        model_files = (
            model_dir / "chatterbox-v3-t3-q4_k.gguf",
            model_dir / "chatterbox-v3-s3gen-q4_k.gguf",
        )
        first_download = not all(path.is_file() for path in model_files)
        self.voice_dir = tempfile.mkdtemp(prefix="voxora-chatterbox-voices-")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]

        command = [
            executable, "--backend", "chatterbox", "-m", "auto",
            "--model-quant", "q4_k", "--codec-quant", "q4_k",
            "--cache-dir", str(model_dir), "--language", "ru",
            "--server", "--host", "127.0.0.1", "--port", str(self.port),
            "--voice-dir", self.voice_dir,
            "--no-gpu", "--threads", "4",
        ]
        descriptor, self.log_path = tempfile.mkstemp(
            prefix="voxora-chatterbox-", suffix=".log")
        os.close(descriptor)
        try:
            with open(self.log_path, "ab") as log:
                self.process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=self.voice_dir,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as exc:
            raise RuntimeError(f"Не вдалося запустити CrispASR:\n{exc}") from exc

        if first_download:
            self.status_callback(
                "Перший запуск: локально завантажуються моделі Chatterbox (близько 700 МБ)…")
        else:
            self.status_callback("Завантажую Chatterbox у пам’ять…")
        health_url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + 40 * 60
        next_status = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Локальний CrispASR завершився під час запуску.\n"
                    f"{self._log_tail()}\nЖурнал: {self.log_path}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    status = json.load(response)
                if status.get("status") == "ok":
                    self.status_callback("Локальний Chatterbox готовий.")
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise RuntimeError(
                        f"Локальний сервер CrispASR повернув HTTP {exc.code} під час запуску.\n"
                        f"Журнал: {self.log_path}") from exc
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= next_status:
                self.status_callback(
                    "Chatterbox ще завантажує модель на процесор…")
                next_status = time.monotonic() + 15
            time.sleep(1)
        raise RuntimeError(
            "Не вдалося запустити локальний Chatterbox за 40 хвилин.\n"
            f"{self._log_tail()}\nЖурнал: {self.log_path}")

    def upload_voice(self, reference_path: str) -> str:
        digest = hashlib.sha256()
        try:
            with open(reference_path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise RuntimeError(f"Не вдалося прочитати зразок голосу:\n{exc}") from exc
        voice_name = f"voxora_{digest.hexdigest()[:20]}"
        boundary = "----VoxoraVoice" + uuid.uuid4().hex
        parts = []
        fields = {
            "name": voice_name,
            "consent_attestation": (
                "I confirm I own this voice or have the voice owner's permission."
            ),
        }
        for field_name, value in fields.items():
            parts.extend((
                f"--{boundary}\r\n".encode("ascii"),
                (f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
                 f"{value}\r\n").encode("utf-8"),
            ))
        try:
            with open(reference_path, "rb") as stream:
                audio = stream.read()
        except OSError as exc:
            raise RuntimeError(f"Не вдалося прочитати зразок голосу:\n{exc}") from exc
        parts.extend((
            f"--{boundary}\r\n".encode("ascii"),
            (f'Content-Disposition: form-data; name="voice"; '
             f'filename="{voice_name}.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode("ascii"),
            audio,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ))
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/voices",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[-1600:]
            raise RuntimeError(
                f"Не вдалося підготувати голос у локальному Chatterbox (HTTP {exc.code}): "
                f"{details or exc.reason}\nЖурнал: {self.log_path}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Не вдалося надіслати зразок локальному Chatterbox: {exc.reason}\n"
                f"Журнал: {self.log_path}") from exc
        if result.get("name") != voice_name:
            raise RuntimeError(
                "Chatterbox не підтвердив збереження локального зразка голосу.\n"
                f"Журнал: {self.log_path}")
        source = os.path.join(self.voice_dir, voice_name + ".wav")
        alias = os.path.join(self.voice_dir, voice_name)
        try:
            shutil.copyfile(source, alias)
        except OSError as exc:
            raise RuntimeError(
                f"Не вдалося підготувати локальний зразок для Chatterbox:\n{exc}") from exc
        return voice_name

    def synthesize(
        self, text: str, voice_name: str, exaggeration: float, output_path: str,
    ) -> None:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(
                "Локальний сервер Chatterbox зупинився.\n"
                f"{self._log_tail()}\nЖурнал: {self.log_path}")
        payload = {
            "input": text,
            "voice": voice_name,
            "language": "ru",
            "response_format": "wav",
            "exaggeration": exaggeration,
            "spoken_disclaimer": False,
            "marking_attestation": (
                "I will disclose that this output contains AI-generated voice."
            ),
            "consent_attestation": (
                "I confirm I own this voice or have the voice owner's permission."
            ),
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/audio/speech",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30 * 60) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[-1600:]
            raise RuntimeError(
                f"Локальний Chatterbox повернув HTTP {exc.code}: "
                f"{details or exc.reason}\nЖурнал: {self.log_path}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Не вдалося звернутися до локального Chatterbox: {exc.reason}\n"
                f"Журнал: {self.log_path}") from exc
        if not audio:
            raise RuntimeError(
                "Chatterbox повернув порожній аудіофайл.\n"
                f"Журнал: {self.log_path}")
        try:
            with open(output_path, "wb") as stream:
                stream.write(audio)
            info = sf.info(output_path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise RuntimeError("WAV не містить аудіоданих.")
        except (OSError, RuntimeError, sf.LibsndfileError) as exc:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"Chatterbox повернув некоректний WAV-файл: {exc}\n"
                f"Журнал: {self.log_path}") from exc

    def stop(self, keep_log: bool = False) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if self.log_path and not keep_log:
            try:
                os.remove(self.log_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    f"Не вдалося прибрати тимчасовий журнал CrispASR:\n{exc}") from exc
        if self.voice_dir:
            try:
                shutil.rmtree(self.voice_dir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(
                    f"Не вдалося прибрати тимчасові голоси CrispASR:\n{exc}") from exc
            self.voice_dir = ""


def format_time(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02}:{seconds:02}.{millis // 100}"


def parse_time(value: str) -> int:
    parts = value.strip().replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            return round((float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])) * 1000)
        if len(parts) == 2:
            return round((float(parts[0]) * 60 + float(parts[1])) * 1000)
    except ValueError:
        pass
    raise ValueError(f"Неправильний формат часу: {value!r}. Приклад: 01:23.4")


def subtitle_match_key(text: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFKC", text).casefold()
        if character.isalnum()
    )


def clip_match_key(name: str) -> str:
    normalized_name = unicodedata.normalize("NFKC", name)
    spoken_text = (
        normalized_name.split(": ", 1)[1]
        if ": " in normalized_name else normalized_name
    )
    return subtitle_match_key(spoken_text)


def subtitle_timestamp(hours: str | None, minutes: str, seconds: str, fraction: str) -> int:
    millis = int((fraction + "00")[:3])
    return ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + millis


def parse_subtitle_file(file_path: str) -> list[SubtitleCue]:
    raw = Path(file_path).read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("cp1251", errors="replace")
    content = content.replace("\r", "").strip()
    if Path(file_path).suffix.lower() == ".txt":
        cues = []
        cursor = 0
        for line in content.splitlines():
            line = line.strip()
            if line:
                cues.append(SubtitleCue(cursor, cursor + 3000, line))
                cursor += 3400
        return cues

    time_pattern = re.compile(
        r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
        r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{1,3})"
    )
    cues = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.splitlines()
        for index, line in enumerate(lines):
            match = time_pattern.search(line.strip())
            if not match:
                continue
            fields = match.groups()
            start = subtitle_timestamp(*fields[:4])
            end = subtitle_timestamp(*fields[4:])
            text = re.sub(r"<[^>]*>|\{[^}]*\}", "", " ".join(lines[index + 1:])).strip()
            if text and end > start:
                cues.append(SubtitleCue(start, end, text))
            break
    return cues


def decode_audio(path: str) -> np.ndarray:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sample_rate != AUDIO_RATE and len(mono):
        sample_count = round(len(mono) * AUDIO_RATE / sample_rate)
        old_positions = np.arange(len(mono), dtype=np.float64)
        new_positions = np.arange(sample_count, dtype=np.float64) * sample_rate / AUDIO_RATE
        mono = np.interp(new_positions, old_positions, mono).astype(np.float32)
    return np.asarray(mono, dtype=np.float32)


def separate_dialogue_from_video(video_path: str, media_folder: str) -> str:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Для обробки відео потрібен FFmpeg у PATH.")
    try:
        import demucs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Не встановлено Demucs. Для розділення мовлення встановіть:\n"
            "python -m pip install demucs\n"
            "Під час першого запуску Demucs завантажить AI-модель."
        ) from exc
    os.makedirs(media_folder, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="demucs_", dir=media_folder)
    source_wav = os.path.join(workdir, "source.wav")
    try:
        run_ffmpeg("-i", video_path, "-vn", "-ac", "2", "-ar", str(AUDIO_RATE),
                   "-c:a", "pcm_s16le", source_wav)
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals",
             "-n", "htdemucs", "-o", workdir, source_wav],
            capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=60 * 60,
        )
        if result.returncode:
            raise RuntimeError(
                "Demucs не зміг розділити аудіо:\n" +
                (result.stderr or result.stdout)[-1800:])
        candidates = list(Path(workdir).rglob("no_vocals.wav"))
        if not candidates:
            raise RuntimeError("Demucs завершився, але фонову доріжку no_vocals.wav не знайдено.")
        output = os.path.join(media_folder, "background_without_dialogue.wav")
        staging = output + ".tmp.wav"
        shutil.copy2(candidates[0], staging)
        os.replace(staging, output)
        return output
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class BackgroundJob(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def run(self):
        try:
            self.succeeded.emit(self.callback())
        except Exception as exc:
            self.failed.emit(str(exc))


class AudioRecorder(QThread):
    saved = Signal(str)
    failed = Signal(str)

    def __init__(self, destination: str):
        super().__init__()
        self.destination = destination
        self.stop_requested = False

    def run(self):
        try:
            import sounddevice as sd

            chunks = []

            def callback(indata, frames, timing, status):
                if status:
                    print(f"Microphone: {status}", file=sys.stderr)
                chunks.append(indata.copy())

            with sd.InputStream(
                samplerate=AUDIO_RATE, channels=1, dtype="float32", callback=callback
            ):
                while not self.stop_requested:
                    self.msleep(40)
            if not chunks:
                raise RuntimeError("Мікрофон не записав аудіо.")
            sf.write(self.destination, np.concatenate(chunks, axis=0), AUDIO_RATE)
            self.saved.emit(self.destination)
        except Exception as exc:
            self.failed.emit(str(exc))


class VideoPlaybackJob(QThread):
    succeeded = Signal(str, str)
    failed = Signal(str)

    def __init__(self, video_path: str, media_folder: str):
        super().__init__()
        self.video_path = video_path
        self.media_folder = media_folder

    def run(self):
        try:
            original, playback = prepare_video_playback(self.video_path, self.media_folder)
            self.succeeded.emit(original, playback)
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceCatalogJob(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            import edge_tts
            voices = asyncio.run(edge_tts.list_voices())
            self.succeeded.emit(sorted(
                ({"name": v["ShortName"], "locale": v["Locale"], "gender": v["Gender"]}
                 for v in voices), key=lambda voice: voice["name"]))
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceSampleJob(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, voice: str | dict):
        super().__init__()
        self.voice = voice

    def run(self):
        mp3_path = os.path.join(tempfile.gettempdir(), f"voxora_sample_{time.time_ns()}.mp3")
        wav_path = os.path.splitext(mp3_path)[0] + ".wav"
        try:
            if isinstance(self.voice, dict) and self.voice.get("engine") == "qwen3":
                reference = self.voice.get("reference", "")
                if not os.path.isfile(reference):
                    raise RuntimeError(f"Не знайдено аудіозразок голосу:\n{reference}")
                run_ffmpeg("-i", reference, "-t", "8", "-ar", str(AUDIO_RATE), "-ac", "1",
                           "-c:a", "pcm_s16le", wav_path)
                self.succeeded.emit(wav_path)
                return
            import edge_tts
            voice = self.voice if isinstance(self.voice, str) else self.voice.get("voice", "")
            language = voice.split("-", 1)[0].lower()
            samples = {
                "uk": "Привіт! Це коротка проба голосу.",
                "zh": "你好，这是语音试听。",
                "pl": "Cześć! To krótka próbka głosu.",
                "de": "Hallo! Dies ist eine kurze Stimmprobe.",
                "fr": "Bonjour ! Ceci est un court test vocal.",
                "es": "¡Hola! Esta es una breve prueba de voz.",
            }
            text = samples.get(language, "Hello! This is a short voice preview.")
            asyncio.run(edge_tts.Communicate(text, voice).save(mp3_path))
            run_ffmpeg("-i", mp3_path, "-ar", str(AUDIO_RATE), "-ac", "2",
                       "-c:a", "pcm_s16le", wav_path)
            self.succeeded.emit(wav_path)
        except Exception as exc:
            for path in (mp3_path, wav_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            self.failed.emit(str(exc))


class OllamaJob(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, model: str, cues: list[SubtitleCue]):
        super().__init__()
        self.model = model
        self.cues = cues

    def run(self):
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:11434/api/tags",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=8) as response:
                models = json.load(response).get("models", [])
            if not any(model.get("name") == self.model for model in models):
                raise RuntimeError(f"Модель {self.model} не знайдена в Ollama.")

            assignments = []
            for offset in range(0, len(self.cues), 25):
                batch = self.cues[offset:offset + 25]
                numbered = [{"id": offset + i, "text": cue.text} for i, cue in enumerate(batch)]
                prompt = (
                    "You are labeling dialogue speakers in a subtitle script. "
                    "Use dialogue meaning and context only; do not claim audio recognition. "
                    "Assign a short consistent character name to each line. Use 'Оповідач' "
                    "for narration and 'Невідомий' when uncertain. Return ONLY JSON as "
                    '{"speakers":[{"id":0,"character":"name"}]}. Lines:\n'
                    + json.dumps(numbered, ensure_ascii=False)
                )
                payload = json.dumps({
                    "model": self.model, "stream": False, "format": "json",
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0},
                }).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat", data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as response:
                    result = json.load(response)
                body = json.loads(result["message"]["content"])
                for item in body.get("speakers", []):
                    cue_id = int(item["id"])
                    character = str(item["character"]).strip()
                    if 0 <= cue_id < len(self.cues) and character:
                        assignments.append((cue_id, character))
            if not assignments:
                raise RuntimeError("Ollama не повернув жодного призначення персонажа.")
            self.succeeded.emit(assignments)
        except urllib.error.URLError as exc:
            self.failed.emit(
                "Не вдалося з’єднатися з Ollama на localhost:11434. "
                "Запустіть Ollama і перевірте `ollama list`.\n" + str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class _WindowsDataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def protect_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if os.name != "nt":
        raise RuntimeError("Безпечне збереження API-ключа підтримується лише у Windows.")
    raw = api_key.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = _WindowsDataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    destination = _WindowsDataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    protect = crypt32.CryptProtectData
    protect.argtypes = [
        ctypes.POINTER(_WindowsDataBlob), wintypes.LPCWSTR,
        ctypes.POINTER(_WindowsDataBlob), wintypes.LPVOID, wintypes.LPVOID,
        wintypes.DWORD, ctypes.POINTER(_WindowsDataBlob),
    ]
    protect.restype = wintypes.BOOL
    if not protect(
        ctypes.byref(source), "Voxora DeepSeek API key", None, None, None,
        0x1, ctypes.byref(destination),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        encrypted = ctypes.string_at(destination.pbData, destination.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        local_free(destination.pbData)


def unprotect_api_key(encrypted_key: str) -> str:
    if not encrypted_key:
        return ""
    if os.name != "nt":
        raise RuntimeError("Безпечне читання API-ключа підтримується лише у Windows.")
    try:
        raw = base64.b64decode(encrypted_key, validate=True)
    except ValueError as exc:
        raise RuntimeError("Збережений API-ключ пошкоджений; введіть його знову.") from exc
    buffer = ctypes.create_string_buffer(raw)
    source = _WindowsDataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    destination = _WindowsDataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    unprotect = crypt32.CryptUnprotectData
    unprotect.argtypes = [
        ctypes.POINTER(_WindowsDataBlob), ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_WindowsDataBlob), wintypes.LPVOID, wintypes.LPVOID,
        wintypes.DWORD, ctypes.POINTER(_WindowsDataBlob),
    ]
    unprotect.restype = wintypes.BOOL
    if not unprotect(
        ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(destination),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(destination.pbData, destination.cbData).decode("utf-8")
    finally:
        local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        local_free(destination.pbData)


class AISettingsDialog(QDialog):
    def __init__(self, parent, configuration: dict):
        super().__init__(parent)
        self.setWindowTitle("Налаштування AI-помічника")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.provider = QComboBox()
        self.provider.addItem("Ollama · локально", "ollama")
        self.provider.addItem("DeepSeek API · хмара", "deepseek")
        provider_index = self.provider.findData(configuration.get("provider", "ollama"))
        self.provider.setCurrentIndex(max(0, provider_index))
        self.endpoint = QLineEdit(configuration.get("endpoint", ""))
        self.endpoint.setPlaceholderText("http://127.0.0.1:11434 або https://api.deepseek.com/v1")
        self.model = QLineEdit(configuration.get("model", ""))
        self.api_key = QLineEdit(configuration.get("api_key", ""))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Потрібен лише для DeepSeek API")
        self._previous_provider = self.provider.currentData()
        self.provider.currentIndexChanged.connect(self._provider_changed)
        form.addRow("Провайдер", self.provider)
        form.addRow("Адреса сервера", self.endpoint)
        form.addRow("Модель", self.model)
        form.addRow("API-ключ", self.api_key)
        layout.addLayout(form)
        privacy = QLabel(
            "Ollama працює локально. Для хмарного DeepSeek передаються лише вибрані текстові "
            "репліки — аудіо та відео не надсилаються. API-ключ шифрується засобами Windows "
            "і доступний лише цьому обліковому запису.")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("Зберегти")
        self.cancel_button = QPushButton("Скасувати")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def _provider_changed(self, _index: int):
        defaults = {
            "ollama": ("http://127.0.0.1:11434", "gemma2:2b"),
            "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
        }
        old_provider = getattr(self, "_previous_provider", None)
        new_provider = self.provider.currentData()
        if old_provider is None:
            self._previous_provider = new_provider
            return
        old_endpoint, old_model = defaults[old_provider]
        new_endpoint, new_model = defaults[new_provider]
        if self.endpoint.text().strip().rstrip("/") == old_endpoint:
            self.endpoint.setText(new_endpoint)
        if self.model.text().strip() == old_model:
            self.model.setText(new_model)
        self._previous_provider = new_provider

    def values(self) -> dict:
        return {
            "provider": self.provider.currentData(),
            "endpoint": self.endpoint.text().strip().rstrip("/"),
            "model": self.model.text().strip(),
            "api_key": self.api_key.text().strip(),
        }


class TextAssistReviewDialog(QDialog):
    def __init__(self, parent, items: list[dict], action: str):
        super().__init__(parent)
        self.setWindowTitle("Перевірка пропозицій AI")
        self.resize(1050, 620)
        layout = QVBoxLayout(self)
        explanation = {
            "translate": "Перевірте переклад. Вибрані тексти буде записано в колонку озвучення.",
            "intonation": (
                "Перевірте пунктуацію для природніших пауз та інтонації. Слова мають бути "
                "незмінними; той самий голосовий зразок залишиться."),
            "stress": "Перевірте наголоси. Застосовуються лише позначки наголосу.",
        }
        layout.addWidget(QLabel(explanation[action]))
        self.table = QTableWidget(len(items), 3)
        self.table.setHorizontalHeaderLabels(["Застосувати", "Було", "Пропозиція AI"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setColumnWidth(0, 90)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for row, item in enumerate(items):
            checkbox = QTableWidgetItem()
            checkbox.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable)
            checkbox.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, checkbox)
            original = QTableWidgetItem(item["original"])
            original.setFlags(original.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, original)
            self.table.setItem(row, 2, QTableWidgetItem(item["text"]))
            self.table.setVerticalHeaderItem(row, QTableWidgetItem(str(item["cue_index"] + 1)))
        self._items = items
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        apply_button = QPushButton("Застосувати вибрані")
        cancel_button = QPushButton("Скасувати")
        apply_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(apply_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)

    def selected_results(self) -> list[tuple[int, str]]:
        return [
            (item["cue_index"], self.table.item(row, 2).text().strip())
            for row, item in enumerate(self._items)
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
            and self.table.item(row, 2).text().strip()
        ]


class TextAssistJob(QThread):
    succeeded = Signal(object)
    progress = Signal(int, int)
    failed = Signal(str)

    ACTIONS = {"translate", "intonation", "stress"}

    def __init__(self, configuration: dict, action: str, cues: list[tuple[int, str]]):
        super().__init__()
        self.configuration = dict(configuration)
        self.action = action
        self.cues = list(cues)

    @staticmethod
    def _json_object(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("AI повернув відповідь не у форматі JSON.")
        result = json.loads(content[start:end + 1])
        if not isinstance(result, dict):
            raise RuntimeError("AI повернув некоректний JSON.")
        return result

    @staticmethod
    def _word_tokens(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text.replace("\u0301", "")).casefold()
        return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)

    def _request(self, messages: list[dict]) -> str:
        endpoint = self.configuration["endpoint"].rstrip("/")
        model = self.configuration["model"]
        if self.configuration["provider"] == "ollama":
            url = endpoint + "/api/chat"
            body = {"model": model, "messages": messages, "stream": False, "format": "json",
                    "options": {"temperature": 0}}
            headers = {"Content-Type": "application/json"}
        else:
            url = endpoint + "/chat/completions"
            body = {"model": model, "messages": messages, "temperature": 0,
                    "response_format": {"type": "json_object"}}
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.configuration["api_key"],
            }
        request = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[-1400:]
            raise RuntimeError(
                f"AI-сервер повернув HTTP {exc.code}: {details or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Не вдалося з’єднатися з AI-сервером {endpoint}: {exc.reason}") from exc
        if self.configuration["provider"] == "ollama":
            content = result.get("message", {}).get("content", "")
        else:
            choices = result.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("AI-сервер повернув порожню відповідь.")
        return content

    def run(self):
        try:
            if self.action not in self.ACTIONS:
                raise ValueError("Невідома дія AI-помічника.")
            system_prompts = {
                "translate": (
                    "Translate dialogue into natural, concise, conversational Ukrainian for dubbing. "
                    "Preserve meaning, speaker intent, names, and every important fact; do not add "
                    "content or stage directions. Use punctuation that helps natural spoken pauses. "
                    "Return only JSON {\"items\":[{\"id\":integer,\"text\":string}]} with one result "
                    "for every supplied id."),
                "intonation": (
                    "Prepare dialogue text for natural spoken intonation. Preserve the exact words "
                    "in the same order; only change punctuation and capitalization. Never add or "
                    "remove words, speaker tags, emotion labels, or stage directions. Return only "
                    "JSON {\"items\":[{\"id\":integer,\"text\":string}]} for every supplied id."),
                "stress": (
                    "Add combining acute U+0301 after stressed vowels to mark word stress. Preserve "
                    "all original characters and word order exactly; only insert acute accents. "
                    "Return only JSON {\"items\":[{\"id\":integer,\"text\":string}]} for every id."),
            }
            completed = {}
            batch_size = 12
            for offset in range(0, len(self.cues), batch_size):
                batch = self.cues[offset:offset + batch_size]
                payload = [{"id": cue_id, "text": text} for cue_id, text in batch]
                content = self._request([
                    {"role": "system", "content": system_prompts[self.action]},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ])
                body = self._json_object(content)
                items = body.get("items")
                if not isinstance(items, list):
                    raise RuntimeError("AI не повернув список пропозицій.")
                expected_ids = {cue_id for cue_id, _ in batch}
                for item in items:
                    if not isinstance(item, dict):
                        raise RuntimeError("AI повернув некоректний елемент відповіді.")
                    cue_id = item.get("id")
                    suggested = item.get("text")
                    if (not isinstance(cue_id, int) or cue_id not in expected_ids
                            or cue_id in completed or not isinstance(suggested, str)
                            or not suggested.strip()):
                        raise RuntimeError("AI повернув пропущений або некоректний номер репліки.")
                    source = next(text for index, text in batch if index == cue_id).strip()
                    suggested = suggested.strip()
                    if self.action == "stress":
                        if suggested.replace("\u0301", "") != source.replace("\u0301", ""):
                            raise RuntimeError(
                                f"AI змінив слова у репліці {cue_id + 1}; правку не застосовано.")
                    elif (self.action == "intonation"
                          and self._word_tokens(suggested) != self._word_tokens(source)):
                        raise RuntimeError(
                            f"AI змінив слова у репліці {cue_id + 1}; правку не застосовано.")
                    completed[cue_id] = suggested
                if set(completed).intersection(expected_ids) != expected_ids:
                    raise RuntimeError("AI не повернув усі репліки поточної групи.")
                self.progress.emit(min(offset + len(batch), len(self.cues)), len(self.cues))
            self.succeeded.emit([
                (cue_id, completed[cue_id]) for cue_id, _ in self.cues
            ])
        except Exception as exc:
            self.failed.emit(str(exc))


class SpeechGenerator(QThread):
    clip_ready = Signal(object)
    progress = Signal(int, int)
    warning = Signal(str)
    status = Signal(str)
    failed = Signal(str)

    def __init__(
        self, cues: list[tuple[int, SubtitleCue]], all_cues: list[SubtitleCue],
        voice_by_character: dict[str, dict], fallback_voice: str,
        translate: bool, folder: str, backend: str,
    ):
        super().__init__()
        self.cues = sorted(cues, key=lambda item: (item[1].start_ms, item[0]))
        self.all_cues = all_cues
        self.voice_by_character = voice_by_character
        self.fallback_voice = fallback_voice
        self.translate = translate
        self.folder = folder
        self.backend = backend

    @staticmethod
    def _prepare_chatterbox_reference(reference: str, cache: dict[str, str]) -> str:
        source = os.path.abspath(reference)
        if not os.path.isfile(source):
            raise RuntimeError(f"Не знайдено WAV/MP3-зразок голосу:\n{source}")
        if source in cache and os.path.isfile(cache[source]):
            return cache[source]
        with tempfile.NamedTemporaryFile(
            prefix="voxora-voice-reference-", suffix=".wav", delete=False
        ) as temporary:
            normalized = temporary.name
        try:
            run_ffmpeg(
                "-i", source, "-t", "8", "-ar", "24000", "-ac", "1",
                "-c:a", "pcm_s16le", normalized,
            )
        except (OSError, RuntimeError):
            try:
                os.remove(normalized)
            except FileNotFoundError:
                pass
            raise
        cache[source] = normalized
        return normalized

    def _fit_to_cue(self, wav_path: str, cue: SubtitleCue, available_ms: int) -> None:
        if available_ms <= 0:
            self.warning.emit(
                f"Для репліки «{cue.text[:40]}» немає вільного проміжку до наступної. "
                "Її буде відтворено послідовно, без накладання.")
            return
        duration = sf.info(wav_path).duration
        available = available_ms / 1000
        if duration <= available + 0.04:
            return
        required_tempo = duration / available
        tempo = min(required_tempo, 1.8)
        fitted_path = wav_path + ".fit.wav"
        try:
            run_ffmpeg(
                "-i", wav_path, "-af", f"rubberband=tempo={tempo:.6f}",
                "-ar", str(AUDIO_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                fitted_path,
            )
            os.replace(fitted_path, wav_path)
        except (OSError, RuntimeError) as exc:
            self.warning.emit(
                f"Не вдалося автоматично вмістити репліку в інтервал: {exc}")
            return
        finally:
            try:
                os.remove(fitted_path)
            except FileNotFoundError:
                pass
        if required_tempo > tempo:
            self.warning.emit(
                f"Репліка «{cue.text[:40]}» довша за проміжок навіть після пришвидшення у 1,8 раза. "
                "Щоб уникнути накладання, наступні репліки можуть трохи зміститися.")

    def run(self):
        stage = "підготовка синтезу"
        server = None
        reference_cache = {}
        success = False
        try:
            if self.backend == "edge":
                import edge_tts
            translator = None
            if self.translate:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source="auto", target="uk")
            if self.backend == "chatterbox":
                server = ChatterboxServer(self.status.emit)
                server.start()
            uploaded_references = {}
            scheduled_end_ms = 0
            for index, (cue_index, cue) in enumerate(self.cues):
                stage = f"переклад репліки {cue_index + 1}"
                manual_dub = cue.dub_text.strip()
                source = manual_dub or cue.text
                spoken = translator.translate(source) if translator and not manual_dub else source
                if not spoken:
                    spoken = source
                profile = (
                    self.voice_by_character.get(cue.character)
                    or self.voice_by_character.get("За замовчуванням", {})
                )
                stem = f"speech_{time.time_ns()}_{index}"
                wav_path = os.path.join(self.folder, stem + ".wav")
                clip_start_ms = max(
                    cue.start_ms,
                    scheduled_end_ms + 20 if scheduled_end_ms else cue.start_ms,
                )
                next_start = min(
                    (other.start_ms for other in self.all_cues
                     if other.start_ms > cue.start_ms),
                    default=cue.end_ms,
                )
                available_ms = next_start - clip_start_ms
                stage = f"синтез репліки {cue_index + 1}"
                if self.backend == "chatterbox":
                    self.status.emit(
                        f"Chatterbox озвучує репліку {cue_index + 1}/{len(self.cues)} "
                        "локально на CPU…")
                if self.backend == "qwen3":
                    reference = profile.get("reference", "")
                    generate_local_speech(spoken, reference, wav_path)
                    voice_label = profile.get("voice") or os.path.basename(reference)
                elif self.backend == "chatterbox":
                    reference = profile.get("reference", "")
                    if not reference:
                        raise RuntimeError(
                            f"Для персонажа «{cue.character or 'За замовчуванням'}» "
                            "не задано зразок голосу.")
                    normalized_reference = self._prepare_chatterbox_reference(
                        reference, reference_cache)
                    if not server:
                        raise RuntimeError("Сервер Chatterbox не запущено.")
                    if normalized_reference not in uploaded_references:
                        uploaded_references[normalized_reference] = server.upload_voice(
                            normalized_reference)
                    server.synthesize(
                        spoken, uploaded_references[normalized_reference],
                        cue.exaggeration, wav_path)
                    voice_label = profile.get("voice") or os.path.basename(reference)
                else:
                    voice = (profile.get("voice")
                             if profile.get("engine", "edge") == "edge"
                             else self.fallback_voice)
                    voice = voice or self.fallback_voice
                    mp3_path = os.path.join(self.folder, stem + ".mp3")
                    try:
                        asyncio.run(edge_tts.Communicate(spoken, voice).save(mp3_path))
                        run_ffmpeg("-i", mp3_path, "-ar", str(AUDIO_RATE), "-ac", "1",
                                   "-c:a", "pcm_s16le", wav_path)
                    finally:
                        try:
                            os.remove(mp3_path)
                        except OSError:
                            pass
                    voice_label = voice
                stage = f"підгонка тривалості репліки {cue_index + 1}"
                self._fit_to_cue(wav_path, cue, available_ms)
                clip_duration_ms = round(sf.info(wav_path).duration * 1000)
                self.clip_ready.emit(AudioClip(
                    wav_path, f"{cue.character or voice_label}: {spoken}",
                    clip_start_ms, index % TRACKS, kind="tts",
                    cue_index=cue_index, cue_id=cue.cue_id))
                scheduled_end_ms = clip_start_ms + clip_duration_ms
                self.progress.emit(index + 1, len(self.cues))
            success = True
        except Exception as exc:
            self.failed.emit(f"Помилка на етапі «{stage}»: {exc}")
        finally:
            for normalized_reference in reference_cache.values():
                try:
                    os.remove(normalized_reference)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    self.warning.emit(
                        f"Не вдалося видалити тимчасовий зразок голосу: {exc}")
            if server:
                try:
                    server.stop(keep_log=not success)
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    self.warning.emit(f"Не вдалося коректно зупинити CrispASR: {exc}")


class AccentJob(QThread):
    succeeded = Signal(int, str)
    failed = Signal(str)

    def __init__(self, model: str, cue_index: int, text: str):
        super().__init__()
        self.model = model
        self.cue_index = cue_index
        self.text = text

    def run(self):
        prompt = (
            "Add Russian or Ukrainian word-stress marks to the supplied sentence. "
            "Preserve every original character and word exactly; only insert the "
            "combining acute accent U+0301 after stressed vowels. Return JSON "
            'with one string field named "text". Input: ' + self.text
        )
        try:
            payload = json.dumps({
                "model": self.model, "stream": False, "format": "json",
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0},
            }).encode("utf-8")
            request = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            body = json.loads(result["message"]["content"])
            text = body.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Ollama не повернув текст із наголосами.")
            self.succeeded.emit(self.cue_index, text.strip())
        except urllib.error.URLError as exc:
            self.failed.emit(
                "Не вдалося з’єднатися з локальною Ollama на localhost:11434.\n"
                + str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceManagerDialog(QDialog):
    def __init__(
        self, parent, voices: list[dict], profiles: list[dict],
        local_voices: list[dict], preview_voice,
    ):
        super().__init__(parent)
        self.setWindowTitle("Персонажі та голоси")
        self.resize(650, 650)
        self.voices = voices
        self.profiles = [dict(profile) for profile in profiles]
        self.local_voices = [dict(voice) for voice in local_voices]
        self.imported_voice_paths: list[str] = []
        self.search = QLineEdit()
        self.search.setPlaceholderText("Пошук збережених голосів або голосів Edge TTS…")
        self.voice_list = QListWidget()
        self.voice_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.character_name = QLineEdit()
        self.character_name.setPlaceholderText(
            "Ім’я персонажа; порожнє поле — голос за замовчуванням")
        self.profile_list = QListWidget()
        self.add_button = QPushButton("Призначити вибрані голоси персонажам")
        self.add_custom_button = QPushButton("Додати голос за ID")
        self.add_local_button = QPushButton("＋ Завантажити голоси WAV/MP3")
        self.remove_local_button = QPushButton("Видалити голос зі списку")
        self.preview_button = QPushButton("▶ Прослухати зразок / голос")
        self.preview_button.setToolTip(
            "Для збереженого локального голосу — відтворити WAV/MP3-зразок; "
            "для Edge TTS — згенерувати коротку пробу.")
        self.remove_button = QPushButton("Видалити персонажа")
        self.done_button = QPushButton("Готово")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Збережені голоси на цьому комп’ютері та каталог Edge TTS"))
        layout.addWidget(self.search)
        layout.addWidget(self.voice_list, 2)
        layout.addWidget(self.character_name)
        layout.addWidget(self.add_button)
        voice_actions = QHBoxLayout()
        voice_actions.addWidget(self.add_custom_button)
        voice_actions.addWidget(self.add_local_button)
        layout.addLayout(voice_actions)
        voice_actions = QHBoxLayout()
        voice_actions.addWidget(self.remove_local_button)
        voice_actions.addWidget(self.preview_button)
        layout.addLayout(voice_actions)
        layout.addWidget(QLabel("Збережені персонажі та їхні голоси"))
        layout.addWidget(self.profile_list, 1)
        row = QHBoxLayout()
        row.addWidget(self.remove_button)
        row.addStretch()
        row.addWidget(self.done_button)
        layout.addLayout(row)
        self.search.textChanged.connect(self._fill_voices)
        self.add_button.clicked.connect(self._assign_voice)
        self.add_custom_button.clicked.connect(self._add_custom_voice)
        self.add_local_button.clicked.connect(self._import_local_voices)
        self.remove_local_button.clicked.connect(self._remove_local_voice)
        self.preview_button.clicked.connect(self._preview_selected)
        self._preview_voice = preview_voice
        self.remove_button.clicked.connect(self._remove_profile)
        self.done_button.clicked.connect(self.accept)
        self._fill_voices()
        self._fill_profiles()

    def _fill_voices(self):
        query = self.search.text().strip().lower()
        self.voice_list.clear()
        for voice in self.local_voices:
            label = f"Локальний · {voice['name']}"
            if not query or query in label.lower():
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, {
                    "name": voice["name"],
                    "voice": voice["name"],
                    "engine": "qwen3",
                    "reference": voice["path"],
                })
                self.voice_list.addItem(item)
        for voice in self.voices:
            label = f"{voice['locale']} · {voice['name']} · {voice.get('gender', '')}"
            if not query or query in label.lower():
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, voice)
                self.voice_list.addItem(item)

    def _fill_profiles(self):
        self.profile_list.clear()
        for profile in self.profiles:
            if profile.get("engine") == "qwen3":
                voice = f"Локальний зразок · {os.path.basename(profile.get('reference', ''))}"
            else:
                voice = profile["voice"]
            self.profile_list.addItem(f"{profile['character']}  →  {voice}")

    def _assign_voice(self):
        items = self.voice_list.selectedItems()
        if not items:
            QMessageBox.information(self, APP_TITLE, "Виберіть один або кілька голосів зі списку.")
            return
        assignments = []
        first_character = self.character_name.text().strip()
        for index, item in enumerate(items):
            voice = item.data(Qt.ItemDataRole.UserRole)
            if len(items) == 1:
                character = first_character
                if not character and voice.get("engine") == "qwen3":
                    character = "За замовчуванням"
                if not character:
                    QMessageBox.information(
                        self, APP_TITLE, "Вкажіть персонажа для вибраного голосу.")
                    return
            else:
                voice_name = voice.get("voice", voice.get("name", ""))
                character, accepted = QInputDialog.getText(
                    self, "Призначення голосів",
                    f"Персонаж для голосу «{voice_name}» ({index + 1} із {len(items)}):",
                    text=first_character if index == 0 else "")
                if not accepted:
                    return
                character = character.strip()
                if not character:
                    QMessageBox.information(
                        self, APP_TITLE, "Для кожного вибраного голосу потрібно вказати персонажа.")
                    return
            assignments.append((voice, character))

        characters = [character.casefold() for _, character in assignments]
        if len(set(characters)) != len(characters):
            QMessageBox.warning(
                self, APP_TITLE,
                "Кожному вибраному голосу призначте окремого персонажа.")
            return
        for voice, character in assignments:
            self._save_voice_assignment(voice, character)
        self._fill_profiles()
        self.profile_list.setCurrentRow(len(self.profiles) - 1)

    def _save_voice_assignment(self, voice: dict, character: str):
        profile = next((p for p in self.profiles if p["character"] == character), None)
        if voice.get("engine") == "qwen3":
            assignment = {
                "character": character,
                "voice": voice["voice"],
                "engine": "qwen3",
                "reference": voice["reference"],
            }
        else:
            assignment = {
                "character": character,
                "voice": voice["name"],
                "engine": "edge",
            }
        if profile:
            profile.update(assignment)
            if voice.get("engine") != "qwen3":
                profile.pop("reference", None)
        else:
            self.profiles.append(assignment)

    def _import_local_voices(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Завантажте один або кілька зразків голосу",
            str(VOICE_SAMPLES_DIR if VOICE_SAMPLES_DIR.is_dir() else Path.home()),
            "Зразки голосу (*.wav *.mp3)")
        if not paths:
            return
        imported = []
        errors = []
        try:
            VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
            for source in paths:
                original = Path(source)
                target = VOICE_LIBRARY_DIR / original.name
                suffix = original.suffix
                stem = original.stem
                counter = 2
                while target.exists():
                    target = VOICE_LIBRARY_DIR / f"{stem} ({counter}){suffix}"
                    counter += 1
                try:
                    shutil.copy2(source, target)
                    entry = {"name": target.stem, "path": str(target.resolve())}
                    self.local_voices.append(entry)
                    self.imported_voice_paths.append(str(target.resolve()))
                    imported.append(original.name)
                except OSError as exc:
                    errors.append(f"{original.name}: {exc}")
        except OSError as exc:
            errors.append(str(exc))
        self._fill_voices()
        if imported:
            QMessageBox.information(
                self, APP_TITLE,
                f"Додано голосів: {len(imported)}. Вони збережуться в бібліотеці Voxora "
                "після натискання «Готово».\n\nТепер виберіть голос у списку, введіть ім’я "
                "персонажа й натисніть «Призначити вибраний голос персонажу».")
        if errors:
            QMessageBox.warning(self, APP_TITLE, "Не вдалося додати деякі файли:\n" + "\n".join(errors))

    def _remove_local_voice(self):
        item = self.voice_list.currentItem()
        voice = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not voice or voice.get("engine") != "qwen3":
            QMessageBox.information(self, APP_TITLE, "Спочатку виберіть збережений локальний голос.")
            return
        answer = QMessageBox.question(
            self, "Видалити збережений голос",
            f"Видалити «{voice['voice']}» з бібліотеки Voxora та стерти його копію з диска? "
            "Призначення цього зразка в проєктах перестануть працювати.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        reference = os.path.abspath(voice["reference"])
        self.local_voices = [
            entry for entry in self.local_voices
            if os.path.abspath(entry["path"]) != reference
        ]
        self._fill_voices()

    def discard_imported_voices(self) -> list[str]:
        errors = []
        for path in self.imported_voice_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        return errors

    def _remove_profile(self):
        row = self.profile_list.currentRow()
        if 0 <= row < len(self.profiles):
            self.profiles.pop(row)
            self._fill_profiles()

    def _add_custom_voice(self):
        voice_id, accepted = QInputDialog.getText(
            self, "Додати голос за ID", "Вставте точний ShortName голосу Edge TTS:")
        if not accepted or not voice_id.strip():
            return
        character = self.character_name.text().strip()
        if not character:
            QMessageBox.information(self, APP_TITLE, "Спочатку введіть ім’я персонажа.")
            return
        voice_id = voice_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9-]+", voice_id):
            QMessageBox.warning(self, APP_TITLE, "ID має містити лише латинські літери, цифри та дефіси.")
            return
        profile = next((p for p in self.profiles if p["character"] == character), None)
        if profile:
            profile.update({"voice": voice_id, "engine": "edge"})
            profile.pop("reference", None)
        else:
            self.profiles.append(
                {"character": character, "voice": voice_id, "engine": "edge"})
        self._fill_profiles()

    def _preview_selected(self):
        item = self.voice_list.currentItem()
        if item:
            voice = item.data(Qt.ItemDataRole.UserRole)
            self._preview_voice(voice if voice.get("engine") == "qwen3" else voice["name"])
            return
        row = self.profile_list.currentRow()
        if self.profile_list.hasFocus() and 0 <= row < len(self.profiles):
            profile = self.profiles[row]
            self._preview_voice(
                profile if profile.get("engine") == "qwen3" else profile["voice"])
            return
        if 0 <= row < len(self.profiles):
            profile = self.profiles[row]
            self._preview_voice(
                profile if profile.get("engine") == "qwen3" else profile["voice"])


class Timeline(QWidget):
    clipMoved = Signal(int, int, int)
    seekRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(TRACKS * TRACK_HEIGHT + 28)
        self.setMouseTracking(True)
        self.clips: list[AudioClip] = []
        self.clip_durations_ms: list[float | None] = []
        self.duration_ms = 60_000
        self.scale = 0.02
        self.playhead_ms = 0
        self._drag_index = -1
        self._drag_offset = 0
        self._drag_position: tuple[int, int] | None = None

    def set_data(self, clips: list[AudioClip], duration_ms: int):
        self.clips = clips
        self.clip_durations_ms = []
        for clip in clips:
            try:
                info = sf.info(clip.path)
                duration = info.frames * 1000 / info.samplerate
                if clip.trim_end_ms > clip.trim_start_ms:
                    duration = min(duration, clip.trim_end_ms) - clip.trim_start_ms
                else:
                    duration -= clip.trim_start_ms
                self.clip_durations_ms.append(max(0.0, duration))
            except (OSError, RuntimeError, ZeroDivisionError):
                self.clip_durations_ms.append(None)
        self.duration_ms = max(1000, duration_ms)
        self.resize(self.sizeHint())
        self.updateGeometry()
        self.update()

    def set_zoom(self, pixels_per_second: int):
        self.scale = max(0.005, min(0.1, pixels_per_second / 1000))
        self.resize(self.sizeHint())
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        content_width = round(max(self.duration_ms, 60_000) * self.scale)
        return QSize(max(900, content_width), TRACKS * TRACK_HEIGHT + 28)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171c24"))
        width_ms = max(self.duration_ms, 60_000)
        for track in range(TRACKS):
            top = track * TRACK_HEIGHT + 20
            painter.fillRect(0, top, self.width(), TRACK_HEIGHT - 1, QColor("#222a35"))
            painter.setPen(QPen(QColor("#3a4554"), 1))
            painter.drawLine(0, top + TRACK_HEIGHT - 1, self.width(), top + TRACK_HEIGHT - 1)
            painter.setPen(QColor("#aeb9c8"))
            painter.drawText(5, top - 4, f"{track + 1}")
        painter.setPen(QPen(QColor("#4a5360"), 1))
        for seconds in range(0, width_ms // 1000 + 1, 10):
            x = round(seconds * 1000 * self.scale)
            painter.drawLine(x, 20, x, TRACKS * TRACK_HEIGHT + 20)
            painter.setPen(QColor("#bdc7d4"))
            painter.drawText(x + 2, 15, format_time(seconds * 1000))
            painter.setPen(QPen(QColor("#4a5360"), 1))
        visible = event.rect()
        for index, clip in enumerate(self.clips):
            duration = self.clip_durations_ms[index]
            if duration is None:
                continue
            width = max(8, duration * self.scale)
            x = clip.start_ms * self.scale
            y = clip.track * TRACK_HEIGHT + 24
            if self._drag_index == index and self._drag_position:
                x = self._drag_position[0] * self.scale
                y = self._drag_position[1] * TRACK_HEIGHT + 24
            if x + width < visible.left() or x > visible.right():
                continue
            if y + TRACK_HEIGHT - 9 < visible.top() or y > visible.bottom():
                continue
            painter.setPen(QPen(QColor("#dce5ef"), 1))
            painter.setBrush(QColor(TRACK_COLORS[clip.track % len(TRACK_COLORS)]))
            painter.drawRoundedRect(round(x), y, round(width), TRACK_HEIGHT - 9, 4, 4)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(round(x) + 5, y + 18, clip.name[:28])
        play_x = round(self.playhead_ms * self.scale)
        painter.setPen(QPen(QColor("#ffdf5d"), 2))
        painter.drawLine(play_x, 0, play_x, TRACKS * TRACK_HEIGHT + 20)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        point = event.position().toPoint()
        for index in range(len(self.clips) - 1, -1, -1):
            clip = self.clips[index]
            duration = self.clip_durations_ms[index]
            if duration is None:
                continue
            x = clip.start_ms * self.scale
            y = clip.track * TRACK_HEIGHT + 24
            width = max(8, duration * self.scale)
            if x <= point.x() <= x + width and y <= point.y() <= y + TRACK_HEIGHT - 9:
                self._drag_index = index
                self._drag_offset = point.x() - round(x)
                self._drag_position = (clip.start_ms, clip.track)
                self.grabMouse()
                return
        self.seekRequested.emit(max(0, round(point.x() / self.scale)))

    def mouseMoveEvent(self, event):
        if self._drag_index < 0 or not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)
        point = event.position().toPoint()
        start_ms = max(0, round((point.x() - self._drag_offset) / self.scale))
        track = max(0, min(TRACKS - 1, (point.y() - 20) // TRACK_HEIGHT))
        self._drag_position = (start_ms, track)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_index >= 0:
            if self._drag_position:
                self.clipMoved.emit(self._drag_index, *self._drag_position)
            self.releaseMouse()
            self._drag_index = -1
            self._drag_position = None
            self.update()


class VoxoraStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1360, 860)
        self.project_path = ""
        self.video_path = ""
        self.playback_video_path = ""
        self.background_path = ""
        self.background_source = ""
        self.clips: list[AudioClip] = []
        self.cues: list[SubtitleCue] = []
        self.voice_profiles: list[dict] = []
        self._replacement_cue_ids: set[str] = set()
        self._accent_pending_row: int | None = None
        self.ai_settings = QSettings("Voxora", "Voxora Studio")
        self.local_voice_library: list[dict] = []
        self.voice_catalog: list[dict] = []
        self.duration_ms = 60_000
        self.job: QThread | None = None
        self.voice_catalog_job: VoiceCatalogJob | None = None
        self.voice_sample_job: VoiceSampleJob | None = None
        self.preview_player: QMediaPlayer | None = None
        self.preview_output: QAudioOutput | None = None
        self.preview_path = ""
        self.preview_active = False
        self.preview_has_background = False
        self.recorder: AudioRecorder | None = None
        self._closing = False
        self._make_ui()
        self._apply_theme()
        try:
            self.local_voice_library = load_voice_library()
        except RuntimeError as exc:
            QMessageBox.warning(self, APP_TITLE, str(exc))
        self.player.errorOccurred.connect(self._media_error)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._playback_changed)
        self.player.mediaStatusChanged.connect(self._video_media_status_changed)
        self.preview_player.mediaStatusChanged.connect(self._sample_media_status_changed)
        self.preview_player.errorOccurred.connect(self._preview_media_error)
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(120)
        self._seek_timer.timeout.connect(self._perform_pending_seek)
        self._pending_seek_position: int | None = None
        self._seek_in_progress = False
        self._seek_target_position: int | None = None
        self._resume_after_seek = False
        self._seek_deadline = 0.0
        self._seek_resume_timer = QTimer(self)
        self._seek_resume_timer.setInterval(50)
        self._seek_resume_timer.timeout.connect(self._check_seek_completion)
        self._preview_sync_timer = QTimer(self)
        self._preview_sync_timer.setInterval(700)
        self._preview_sync_timer.timeout.connect(self._sync_preview_position)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(30_000)
        self._autosave_timer.timeout.connect(self.save_project)
        self._autosave_timer.start()

    def _make_ui(self):
        toolbar = QToolBar("Project")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for title, slot in (
            ("Новий проєкт", self.new_project),
            ("Відкрити проєкт", self.open_project),
            ("Зберегти", self.save_project),
            ("Відео…", self.open_video),
            ("Озвучка з Dub Studio…", self.import_dub_studio_voiceover),
            ("＋ Аудіо", self.import_audio),
            ("● Запис", self.toggle_recording),
            ("Субтитри…", self.import_subtitles),
            ("Персонажі / голоси", self.manage_voices),
            ("AI-помічник…", self.configure_ai_assistant),
            ("Створити голосові кліпи", self.generate_speech),
            ("▶ Прослухати результат", self.preview_result),
            ("Відділити мовлення (AI)", self.separate_original_dialogue),
            ("Експорт MP4", self.export_video),
        ):
            action = QAction(title, self)
            action.triggered.connect(slot)
            toolbar.addAction(action)
        self.project_label = QLabel("  Проєкт не відкрито  ")
        toolbar.addWidget(self.project_label)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.55)
        self.player.setAudioOutput(self.audio_output)
        self.preview_player = QMediaPlayer(self)
        self.preview_output = QAudioOutput(self)
        self.preview_output.setVolume(1.0)
        self.preview_player.setAudioOutput(self.preview_output)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        self.video_widget.setMinimumSize(420, 250)
        self.video_button = QPushButton("Вибрати відео…")
        self.video_button.clicked.connect(self.open_video)
        self.play_button = QPushButton("▶ Відтворити")
        self.play_button.clicked.connect(self.toggle_play)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, self.duration_ms)
        self.seek_slider.sliderMoved.connect(self._preview_seek_position)
        self.seek_slider.sliderReleased.connect(self._seek_from_slider)
        self.original_volume = QSlider(Qt.Orientation.Horizontal)
        self.original_volume.setRange(0, 100)
        self.original_volume.setValue(55)
        self.original_volume.setFixedWidth(110)
        self.original_volume.setToolTip(
            "Рівень оригінального аудіо; під час прослуховування дубляжу — "
            "гучність відділеного фону та ефектів.")
        self.original_volume.valueChanged.connect(self._original_volume_changed)

        video_box = QWidget()
        video_layout = QVBoxLayout(video_box)
        video_layout.setContentsMargins(4, 4, 4, 4)
        video_layout.addWidget(self.video_widget, 1)
        controls = QHBoxLayout()
        controls.addWidget(self.video_button, 1)
        controls.addWidget(self.play_button)
        controls.addWidget(self.time_label)
        controls.addWidget(QLabel("Фон / оригінал"))
        controls.addWidget(self.original_volume)
        video_layout.addLayout(controls)
        video_layout.addWidget(self.seek_slider)

        self.clip_table = QTableWidget(0, 7)
        self.clip_table.setHorizontalHeaderLabels(
            ["Позиція", "Доріжка", "Кліп", "Гучність", "Обрізати від (мс)",
             "Обрізати до (мс)", "Вимкнено"])
        self.clip_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.clip_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.clip_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.clip_table.itemChanged.connect(self._clip_item_changed)
        self.clip_table.cellDoubleClicked.connect(self._seek_to_clip_row)
        clip_controls = QHBoxLayout()
        self.preview_button = QPushButton("▶ Прослухати кліп")
        self.preview_button.setToolTip(
            "Спочатку виберіть кліп у списку. Для нового дубляжу натисніть "
            "«Створити голосові кліпи» у верхній панелі.")
        self.preview_button.clicked.connect(self.preview_selected_clip)
        self.stop_preview_button = QPushButton("■ Зупинити аудіо")
        self.stop_preview_button.clicked.connect(self._stop_audio)
        self.delete_clip_button = QPushButton("Видалити вибрані")
        self.delete_clip_button.clicked.connect(self.delete_selected_clips)
        clip_controls.addWidget(self.preview_button)
        clip_controls.addWidget(self.stop_preview_button)
        clip_controls.addWidget(self.delete_clip_button)
        clip_box = QWidget()
        clip_layout = QVBoxLayout(clip_box)
        clip_layout.addWidget(QLabel("Аудіокліпи — подвійне клацання по позиції відкриває таймкод"))
        clip_layout.addWidget(self.clip_table, 1)
        clip_layout.addLayout(clip_controls)

        self.cue_table = QTableWidget(0, 6)
        self.cue_table.setHorizontalHeaderLabels(
            ["Початок", "Кінець", "Персонаж", "Субтитр", "Текст озвучення",
             "Виразність"])
        self.cue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.cue_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.cue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.cue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.cue_table.setColumnWidth(5, 90)
        self.cue_table.itemChanged.connect(self._cue_item_changed)
        self.translate_check = QCheckBox("Перекласти текст на українську перед синтезом")
        self.tts_backend_combo = QComboBox()
        self.tts_backend_combo.addItem("Qwen3 · локально", "qwen3")
        self.tts_backend_combo.addItem("Chatterbox · локально", "chatterbox")
        self.tts_backend_combo.addItem("Edge TTS · онлайн", "edge")
        self.tts_backend_combo.setToolTip(
            "Qwen3 і Chatterbox працюють локально з WAV/MP3-зразками голосів. "
            "Chatterbox також має повзунок виразності для кожної репліки.")
        self.tts_backend_combo.setMinimumWidth(145)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["uk-UA-PolinaNeural", "uk-UA-OstapNeural", "en-US-JennyNeural"])
        self.tts_backend_combo.currentIndexChanged.connect(self._update_tts_controls)
        self.subtitle_add = QPushButton("＋ Репліка")
        self.subtitle_add.clicked.connect(self.add_cue)
        self.subtitle_delete = QPushButton("− Видалити репліку")
        self.subtitle_delete.clicked.connect(self.delete_cue)
        self.ollama_button = QPushButton("Персонажі через Ollama")
        self.ollama_button.clicked.connect(self.assign_characters_with_ollama)
        self.regenerate_cue_button = QPushButton("Оновити звук репліки")
        self.regenerate_cue_button.setToolTip(
            "Створити заново аудіо лише для вибраної репліки; інші кліпи не зміняться.")
        self.regenerate_cue_button.clicked.connect(self._regenerate_selected_cue)
        self.accent_cue_button = QPushButton("Наголос AI")
        self.accent_cue_button.setToolTip(
            "Локальна Ollama запропонує наголоси для вибраного тексту. "
            "Після перевірки тексту оновиться лише ця аудіорепліка.")
        self.accent_cue_button.clicked.connect(self.accent_selected_cue)
        self.ai_assist_button = QPushButton("AI: переклад / інтонація")
        self.ai_assist_button.setToolTip(
            "Для вибраних реплік запропонувати переклад, виразну пунктуацію або наголоси. "
            "Переглянь зміни перед застосуванням.")
        self.ai_assist_button.clicked.connect(self.assist_selected_cues)
        cue_box = QWidget()
        cue_layout = QVBoxLayout(cue_box)
        cue_layout.addWidget(QLabel("Субтитри й текст дубляжу"))
        cue_layout.addWidget(self.cue_table, 1)
        cue_layout.addWidget(QLabel(
            "Зразок голосу вибирається в «Персонажі / голоси»; щоб створити озвучку з тексту, "
            "натисніть «Створити голосові кліпи» зверху."))
        cue_row = QHBoxLayout()
        cue_row.addWidget(self.subtitle_add)
        cue_row.addWidget(self.subtitle_delete)
        cue_row.addWidget(self.ollama_button)
        cue_row.addWidget(self.tts_backend_combo)
        cue_row.addWidget(self.voice_combo, 1)
        cue_row.addWidget(self.translate_check)
        cue_layout.addLayout(cue_row)
        cue_edit_row = QHBoxLayout()
        cue_edit_row.addWidget(self.regenerate_cue_button)
        cue_edit_row.addWidget(self.accent_cue_button)
        cue_edit_row.addWidget(self.ai_assist_button)
        cue_edit_row.addStretch()
        cue_layout.addLayout(cue_edit_row)
        cue_layout.addWidget(QLabel(
            "Той самий зразок голосу використовується для кожної репліки персонажа. "
            "Chatterbox підтримує російську й дає змогу налаштувати виразність кожної репліки "
            "(0–2, типове значення 0,5; це не вибір конкретної емоції)."))

        self.timeline = Timeline()
        self.timeline.clipMoved.connect(self._move_clip)
        self.timeline.seekRequested.connect(self._request_seek)
        self.timeline_box = QWidget()
        timeline_layout = QVBoxLayout(self.timeline_box)
        timeline_heading = QHBoxLayout()
        timeline_heading.addWidget(QLabel("Таймлайн — перетягніть кліп між доріжками; клацання — перехід до часу"))
        timeline_heading.addStretch()
        timeline_heading.addWidget(QLabel("Масштаб"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(5, 100)
        self.zoom_slider.setValue(20)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self.timeline.set_zoom)
        timeline_heading.addWidget(self.zoom_slider)
        timeline_layout.addLayout(timeline_heading)
        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(False)
        timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        timeline_scroll.setWidget(self.timeline)
        timeline_layout.addWidget(timeline_scroll)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(video_box)
        top.addWidget(cue_box)
        top.setSizes([560, 780])
        lower = QSplitter(Qt.Orientation.Vertical)
        lower.addWidget(top)
        lower.addWidget(clip_box)
        lower.addWidget(self.timeline_box)
        lower.setSizes([430, 220, 270])
        self.setCentralWidget(lower)
        self._add_shortcut("Ctrl+S", self.save_project)
        self._add_shortcut("Space", self.toggle_play)
        self._add_shortcut("Ctrl+O", self.open_project)
        self._add_shortcut("Ctrl+I", self.import_audio)
        self._add_shortcut("Delete", self.delete_selected_clips)

    def _add_shortcut(self, sequence: str, slot):
        action = QAction(self)
        action.setShortcut(sequence)
        action.triggered.connect(slot)
        self.addAction(action)

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #202630; color: #e8edf3; font: 10pt 'Segoe UI'; }
            QToolBar { background: #171c24; border: 0; spacing: 4px; padding: 5px; }
            QPushButton { background: #303a49; border: 1px solid #465366; border-radius: 4px; padding: 6px 9px; }
            QPushButton:hover { background: #40516a; }
            QTableWidget, QComboBox { background: #171c24; border: 1px solid #394554; }
            QHeaderView::section { background: #303a49; padding: 5px; border: 0; }
            QProgressBar { border: 1px solid #465366; text-align: center; }
            QProgressBar::chunk { background: #4a90e2; }
        """)

    def _error(self, title: str, details: str):
        if "403" in details and "speech.platform.bing.com" in details:
            details += (
                "\n\nEdge TTS відхилив запит. Оновіть пакет: "
                "python -m pip install --upgrade edge-tts. "
                "Якщо помилка повторюється, скористайтеся імпортом готового аудіо."
            )
        QMessageBox.critical(self, title, details)

    def configure_ai_assistant(self):
        provider = self.ai_settings.value("ai/provider", "ollama")
        endpoint_default = (
            "https://api.deepseek.com/v1" if provider == "deepseek"
            else "http://127.0.0.1:11434")
        model_default = "deepseek-chat" if provider == "deepseek" else "gemma2:2b"
        try:
            api_key = unprotect_api_key(
                str(self.ai_settings.value("ai/deepseek_key", "")))
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "AI-помічник", str(exc))
            api_key = ""
        configuration = {
            "provider": provider,
            "endpoint": str(self.ai_settings.value("ai/endpoint", endpoint_default)),
            "model": str(self.ai_settings.value("ai/model", model_default)),
            "api_key": api_key,
        }
        dialog = AISettingsDialog(self, configuration)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        parsed = urllib.parse.urlparse(values["endpoint"])
        if (not values["endpoint"] or not parsed.scheme or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            QMessageBox.warning(self, "AI-помічник", "Вкажіть повну адресу AI-сервера.")
            return
        if values["provider"] == "deepseek":
            if parsed.scheme != "https":
                QMessageBox.warning(
                    self, "AI-помічник",
                    "Для DeepSeek API потрібна адреса HTTPS, щоб не передавати API-ключ "
                    "незахищеним з’єднанням.")
                return
            if not values["api_key"]:
                QMessageBox.warning(self, "AI-помічник", "Введіть API-ключ DeepSeek.")
                return
        elif parsed.scheme not in {"http", "https"}:
            QMessageBox.warning(self, "AI-помічник", "Адреса сервера має починатися з HTTP або HTTPS.")
            return
        elif (parsed.scheme == "http"
              and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}):
            QMessageBox.warning(
                self, "AI-помічник",
                "Незашифроване HTTP-з’єднання дозволене лише до локальної Ollama. "
                "Для інших серверів використовуйте HTTPS.")
            return
        if not values["model"]:
            QMessageBox.warning(self, "AI-помічник", "Вкажіть назву моделі.")
            return
        try:
            if values["api_key"]:
                encrypted_key = protect_api_key(values["api_key"])
                self.ai_settings.setValue("ai/deepseek_key", encrypted_key)
            self.ai_settings.setValue("ai/provider", values["provider"])
            self.ai_settings.setValue("ai/endpoint", values["endpoint"])
            self.ai_settings.setValue("ai/model", values["model"])
            self.ai_settings.sync()
            if self.ai_settings.status() != QSettings.Status.NoError:
                raise RuntimeError("Windows не зміг зберегти налаштування AI.")
        except (OSError, RuntimeError) as exc:
            self._error("Не вдалося зберегти налаштування AI", str(exc))
            return
        provider_name = "DeepSeek API" if values["provider"] == "deepseek" else "Ollama"
        self.statusBar().showMessage(f"AI-помічник налаштований: {provider_name} · {values['model']}", 8000)

    def _ai_assistant_configuration(self) -> dict:
        provider = str(self.ai_settings.value("ai/provider", "ollama"))
        configuration = {
            "provider": provider,
            "endpoint": str(self.ai_settings.value(
                "ai/endpoint",
                "https://api.deepseek.com/v1" if provider == "deepseek"
                else "http://127.0.0.1:11434")),
            "model": str(self.ai_settings.value(
                "ai/model", "deepseek-chat" if provider == "deepseek" else "gemma2:2b")),
        }
        if provider == "deepseek":
            try:
                configuration["api_key"] = unprotect_api_key(
                    str(self.ai_settings.value("ai/deepseek_key", "")))
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Не вдалося прочитати збережений API-ключ. Відкрийте налаштування "
                    f"AI-помічника і введіть його знову.\n{exc}") from exc
            if not configuration["api_key"]:
                raise RuntimeError(
                    "Для DeepSeek потрібен API-ключ. Відкрийте налаштування AI-помічника.")
        if not configuration["endpoint"].strip() or not configuration["model"].strip():
            raise RuntimeError("Відкрийте налаштування AI-помічника й укажіть сервер та модель.")
        return configuration

    def _update_tts_controls(self, _index: int = 0):
        backend = self.tts_backend_combo.currentData()
        local = backend in {"qwen3", "chatterbox"}
        self.voice_combo.setEnabled(not local)
        self.voice_combo.setToolTip(
            "Голос з цього списку використовується Edge TTS."
            if not local else "Для локального синтезу голос задається WAV/MP3-зразком персонажа.")
        for row in range(self.cue_table.rowCount()):
            control = self.cue_table.cellWidget(row, 5)
            if isinstance(control, QDoubleSpinBox):
                control.setEnabled(backend == "chatterbox")
        self.save_project()

    def _project_media_dir(self) -> str:
        if not self.project_path:
            raise RuntimeError("Спершу створіть або збережіть проєкт.")
        folder = os.path.join(os.path.dirname(self.project_path), "voxora_media")
        os.makedirs(folder, exist_ok=True)
        return folder

    def new_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Новий проєкт", "", "Voxora Studio (*.voxora)")
        if not path:
            return
        if not path.lower().endswith(".voxora"):
            path += ".voxora"
        self._stop_preview_players(remove_file=True)
        self._stop_audio()
        self.player.stop()
        self._reset_seek_state()
        self.project_path = path
        self.video_path = ""
        self.playback_video_path = ""
        self.background_path = ""
        self.background_source = ""
        self.clips.clear()
        self.cues.clear()
        self.voice_profiles.clear()
        self.player.setSource(QUrl())
        self.duration_ms = 60_000
        self.video_button.setText("Вибрати відео…")
        self._refresh()
        self.save_project()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Відкрити проєкт", "", "Voxora Studio (*.voxora)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            if data.get("version") != APP_VERSION:
                raise ValueError("Версія проєкту не підтримується цією версією програми.")
            new_clips = [AudioClip(**entry) for entry in data.get("clips", [])]
            new_cues = [SubtitleCue(**entry) for entry in data.get("cues", [])]
            if any(cue.start_ms < 0 or cue.end_ms <= cue.start_ms for cue in new_cues):
                raise ValueError("Проєкт містить некоректний час субтитрів.")
            if any(
                isinstance(cue.exaggeration, bool)
                or not isinstance(cue.exaggeration, (int, float))
                or not 0 <= cue.exaggeration <= 2
                for cue in new_cues
            ):
                raise ValueError("Проєкт містить некоректний рівень виразності голосу.")
            for clip in new_clips:
                if not isinstance(clip.track, int) or not 0 <= clip.track < TRACKS:
                    raise ValueError("Проєкт містить некоректний номер аудіодоріжки.")
                if clip.start_ms < 0 or not 0 <= clip.gain <= 4:
                    raise ValueError("Проєкт містить некоректний час або гучність кліпу.")
                if clip.trim_start_ms < 0 or (
                    clip.trim_end_ms and clip.trim_end_ms <= clip.trim_start_ms
                ):
                    raise ValueError("Проєкт містить некоректне обрізання аудіо.")
            self._stop_audio()
            self.player.stop()
            self._reset_seek_state()
            self.project_path = path
            self.video_path = data.get("video", "")
            if self.video_path and not os.path.isabs(self.video_path):
                self.video_path = os.path.normpath(os.path.join(os.path.dirname(path), self.video_path))
            self.playback_video_path = ""
            self.background_path = data.get("background_path", "")
            self.background_source = data.get("background_source", "")
            if self.background_path and not os.path.isabs(self.background_path):
                self.background_path = os.path.normpath(
                    os.path.join(os.path.dirname(path), self.background_path))
            if self.background_source and not os.path.isabs(self.background_source):
                self.background_source = os.path.normpath(
                    os.path.join(os.path.dirname(path), self.background_source))
            for clip in new_clips:
                if not os.path.isabs(clip.path):
                    clip.path = os.path.normpath(os.path.join(os.path.dirname(path), clip.path))
            self._stop_preview_players(remove_file=True)
            self.clips = new_clips
            self.cues = new_cues
            self._reconcile_cue_clip_links()
            self.voice_profiles = [
                dict(profile) for profile in data.get("voice_profiles", [])
                if profile.get("character") and profile.get("voice")
            ]
            backend_index = self.tts_backend_combo.findData(data.get("tts_backend", "qwen3"))
            if backend_index >= 0:
                self.tts_backend_combo.setCurrentIndex(backend_index)
            self.duration_ms = max(1000, int(data.get("duration_ms", 60_000)))
            self.player.setSource(QUrl())
            self.video_button.setText(os.path.basename(self.video_path) if self.video_path else "Вибрати відео…")
            self._refresh()
            if self.video_path and os.path.isfile(self.video_path):
                self._start_video_playback(self.video_path)
            elif self.video_path:
                QMessageBox.warning(
                    self, APP_TITLE,
                    f"Не знайдено відеофайл проєкту:\n{self.video_path}\n\n"
                    "Виберіть його кнопкою «Відео…».")
            self.save_project()
            missing = sum(not os.path.isfile(clip.path) for clip in self.clips)
            if missing:
                QMessageBox.warning(self, APP_TITLE, f"Не знайдено аудіофайлів кліпів: {missing}.")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._error("Не вдалося відкрити проєкт", str(exc))

    def _reconcile_cue_clip_links(self):
        seen_ids: set[str] = set()
        for cue in self.cues:
            if not cue.cue_id or cue.cue_id in seen_ids:
                cue.cue_id = uuid.uuid4().hex
            seen_ids.add(cue.cue_id)
        cues_by_id = {cue.cue_id: index for index, cue in enumerate(self.cues)}
        linked_ids: set[str] = set()
        for clip in self.clips:
            if clip.kind == "tts" and clip.cue_id and clip.cue_id not in cues_by_id:
                clip.cue_id = ""
                clip.cue_index = -1

        def link(clip: AudioClip, cue_index: int):
            cue = self.cues[cue_index]
            clip.cue_id = cue.cue_id
            clip.cue_index = cue_index
            linked_ids.add(cue.cue_id)

        for clip in self.clips:
            if clip.kind != "tts" or clip.cue_id not in cues_by_id:
                continue
            cue_index = cues_by_id[clip.cue_id]
            if clip.cue_id not in linked_ids:
                link(clip, cue_index)
            else:
                clip.cue_id = ""
                clip.cue_index = -1

        for clip in self.clips:
            if clip.kind != "tts" or clip.cue_id:
                continue
            cue_index = clip.cue_index
            if not 0 <= cue_index < len(self.cues) or self.cues[cue_index].cue_id in linked_ids:
                continue
            cue = self.cues[cue_index]
            cue_keys = {
                subtitle_match_key(cue.text),
                subtitle_match_key(cue.dub_text),
            } - {""}
            if clip_match_key(clip.name) in cue_keys:
                link(clip, cue_index)

        for clip in self.clips:
            if clip.kind != "tts" or clip.cue_id:
                continue
            clip_key = clip_match_key(clip.name)
            if not clip_key:
                continue
            candidates = [
                index for index, cue in enumerate(self.cues)
                if cue.cue_id not in linked_ids and clip_key in {
                    subtitle_match_key(cue.text),
                    subtitle_match_key(cue.dub_text),
                }
            ]
            if candidates:
                cue_index = min(
                    candidates,
                    key=lambda index: abs(self.cues[index].start_ms - clip.start_ms),
                )
                link(clip, cue_index)

    def save_project(self):
        if not self.project_path:
            return
        project_dir = os.path.dirname(os.path.abspath(self.project_path))

        def project_relative(path):
            if not path:
                return path
            absolute = os.path.abspath(path)
            try:
                if os.path.commonpath((project_dir, absolute)) == project_dir:
                    return os.path.relpath(absolute, project_dir)
            except ValueError:
                pass
            return path

        video = project_relative(self.video_path)
        background = project_relative(self.background_path)
        background_source = project_relative(self.background_source)
        clips = []
        for clip in self.clips:
            entry = asdict(clip)
            entry["path"] = project_relative(clip.path)
            clips.append(entry)
        data = {
            "version": APP_VERSION,
            "video": video,
            "background_path": background,
            "background_source": background_source,
            "duration_ms": self.duration_ms,
            "clips": clips,
            "cues": [asdict(cue) for cue in self.cues],
            "voice_profiles": self.voice_profiles,
            "tts_backend": self.tts_backend_combo.currentData(),
        }
        temporary = self.project_path + ".tmp"
        backup = self.project_path + ".bak"
        backup_temporary = backup + ".tmp"
        backup_warning = ""
        try:
            with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if os.path.isfile(self.project_path):
                try:
                    with open(self.project_path, encoding="utf-8") as existing:
                        previous = json.load(existing)
                    if isinstance(previous, dict) and previous.get("version") == APP_VERSION:
                        shutil.copy2(self.project_path, backup_temporary)
                        with open(backup_temporary, "r+b") as backup_stream:
                            os.fsync(backup_stream.fileno())
                        os.replace(backup_temporary, backup)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as backup_exc:
                    backup_warning = f"Не вдалося створити резервну копію проєкту: {backup_exc}"
                    try:
                        os.remove(backup_temporary)
                    except OSError:
                        pass
            os.replace(temporary, self.project_path)
            self.project_label.setText(f"  {os.path.basename(self.project_path)}  ")
            self.setWindowTitle(f"{APP_TITLE} — {os.path.basename(self.project_path)}")
            if backup_warning:
                self.statusBar().showMessage(backup_warning, 15000)
        except OSError as exc:
            cleanup_errors = []
            for path in (temporary, backup_temporary):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_exc:
                    cleanup_errors.append(f"{path}: {cleanup_exc}")
            details = str(exc)
            if cleanup_errors:
                details += "\nНе вдалося прибрати тимчасові файли:\n" + "\n".join(cleanup_errors)
            self._error("Не вдалося зберегти проєкт", details)

    def open_video(self):
        start_dir = os.path.dirname(self.video_path) if self.video_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Вибрати відео", start_dir,
            "Відео (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.mpeg *.mpg *.wmv *.flv);;Усі файли (*.*)")
        if not path:
            return
        if not self.project_path:
            project, _ = QFileDialog.getSaveFileName(
                self, "Створити проєкт для цього відео",
                os.path.splitext(path)[0] + ".voxora", "Voxora Studio (*.voxora)")
            if not project:
                return
            if not project.lower().endswith(".voxora"):
                project += ".voxora"
            self.project_path = project
        self._stop_preview_players(remove_file=True)
        if self.background_path and (
            not self.background_source
            or os.path.normcase(os.path.abspath(self.background_source))
            != os.path.normcase(os.path.abspath(path))
        ):
            self.background_path = ""
            self.background_source = ""
        self._reset_seek_state()
        self.video_path = path
        self.playback_video_path = ""
        self.player.stop()
        self.player.setSource(QUrl())
        self.video_button.setText(os.path.basename(path))
        self._start_video_playback(path)
        self.save_project()

    def _start_video_playback(self, path: str):
        if self.job and self.job.isRunning():
            QMessageBox.information(
                self, APP_TITLE, "Дочекайтеся завершення поточної операції перед відкриттям відео.")
            return
        try:
            media_folder = self._project_media_dir()
        except RuntimeError as exc:
            self._error("Не вдалося підготувати відтворення", str(exc))
            return
        self._reset_seek_state()
        self.player.stop()
        self.player.setSource(QUrl())
        self.playback_video_path = ""
        self.statusBar().showMessage(
            "Перевіряю відео. Для HEVC без готової H.264-копії "
            "підготовка сумісного відео може тривати кілька хвилин…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._set_busy(True)
        job = VideoPlaybackJob(path, media_folder)
        self.job = job
        job.succeeded.connect(self._video_playback_ready)
        job.failed.connect(self._video_playback_failed)
        job.finished.connect(self._job_finished)
        job.start()

    def _video_playback_ready(self, original_path: str, playback_path: str):
        if os.path.normcase(os.path.abspath(original_path)) != os.path.normcase(
            os.path.abspath(self.video_path)
        ):
            return
        self.playback_video_path = playback_path
        self.player.setSource(QUrl.fromLocalFile(playback_path))
        if os.path.normcase(os.path.abspath(playback_path)) != os.path.normcase(
            os.path.abspath(original_path)
        ):
            self.statusBar().showMessage(
                "Для стабільного відтворення використовується сумісна копія H.264. "
                "Оригінал залишився без змін.", 12000)
        else:
            self.statusBar().showMessage("Відео готове до відтворення.", 5000)

    def _video_playback_failed(self, error: str):
        self.playback_video_path = ""
        self.player.setSource(QUrl())
        self._error("Не вдалося підготувати відео до відтворення", error)

    def _media_error(self, error, message: str):
        if error == QMediaPlayer.Error.NoError:
            return
        self.statusBar().showMessage(f"Помилка відео: {message}", 12000)
        QMessageBox.warning(
            self, "Відео не відтворюється",
            f"{message}\n\nДеякі кодеки (зокрема HEVC/H.265) можуть не підтримуватися Windows. "
            "Спробуйте встановити HEVC Video Extensions або конвертувати відео в H.264.")

    def _duration_changed(self, duration: int):
        if duration > 0:
            self.duration_ms = duration
            self.seek_slider.setRange(0, duration)
            self.timeline.set_data(self.clips, duration)

    def _position_changed(self, position: int):
        if not self.seek_slider.isSliderDown():
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(position)
            self.seek_slider.blockSignals(False)
        self.time_label.setText(f"{format_time(position)} / {format_time(self.duration_ms)}")
        self.timeline.playhead_ms = position
        self.timeline.update()
        if (self._seek_in_progress and self._seek_target_position is not None
                and abs(position - self._seek_target_position) <= 1000):
            self._finish_seek()

    def _preview_seek_position(self, position: int):
        self.time_label.setText(
            f"{format_time(position)} / {format_time(self.duration_ms)}")

    def _seek_from_slider(self):
        self._request_seek(self.seek_slider.value())

    def _request_seek(self, position: int):
        if not self._seek_in_progress and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._resume_after_seek = True
            self.player.pause()
        self._pending_seek_position = max(0, min(int(position), self.duration_ms))
        self._seek_timer.start()

    def _perform_pending_seek(self):
        position = self._pending_seek_position
        self._pending_seek_position = None
        if position is None or self.player.source().isEmpty():
            self._resume_after_seek = False
            return
        if not self.player.isSeekable():
            self.statusBar().showMessage("Це відео поки що не підтримує перемотування.", 5000)
            self._resume_after_seek = False
            return
        self._seek_in_progress = True
        self._seek_target_position = position
        self.player.setPosition(position)
        self._seek_deadline = time.monotonic() + 3.0
        self._seek_resume_timer.start()

    def _check_seek_completion(self):
        if not self._seek_in_progress:
            self._seek_resume_timer.stop()
            return
        if self._pending_seek_position is not None:
            self._seek_resume_timer.stop()
            return
        if (self._seek_target_position is not None
                and abs(self.player.position() - self._seek_target_position) <= 1000):
            self._finish_seek()
            return
        if time.monotonic() >= self._seek_deadline:
            self.statusBar().showMessage(
                "Перемотування не підтвердило точну позицію; відтворення продовжиться "
                "з найближчого доступного місця.", 8000)
            self._finish_seek()

    def _finish_seek(self):
        if self._pending_seek_position is not None:
            self._seek_in_progress = False
            self._seek_target_position = None
            self._seek_resume_timer.stop()
            self._seek_timer.start()
            return
        self._seek_resume_timer.stop()
        self._seek_in_progress = False
        self._seek_target_position = None
        if self._resume_after_seek:
            self._resume_after_seek = False
            if self.preview_active and self.preview_player:
                self.preview_player.setPosition(self.player.position())
                self.preview_player.play()
            self.player.play()

    def _reset_seek_state(self):
        self._seek_timer.stop()
        self._seek_resume_timer.stop()
        self._pending_seek_position = None
        self._seek_in_progress = False
        self._seek_target_position = None
        self._resume_after_seek = False
        self._seek_deadline = 0.0

    def _playback_changed(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("⏸ Пауза" if playing else "▶ Відтворити")
        if self.preview_active and self.preview_player:
            if playing:
                self.preview_player.play()
            else:
                self.preview_player.pause()

    def _original_volume_changed(self, value: int):
        if self.preview_active:
            self.audio_output.setVolume(0)
        else:
            self.audio_output.setVolume(value / 100)

    def _video_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia and self.preview_active:
            self._stop_preview_players()

    def _sync_preview_position(self):
        if (not self.preview_active or not self.preview_player
                or self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState):
            return
        difference = self.preview_player.position() - self.player.position()
        if abs(difference) > 450:
            self.preview_player.setPosition(self.player.position())

    def preview_result(self):
        if not self.video_path or not os.path.isfile(self.video_path):
            QMessageBox.information(self, APP_TITLE, "Спершу виберіть відео.")
            return
        if not self.clips:
            QMessageBox.information(self, APP_TITLE, "Додайте або згенеруйте голосові кліпи.")
            return
        try:
            folder = self._project_media_dir()
        except RuntimeError as exc:
            self._error("Проєкт не створено", str(exc))
            return
        handle = tempfile.NamedTemporaryFile(
            prefix="preview_", suffix=".wav", delete=False, dir=folder)
        preview_path = handle.name
        handle.close()
        clips = [AudioClip(**asdict(clip)) for clip in self.clips]
        background = self.background_path if (
            self.background_path and os.path.isfile(self.background_path)) else None
        background_gain = self.original_volume.value() / 100
        duration = self.duration_ms
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        job = BackgroundJob(
            lambda: self._render_mix(
                preview_path, background, clips, duration, background_gain))
        self.job = job
        job.succeeded.connect(lambda _result: self._preview_ready(preview_path, background is not None))
        job.failed.connect(lambda text: self._preview_failed(text, preview_path))
        job.finished.connect(self._job_finished)
        job.start()

    def _preview_failed(self, message: str, path: str):
        try:
            os.remove(path)
        except OSError:
            pass
        self._error("Не вдалося підготувати прослуховування", message)

    def _preview_ready(self, path: str, has_background_stem: bool):
        self._stop_preview_players(remove_file=True)
        self.preview_path = path
        self.preview_active = True
        self.preview_has_background = has_background_stem
        self.preview_player.setSource(QUrl.fromLocalFile(path))
        position = self.player.position()
        self.preview_player.setPosition(position)
        self.audio_output.setVolume(0)
        self.player.play()
        self.preview_player.play()
        self._preview_sync_timer.start()
        self.statusBar().showMessage(
            "Під час прослуховування оригінальна доріжка вимкнена. "
            + ("Додається відділена фонова доріжка." if has_background_stem
               else "Зараз чути лише нове озвучення; відділіть фон AI, щоб лишити ефекти."),
            10000)

    def _stop_preview_players(self, remove_file=False):
        self._preview_sync_timer.stop()
        self.preview_active = False
        self.preview_has_background = False
        if self.preview_player:
            self.preview_player.stop()
            self.preview_player.setSource(QUrl())
        self.audio_output.setVolume(self.original_volume.value() / 100)
        if remove_file and self.preview_path:
            try:
                os.remove(self.preview_path)
            except OSError:
                pass
            self.preview_path = ""

    def toggle_play(self):
        if self._seek_in_progress or self._pending_seek_position is not None:
            self._resume_after_seek = True
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif self.playback_video_path and os.path.isfile(self.playback_video_path):
            if self.preview_active and self.preview_player:
                self.preview_player.setPosition(self.player.position())
                self.preview_player.play()
            self.player.play()
        else:
            QMessageBox.information(self, APP_TITLE, "Спершу виберіть відео.")

    def import_audio(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Додати аудіокліпи", "", AUDIO_FILTER)
        if not paths:
            return
        try:
            folder = self._project_media_dir()
        except RuntimeError:
            QMessageBox.information(self, APP_TITLE, "Спершу відкрийте відео або створіть проєкт.")
            return
        added = 0
        for path in paths:
            target = os.path.join(folder, f"audio_{time.time_ns()}_{added}.wav")
            try:
                run_ffmpeg("-i", path, "-vn", "-ar", str(AUDIO_RATE), "-ac", "1",
                           "-c:a", "pcm_s16le", target)
                self.clips.append(AudioClip(target, os.path.basename(path),
                                            start_ms=self.player.position(), track=0))
                added += 1
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                QMessageBox.warning(self, "Помилка імпорту аудіо", f"{path}\n\n{exc}")
                try:
                    os.remove(target)
                except OSError:
                    pass
        if added:
            self._refresh()
            self.save_project()

    def import_dub_studio_voiceover(self):
        QMessageBox.information(
            self, "Імпорт озвучки з Dub Studio",
            "1. У Dub Studio виберіть потрібного спікера та згенеруйте його репліки.\n"
            "2. Експортуйте результат як WAV або MP3.\n"
            "3. Поверніться сюди й виберіть цей файл.\n\n"
            f"Озвучка буде вставлена на поточний таймкод відео ({format_time(self.player.position())}); "
            "потім її можна пересунути на таймлайні.")
        self.import_audio()

    def toggle_recording(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop_requested = True
            self.statusBar().showMessage("Завершую запис…")
            return
        try:
            folder = self._project_media_dir()
        except RuntimeError:
            QMessageBox.information(self, APP_TITLE, "Спершу відкрийте відео або створіть проєкт.")
            return
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            QMessageBox.warning(self, APP_TITLE, "Для запису встановіть пакет: python -m pip install sounddevice")
            return
        destination = os.path.join(folder, f"recording_{time.time_ns()}.wav")
        self._record_start_ms = self.player.position()
        self.recorder = AudioRecorder(destination)
        self.recorder.saved.connect(self._record_saved)
        self.recorder.failed.connect(lambda text: self._error("Помилка запису", text))
        self.recorder.start()
        self.statusBar().showMessage("Запис мікрофона триває — натисніть «● Запис» для зупинки.")

    def _record_saved(self, path: str):
        self.clips.append(AudioClip(path, "Запис мікрофона", self._record_start_ms, kind="recording"))
        self._refresh()
        self.save_project()
        self.statusBar().showMessage("Запис додано до проєкту", 5000)

    def preview_selected_clip(self):
        row = self.clip_table.currentRow()
        if not 0 <= row < len(self.clips):
            QMessageBox.information(self, APP_TITLE, "Виберіть аудіокліп у списку.")
            return
        clip = self.clips[row]
        try:
            if clip.muted:
                QMessageBox.information(self, APP_TITLE, "Цей кліп вимкнений.")
                return
            data = decode_audio(clip.path)
            start = max(0, round(clip.trim_start_ms * AUDIO_RATE / 1000))
            end = len(data) if not clip.trim_end_ms else min(
                len(data), round(clip.trim_end_ms * AUDIO_RATE / 1000))
            data = data[start:end] * clip.gain
            if not len(data):
                raise RuntimeError("У кліпі немає аудіо після обрізання.")
            self._stop_preview_players(remove_file=True)
            handle = tempfile.NamedTemporaryFile(
                prefix="voxora_clip_", suffix=".wav", delete=False)
            self.preview_path = handle.name
            handle.close()
            sf.write(self.preview_path, np.clip(data, -1, 1), AUDIO_RATE)
            self.preview_player.setSource(QUrl.fromLocalFile(self.preview_path))
            self.preview_player.play()
            self.statusBar().showMessage(f"Відтворюється кліп: {clip.name}", 5000)
        except Exception as exc:
            self._error("Не вдалося прослухати кліп", str(exc))

    @staticmethod
    def _stop_audio():
        try:
            import sounddevice as sd
            sd.stop()
        except (ImportError, RuntimeError):
            pass

    def import_subtitles(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Імпортувати субтитри", "", "Субтитри й текст (*.srt *.vtt *.txt);;Усі файли (*.*)")
        if not path:
            return
        try:
            cues = parse_subtitle_file(path)
            if not cues:
                raise ValueError("У файлі не знайдено реплік із часовими мітками або текстом.")
            self.cues = cues
            self._reconcile_cue_clip_links()
            self._refresh_cues()
            self.save_project()
        except (OSError, ValueError) as exc:
            self._error("Помилка субтитрів", str(exc))

    def add_cue(self):
        now = self.player.position()
        self.cues.append(SubtitleCue(now, now + 3000, "", ""))
        self._refresh_cues()
        self.cue_table.setCurrentCell(len(self.cues) - 1, 2)
        self.save_project()

    def delete_cue(self):
        rows = sorted(
            {item.row() for item in self.cue_table.selectedItems()}
            | ({self.cue_table.currentRow()} if self.cue_table.currentRow() >= 0 else set()),
            reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.cues):
                self.cues.pop(row)
        self._refresh_cues()
        self.save_project()

    def generate_speech(
        self, cue_indices: list[int] | None = None, replace_existing: bool = False,
    ):
        if self.job and self.job.isRunning():
            QMessageBox.information(self, APP_TITLE, "Озвучення вже виконується.")
            return
        if not self.cues:
            QMessageBox.information(self, APP_TITLE, "Спершу імпортуйте субтитри або додайте репліки.")
            return
        selected_indices = (
            set(cue_indices) if cue_indices is not None else set(range(len(self.cues)))
        )
        cues_to_speak = [
            (index, cue) for index, cue in enumerate(self.cues)
            if index in selected_indices and (cue.text.strip() or cue.dub_text.strip())
        ]
        if not cues_to_speak:
            QMessageBox.information(self, APP_TITLE, "У субтитрах немає тексту для озвучення.")
            return
        backend = self.tts_backend_combo.currentData() or "edge"
        profiles = {
            profile["character"]: dict(profile)
            for profile in self.voice_profiles
        }
        if backend in {"qwen3", "chatterbox"}:
            try:
                if backend == "qwen3":
                    local_tts_configuration()
                else:
                    chatterbox_tts_configuration()
            except RuntimeError as exc:
                self._error("Локальний синтез недоступний", str(exc))
                return
            if self.translate_check.isChecked() and any(
                not cue.dub_text.strip() for _, cue in cues_to_speak
            ):
                QMessageBox.warning(
                    self, APP_TITLE,
                    "Локальна модель підтримує російську, але не українську. "
                    "Вимкніть переклад на українську або введіть російський текст у колонку "
                    "«Текст озвучення». Переклад через Google надсилав би текст в інтернет.")
                return
            if backend == "chatterbox" and any(
                re.search(r"[іїєґІЇЄҐ]", cue.dub_text.strip() or cue.text)
                for _, cue in cues_to_speak
            ):
                QMessageBox.warning(
                    self, APP_TITLE,
                    "Chatterbox у цій програмі озвучує російською. В одному з текстів є "
                    "українські літери; відредагуйте його російською або виберіть Edge TTS.")
                return
            missing = set()
            for _, cue in cues_to_speak:
                profile = profiles.get(cue.character) or profiles.get("За замовчуванням", {})
                reference = profile.get("reference", "")
                if (
                    not os.path.isfile(reference)
                    or (backend == "qwen3" and profile.get("engine") != "qwen3")
                ):
                    missing.add(cue.character or "За замовчуванням")
            missing = sorted(missing)
            if missing:
                QMessageBox.warning(
                    self, APP_TITLE,
                    "Додайте локальний WAV/MP3-зразок для персонажів: "
                    + ", ".join(missing)
                    + ". У «Персонажі та голоси» залиште ім’я порожнім, щоб задати голос "
                    "за замовчуванням для реплік без персонажа.")
                return
            if backend == "chatterbox":
                answer = QMessageBox.question(
                    self, "Підтвердження права на зразки голосу",
                    "Підтвердіть, що ви маєте право використовувати кожен вибраний зразок "
                    "для клонування голосу. Обробка відбуватиметься локально на цьому "
                    "комп’ютері; зразки не надсилатимуться в інтернет.\n\n"
                    "Щоб у кожній короткій репліці не повторювалося словесне оголошення, "
                    "підтвердіть також, що позначатимете кінцеве відео як озвучене ШІ. "
                    "Вбудоване маркування синтетичного аудіо не вимикається.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    return
        try:
            if backend == "edge":
                import edge_tts  # noqa: F401
            if self.translate_check.isChecked():
                import deep_translator  # noqa: F401
            folder = self._project_media_dir()
        except ImportError as exc:
            QMessageBox.warning(self, APP_TITLE, f"Не встановлено пакет {exc.name}.\n"
                                  "Виконайте: python -m pip install edge-tts deep-translator")
            return
        except RuntimeError as exc:
            self._error("Проєкт не створено", str(exc))
            return
        if not replace_existing and cue_indices is None and any(
            clip.kind == "tts" for clip in self.clips
        ):
            answer = QMessageBox.question(
                self, "Повторна генерація",
                "У проєкті вже є згенеровані голоси. Видалити їх і створити озвучення за поточними "
                "персонажами/голосами?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.clips = [clip for clip in self.clips if clip.kind != "tts"]
            self._refresh_clips()
        self._replacement_cue_ids = (
            {cue.cue_id for _, cue in cues_to_speak} if replace_existing else set())
        self.progress.setVisible(True)
        self.progress.setRange(0, len(cues_to_speak))
        self.progress.setValue(0)
        self._set_busy(True)
        self.job = SpeechGenerator(
            [(index, SubtitleCue(**asdict(cue))) for index, cue in cues_to_speak],
            [SubtitleCue(**asdict(cue)) for cue in self.cues],
            profiles, self.voice_combo.currentText(), self.translate_check.isChecked(),
            folder, backend)
        self.job.clip_ready.connect(self._speech_clip_ready)
        self.job.progress.connect(lambda current, total: self.progress.setValue(current))
        self.job.warning.connect(lambda text: self.statusBar().showMessage(text, 15000))
        self.job.status.connect(lambda text: self.statusBar().showMessage(text))
        self.job.failed.connect(lambda text: self._error("Помилка синтезу голосу", text))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def _selected_cue_row(self) -> int:
        row = self.cue_table.currentRow()
        if 0 <= row < len(self.cues):
            return row
        selected = self.cue_table.selectionModel().selectedRows()
        return selected[0].row() if selected else -1

    def _find_cue_clip(self, cue_index: int) -> int:
        if not 0 <= cue_index < len(self.cues):
            return -1
        cue = self.cues[cue_index]
        for index, clip in enumerate(self.clips):
            if clip.kind == "tts" and clip.cue_id == cue.cue_id:
                return index
        for index, clip in enumerate(self.clips):
            if clip.kind == "tts" and clip.cue_id == "" and clip.cue_index == cue_index:
                clip.cue_id = cue.cue_id
                return index
        candidates = [
            index for index, clip in enumerate(self.clips)
            if clip.kind == "tts" and not clip.cue_id
            and clip_match_key(clip.name) in {
                subtitle_match_key(cue.text), subtitle_match_key(cue.dub_text),
            }
        ]
        if candidates:
            match = min(
                candidates, key=lambda index: abs(self.clips[index].start_ms - cue.start_ms))
            self.clips[match].cue_id = cue.cue_id
            self.clips[match].cue_index = cue_index
            return match
        return -1

    def _regenerate_selected_cue(self):
        row = self._selected_cue_row()
        if row < 0:
            QMessageBox.information(self, APP_TITLE, "Спершу виберіть репліку.")
            return
        self._regenerate_cue(row)

    def _regenerate_cue(self, row: int):
        if not 0 <= row < len(self.cues):
            return
        if self._find_cue_clip(row) < 0:
            self.statusBar().showMessage(
                "Текст збережено. Для цієї репліки ще немає аудіо-кліпу.", 7000)
            return
        self.generate_speech([row], replace_existing=True)

    def manage_voices(self):
        if self.tts_backend_combo.currentData() in {"qwen3", "chatterbox"}:
            self._show_voice_manager()
            return
        if self.voice_catalog:
            self._show_voice_manager()
            return
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            QMessageBox.warning(self, APP_TITLE, "Для каталогу голосів встановіть: python -m pip install edge-tts")
            return
        if self.voice_catalog_job and self.voice_catalog_job.isRunning():
            return
        self.statusBar().showMessage("Завантажую каталог голосів Edge TTS…")
        self.voice_catalog_job = VoiceCatalogJob(self)
        self.voice_catalog_job.succeeded.connect(self._voice_catalog_loaded)
        self.voice_catalog_job.failed.connect(
            lambda error: self._error("Не вдалося отримати каталог голосів", error))
        self.voice_catalog_job.start()

    def _voice_catalog_loaded(self, voices):
        self.voice_catalog = voices
        self.statusBar().showMessage(f"Голосів доступно: {len(voices)}", 5000)
        self._show_voice_manager()

    def _show_voice_manager(self):
        dialog = VoiceManagerDialog(
            self, self.voice_catalog, self.voice_profiles,
            self.local_voice_library, self.preview_voice)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            old_paths = {os.path.abspath(entry["path"]) for entry in self.local_voice_library}
            new_paths = {os.path.abspath(entry["path"]) for entry in dialog.local_voices}
            removed_paths = old_paths - new_paths
            removed_paths.update(
                os.path.abspath(path) for path in dialog.imported_voice_paths
                if os.path.abspath(path) not in new_paths
            )
            try:
                save_voice_library(dialog.local_voices)
            except RuntimeError as exc:
                QMessageBox.critical(self, APP_TITLE, str(exc))
                cleanup_errors = dialog.discard_imported_voices()
                if cleanup_errors:
                    QMessageBox.warning(
                        self, APP_TITLE,
                        "Не вдалося прибрати тимчасові копії:\n" + "\n".join(cleanup_errors))
                return
            self.local_voice_library = dialog.local_voices
            self.voice_profiles = dialog.profiles
            self._refresh_cues()
            self.save_project()
            delete_errors = []
            for path in removed_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    delete_errors.append(f"{path}: {exc}")
            if delete_errors:
                QMessageBox.warning(
                    self, APP_TITLE,
                    "Не вдалося видалити деякі файли голосів:\n" + "\n".join(delete_errors))
        else:
            cleanup_errors = dialog.discard_imported_voices()
            if cleanup_errors:
                QMessageBox.warning(
                    self, APP_TITLE,
                    "Не вдалося прибрати тимчасові копії:\n" + "\n".join(cleanup_errors))

    def preview_voice(self, voice: str | dict):
        if self.voice_sample_job and self.voice_sample_job.isRunning():
            QMessageBox.information(self, APP_TITLE, "Зараз уже готується проба голосу.")
            return
        self.voice_sample_job = VoiceSampleJob(voice)
        self.voice_sample_job.succeeded.connect(self._play_voice_sample)
        self.voice_sample_job.failed.connect(
            lambda text: self._error("Не вдалося прослухати голос", text))
        self.voice_sample_job.start()

    def _preview_media_error(self, error, message: str):
        if error != QMediaPlayer.Error.NoError:
            self.statusBar().showMessage(
                "Не вдалося відтворити аудіо: " + (message or error.name), 10000)

    def _play_voice_sample(self, path: str):
        self._stop_preview_players(remove_file=True)
        self.preview_path = path
        self.preview_active = False
        self.preview_player.setSource(QUrl.fromLocalFile(path))
        self.preview_player.play()

    def _sample_media_status_changed(self, status):
        if (status == QMediaPlayer.MediaStatus.EndOfMedia
                and not self.preview_active and self.preview_path):
            self.preview_player.stop()
            self.preview_player.setSource(QUrl())
            try:
                os.remove(self.preview_path)
            except OSError:
                pass
            self.preview_path = ""
            self.statusBar().showMessage("Прослуховування завершено.", 3000)

    def assign_characters_with_ollama(self):
        if not self.cues:
            QMessageBox.information(self, APP_TITLE, "Спершу імпортуйте субтитри.")
            return
        if self.job and self.job.isRunning():
            QMessageBox.information(self, APP_TITLE, "Дочекайтеся завершення поточної операції.")
            return
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def list_models():
            request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(request, timeout=8) as response:
                models = json.load(response).get("models", [])
            return [model["name"] for model in models if model.get("name")]

        self.job = BackgroundJob(list_models)
        self.job.succeeded.connect(self._ollama_models_loaded)
        self.job.failed.connect(lambda text: self._error("Ollama недоступна", text))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def accent_selected_cue(self):
        row = self._selected_cue_row()
        if row < 0:
            QMessageBox.information(self, APP_TITLE, "Спершу виберіть репліку.")
            return
        if self.job and self.job.isRunning():
            QMessageBox.information(self, APP_TITLE, "Дочекайтеся завершення поточної операції.")
            return
        text = (self.cues[row].dub_text or self.cues[row].text).strip()
        if not text:
            QMessageBox.information(self, APP_TITLE, "У вибраній репліці немає тексту.")
            return
        self._accent_pending_row = row
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        def list_models():
            request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(request, timeout=8) as response:
                models = json.load(response).get("models", [])
            return [model["name"] for model in models if model.get("name")]

        self.job = BackgroundJob(list_models)
        self.job.succeeded.connect(self._accent_models_loaded)
        self.job.failed.connect(lambda error: self._error("Ollama недоступна", error))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def assist_selected_cues(self):
        if self.job and self.job.isRunning():
            QMessageBox.information(self, APP_TITLE, "Дочекайтеся завершення поточної операції.")
            return
        rows = {index.row() for index in self.cue_table.selectionModel().selectedRows()}
        if not rows and 0 <= self.cue_table.currentRow() < len(self.cues):
            rows.add(self.cue_table.currentRow())
        rows = {row for row in rows if 0 <= row < len(self.cues)}
        if not rows:
            QMessageBox.information(
                self, APP_TITLE, "Виберіть одну або кілька реплік у таблиці субтитрів.")
            return
        actions = [
            "Перекласти українською природно для дубляжу",
            "Покращити паузи й інтонацію (слова не змінювати)",
            "Розставити наголоси",
        ]
        selected_action, accepted = QInputDialog.getItem(
            self, "AI-помічник", "Що зробити з вибраними репліками?", actions, 0, False)
        if not accepted:
            return
        action = {
            actions[0]: "translate",
            actions[1]: "intonation",
            actions[2]: "stress",
        }[selected_action]
        try:
            configuration = self._ai_assistant_configuration()
        except RuntimeError as exc:
            QMessageBox.warning(self, "AI-помічник не налаштований", str(exc))
            self.configure_ai_assistant()
            return
        if configuration["provider"] == "deepseek":
            answer = QMessageBox.question(
                self, "Надіслати текст у DeepSeek?",
                f"DeepSeek API отримає текст {len(rows)} вибраних реплік. "
                "Аудіо та відео не надсилаються. Продовжити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        cue_text = [
            (row, (self.cues[row].dub_text or self.cues[row].text).strip())
            for row in sorted(rows)
        ]
        cue_text = [(row, text) for row, text in cue_text if text]
        if not cue_text:
            QMessageBox.information(self, APP_TITLE, "У вибраних репліках немає тексту.")
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, len(cue_text))
        self.progress.setValue(0)
        self._set_busy(True)
        self.job = TextAssistJob(configuration, action, cue_text)
        self.job.progress.connect(lambda current, total: self.progress.setValue(current))
        self.job.succeeded.connect(lambda result: self._ai_assist_ready(result, action))
        self.job.failed.connect(lambda error: self._error("AI-помічник", error))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def _ai_assist_ready(self, suggestions, action: str):
        items = []
        for cue_index, suggested in suggestions:
            if not 0 <= cue_index < len(self.cues):
                continue
            cue = self.cues[cue_index]
            items.append({
                "cue_index": cue_index,
                "original": (cue.dub_text or cue.text).strip(),
                "text": suggested,
            })
        if not items:
            QMessageBox.warning(self, APP_TITLE, "AI не повернув пропозицій для вибраних реплік.")
            return
        review = TextAssistReviewDialog(self, items, action)
        if review.exec() != QDialog.DialogCode.Accepted:
            return
        changes = review.selected_results()
        if not changes:
            return
        for cue_index, text in changes:
            self.cues[cue_index].dub_text = text
        self._refresh_cues()
        self.save_project()
        self.statusBar().showMessage(
            f"Застосовано пропозиції до реплік: {len(changes)}. Голосові зразки не змінені.", 8000)
        existing_rows = [
            cue_index for cue_index, _ in changes if self._find_cue_clip(cue_index) >= 0
        ]
        if existing_rows:
            answer = QMessageBox.question(
                self, "Оновити озвучення?",
                f"Перегенерувати звук для {len(existing_rows)} змінених реплік? "
                "Призначений зразок голосу залишиться тим самим.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if answer == QMessageBox.StandardButton.Yes:
                self.generate_speech(existing_rows, replace_existing=True)

    def _accent_models_loaded(self, models: list[str]):
        if not models:
            QMessageBox.warning(
                self, APP_TITLE, "В Ollama немає встановлених моделей. Перевірте команду `ollama list`.")
            return
        row = self._accent_pending_row
        if row is None or not 0 <= row < len(self.cues):
            return
        model, accepted = QInputDialog.getItem(
            self, "Наголоси через Ollama",
            "Локальна модель працюватиме лише з текстом вибраної репліки:",
            models, 0, False)
        if not accepted:
            return
        text = (self.cues[row].dub_text or self.cues[row].text).strip()
        self.job = AccentJob(model, row, text)
        self.job.succeeded.connect(self._accent_ready)
        self.job.failed.connect(lambda error: self._error("Не вдалося розставити наголоси", error))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def _accent_ready(self, row: int, accented_text: str):
        if not 0 <= row < len(self.cues):
            return
        original = (self.cues[row].dub_text or self.cues[row].text).strip()
        if accented_text.replace("\u0301", "") != original.replace("\u0301", ""):
            self._error(
                "Текст Ollama змінено",
                "Модель змінила символи, окрім позначок наголосу. Правку не застосовано.")
            return
        self.cues[row].dub_text = accented_text
        self._refresh_cues()
        self.save_project()
        self.statusBar().showMessage(
            "Наголоси додано. Перевірте текст; оновлюю аудіо лише цієї репліки.", 8000)
        QTimer.singleShot(100, lambda cue_row=row: self._regenerate_cue(cue_row))

    def _ollama_models_loaded(self, models: list[str]):
        if not models:
            QMessageBox.warning(
                self, APP_TITLE, "В Ollama немає встановлених моделей. Перевірте команду `ollama list`.")
            return
        model, accepted = QInputDialog.getItem(
            self, "Аналіз субтитрів через Ollama",
            "Модель аналізуватиме текст субтитрів, не звукову доріжку:", models, 0, False)
        if not accepted:
            return
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.job = OllamaJob(model, [SubtitleCue(**asdict(cue)) for cue in self.cues])
        self.job.succeeded.connect(self._ollama_assignments_ready)
        self.job.failed.connect(lambda text: self._error("Не вдалося проаналізувати субтитри", text))
        self.job.finished.connect(self._job_finished)
        self.job.start()

    def _ollama_assignments_ready(self, assignments):
        default_voice = self.voice_combo.currentText()
        changed = 0
        for cue_id, character in assignments:
            if 0 <= cue_id < len(self.cues):
                self.cues[cue_id].character = character
                changed += 1
                if not any(profile["character"] == character for profile in self.voice_profiles):
                    self.voice_profiles.append({"character": character, "voice": default_voice})
        self._refresh_cues()
        self.save_project()
        QMessageBox.information(
            self, "Аналіз Ollama завершено",
            f"Призначено персонажів для реплік: {changed}.\n"
            "Це припущення за змістом субтитрів, а не розпізнавання голосу. "
            "Перевірте результат і призначте голоси персонажам.")

    def separate_original_dialogue(self):
        if not self.video_path or not os.path.isfile(self.video_path):
            QMessageBox.information(self, APP_TITLE, "Спершу відкрийте відео.")
            return
        if self.background_path and os.path.isfile(self.background_path):
            answer = QMessageBox.question(
                self, "Фонова доріжка вже створена",
                "Для цього проєкту вже є AI-доріжка без мовлення. Створити її заново?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            folder = self._project_media_dir()
        except RuntimeError as exc:
            self._error("Проєкт не створено", str(exc))
            return
        answer = QMessageBox.warning(
            self, "AI-розділення аудіо",
            "Demucs відділить вокал/мовлення від фонової доріжки. Фоновий stem зберігає "
            "музику й ефекти, але AI може залишити артефакти або частину голосу. "
            "Під час першого запуску модель завантажиться з інтернету; обробка може тривати довго.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Ok:
            return
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        video_path = self.video_path
        job = BackgroundJob(
            lambda: separate_dialogue_from_video(video_path, folder))
        self.job = job
        job.succeeded.connect(self._dialogue_separated)
        job.failed.connect(lambda text: self._error("Не вдалося відділити мовлення", text))
        job.finished.connect(self._job_finished)
        job.start()

    def _dialogue_separated(self, path):
        self.background_path = path
        self.background_source = os.path.abspath(self.video_path)
        self.save_project()
        self.statusBar().showMessage(
            "Створено фонову доріжку без основного вокалу/мовлення.", 10000)

    def _speech_clip_ready(self, clip: AudioClip):
        existing_index = (
            next((index for index, existing in enumerate(self.clips)
                  if existing.kind == "tts" and existing.cue_id == clip.cue_id), -1)
            if clip.cue_id and clip.cue_id in self._replacement_cue_ids
            else -1
        )
        if existing_index >= 0:
            existing = self.clips[existing_index]
            clip.track = existing.track
            clip.gain = existing.gain
            clip.muted = existing.muted
            clip.trim_start_ms = existing.trim_start_ms
            clip.trim_end_ms = existing.trim_end_ms
            self.clips[existing_index] = clip
        else:
            self.clips.append(clip)
        self._refresh_clips()
        self.timeline.set_data(self.clips, self.duration_ms)
        self.save_project()
        self.statusBar().showMessage(
            ("Оновлено лише вибрану репліку: " if existing_index >= 0
             else "Створено голосовий кліп: ") + clip.name, 5000)

    def _job_finished(self):
        if self.sender() is not self.job:
            return
        self._replacement_cue_ids.clear()
        self._set_busy(False)
        self.progress.setVisible(False)
        self.save_project()

    def _set_busy(self, busy: bool):
        self.findChild(QToolBar).setEnabled(not busy)
        self.centralWidget().setEnabled(not busy)

    def export_video(self):
        if not self.video_path or not os.path.isfile(self.video_path):
            QMessageBox.information(self, APP_TITLE, "Спершу виберіть доступне відео.")
            return
        if not self.clips:
            QMessageBox.information(self, APP_TITLE, "Додайте аудіокліпи перед експортом.")
            return
        output, _ = QFileDialog.getSaveFileName(
            self, "Експортувати дубляж", os.path.splitext(self.video_path)[0] + "_voxora.mp4",
            "MP4 (*.mp4)")
        if not output:
            return
        if not output.lower().endswith(".mp4"):
            output += ".mp4"
        background = self.background_path if (
            self.background_path and os.path.isfile(self.background_path)) else None
        keep_original_audio = False
        if not background:
            choice = QMessageBox(self)
            choice.setWindowTitle("Оригінальне мовлення")
            choice.setIcon(QMessageBox.Icon.Warning)
            choice.setText("Для цього відео ще немає AI-доріжки без мовлення.")
            choice.setInformativeText(
                "Щоб прибрати голоси, але зберегти музику та ефекти, спершу потрібне AI-розділення. "
                "Без нього можна або вимкнути весь оригінальний звук, або залишити його разом із мовленням.")
            ai_button = choice.addButton(
                "Спершу AI-розділення (рекомендовано)", QMessageBox.ButtonRole.AcceptRole)
            mute_button = choice.addButton(
                "Експортувати без усього оригінального звуку", QMessageBox.ButtonRole.DestructiveRole)
            keep_button = choice.addButton(
                "Залишити оригінал (голоси будуть чути)", QMessageBox.ButtonRole.ActionRole)
            choice.addButton("Скасувати", QMessageBox.ButtonRole.RejectRole)
            choice.setDefaultButton(ai_button)
            choice.exec()
            selected = choice.clickedButton()
            if selected is ai_button:
                self.separate_original_dialogue()
                return
            if selected is keep_button:
                keep_original_audio = True
            elif selected is not mute_button:
                return
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        background_volume = self.original_volume.value() / 100
        job = BackgroundJob(lambda: self._export_worker(
            output, background_volume, background, keep_original_audio))
        self.job = job
        job.succeeded.connect(lambda path: self._export_done(path))
        job.failed.connect(lambda text: self._error("Помилка експорту", text))
        job.finished.connect(self._job_finished)
        job.start()

    def _render_mix(
        self, destination: str, bed_path: str | None = None,
        clips_snapshot: list[AudioClip] | None = None,
        duration_snapshot: int | None = None,
        background_gain: float = 1.0,
    ):
        block_size = 65536
        duration_ms = duration_snapshot if duration_snapshot is not None else self.duration_ms
        render_clips = list(clips_snapshot if clips_snapshot is not None else self.clips)
        if bed_path:
            render_clips.insert(0, AudioClip(
                bed_path, "Оригінальні ефекти та музика",
                gain=background_gain, kind="background"))
        output_frames = max(1, round(duration_ms * AUDIO_RATE / 1000))
        handles: list[tuple[AudioClip, sf.SoundFile, int, int]] = []
        try:
            for clip in render_clips:
                if clip.muted or not os.path.isfile(clip.path):
                    continue
                handle = sf.SoundFile(clip.path, mode="r")
                source_start = max(0, round(clip.trim_start_ms * handle.samplerate / 1000))
                source_end = (min(len(handle), round(clip.trim_end_ms * handle.samplerate / 1000))
                              if clip.trim_end_ms else len(handle))
                if source_end <= source_start:
                    handle.close()
                    continue
                clip_start = round(clip.start_ms * AUDIO_RATE / 1000)
                clip_frames = round((source_end - source_start) * AUDIO_RATE / handle.samplerate)
                handles.append((clip, handle, source_start, min(clip_start + clip_frames, output_frames)))
            with sf.SoundFile(destination, "w", AUDIO_RATE, 1, subtype="PCM_16") as out:
                for block_start in range(0, output_frames, block_size):
                    frames = min(block_size, output_frames - block_start)
                    mixed = np.zeros(frames, dtype=np.float32)
                    block_end = block_start + frames
                    for clip, handle, source_start, clip_end in handles:
                        clip_start = round(clip.start_ms * AUDIO_RATE / 1000)
                        overlap_start = max(block_start, clip_start)
                        overlap_end = min(block_end, clip_end)
                        if overlap_end <= overlap_start:
                            continue
                        source_offset = round(source_start + (overlap_start - clip_start) * handle.samplerate / AUDIO_RATE)
                        source_count = round((overlap_end - overlap_start) * handle.samplerate / AUDIO_RATE)
                        handle.seek(source_offset)
                        samples = handle.read(source_count, dtype="float32", always_2d=True).mean(axis=1)
                        if handle.samplerate != AUDIO_RATE and len(samples):
                            desired = overlap_end - overlap_start
                            samples = np.interp(
                                np.linspace(0, max(0, len(samples) - 1), desired),
                                np.arange(len(samples)), samples).astype(np.float32)
                        target_offset = overlap_start - block_start
                        count = min(len(samples), overlap_end - overlap_start)
                        if count:
                            mixed[target_offset:target_offset + count] += samples[:count] * clip.gain
                    out.write(np.clip(mixed, -1, 1))
        finally:
            for _, handle, _, _ in handles:
                handle.close()

    def _export_worker(
        self, output: str, background_volume: float = 0.25,
        background_path: str | None = None,
        keep_original_audio: bool = True,
    ) -> str:
        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        mix_path = temporary.name
        temporary.close()
        try:
            bed_path = background_path or self.background_path
            if not bed_path or not os.path.isfile(bed_path):
                bed_path = None
            self._render_mix(mix_path, bed_path, background_gain=background_volume)
            ffprobe = shutil.which("ffprobe")
            has_audio = True
            if ffprobe:
                result = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                     "stream=index", "-of", "csv=p=0", self.video_path],
                    capture_output=True, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=60)
                has_audio = bool(result.stdout.strip())
            if has_audio and not bed_path and keep_original_audio:
                args = ["-i", self.video_path, "-i", mix_path, "-filter_complex",
                        f"[0:a:0]volume={background_volume:.3f}[bg];"
                        "[bg][1:a:0]amix=inputs=2:duration=first:normalize=0[a]",
                        "-map", "0:v:0", "-map", "[a]"]
            else:
                args = ["-i", self.video_path, "-i", mix_path, "-map", "0:v:0", "-map", "1:a:0"]
            run_ffmpeg(*args, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                       "-movflags", "+faststart", output)
            self._write_srt(os.path.splitext(output)[0] + ".srt")
            return output
        finally:
            try:
                os.remove(mix_path)
            except OSError:
                pass

    def _write_srt(self, path: str):
        def timestamp(value):
            value = max(0, int(value))
            hours, rem = divmod(value, 3_600_000)
            minutes, rem = divmod(rem, 60_000)
            seconds, millis = divmod(rem, 1000)
            return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8-sig") as stream:
            for index, cue in enumerate(self.cues, 1):
                text = cue.dub_text.strip() or cue.text
                stream.write(f"{index}\n{timestamp(cue.start_ms)} --> {timestamp(cue.end_ms)}\n{text}\n\n")
        os.replace(temporary, path)

    def _export_done(self, path: str):
        QMessageBox.information(
            self, "Експорт готовий",
            f"Відео:\n{path}\n\nСубтитри:\n{os.path.splitext(path)[0]}.srt")

    def _refresh(self):
        self._refresh_clips()
        self._refresh_cues()
        self.timeline.set_data(self.clips, self.duration_ms)
        self.save_project()

    def _refresh_clips(self):
        self.clip_table.blockSignals(True)
        self.clip_table.setRowCount(len(self.clips))
        for row, clip in enumerate(self.clips):
            values = [
                format_time(clip.start_ms), str(clip.track + 1), clip.name,
                f"{clip.gain:.2f}", str(clip.trim_start_ms), str(clip.trim_end_ms),
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if col == 2:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if clip.muted:
                    cell.setForeground(QColor("#8993a1"))
                self.clip_table.setItem(row, col, cell)
            muted_cell = QTableWidgetItem()
            muted_cell.setFlags(
                muted_cell.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled)
            muted_cell.setCheckState(
                Qt.CheckState.Checked if clip.muted else Qt.CheckState.Unchecked)
            self.clip_table.setItem(row, 6, muted_cell)
        self.clip_table.blockSignals(False)
        self.timeline.set_data(self.clips, self.duration_ms)

    def _clip_item_changed(self, item: QTableWidgetItem):
        row, column = item.row(), item.column()
        if not 0 <= row < len(self.clips):
            return
        clip = self.clips[row]
        try:
            if column == 0:
                clip.start_ms = max(0, parse_time(item.text()))
            elif column == 1:
                track = int(item.text()) - 1
                if not 0 <= track < TRACKS:
                    raise ValueError(f"Доріжка має бути від 1 до {TRACKS}.")
                clip.track = track
            elif column == 3:
                gain = float(item.text().replace(",", "."))
                if not 0 <= gain <= 4:
                    raise ValueError("Гучність кліпу має бути в межах 0–4.")
                clip.gain = gain
            elif column in (4, 5):
                value = int(item.text())
                if value < 0:
                    raise ValueError("Час обрізання не може бути від’ємним.")
                if column == 4:
                    clip.trim_start_ms = value
                else:
                    clip.trim_end_ms = value
                if clip.trim_end_ms and clip.trim_end_ms <= clip.trim_start_ms:
                    raise ValueError("Кінець обрізання має бути пізніше за початок.")
            elif column == 6:
                clip.muted = item.checkState() == Qt.CheckState.Checked
            if column in (4, 5):
                self.timeline.set_data(self.clips, self.duration_ms)
            else:
                self.timeline.update()
            self.save_project()
        except ValueError as exc:
            self._error("Некоректні значення кліпу", str(exc))
            self._refresh_clips()

    def _refresh_cues(self):
        self.cue_table.blockSignals(True)
        self.cue_table.setRowCount(len(self.cues))
        for row, cue in enumerate(self.cues):
            for column, value in enumerate((
                format_time(cue.start_ms), format_time(cue.end_ms), "", cue.text, cue.dub_text,
            )):
                self.cue_table.setItem(row, column, QTableWidgetItem(value))
            character_box = QComboBox()
            character_box.setEditable(True)
            character_box.addItem("")
            character_box.addItems(
                profile["character"] for profile in self.voice_profiles
                if profile["character"])
            character_box.setCurrentText(cue.character)
            character_box.currentTextChanged.connect(
                lambda value, cue_row=row: self._cue_character_changed(cue_row, value))
            self.cue_table.setCellWidget(row, 2, character_box)
            expression = QDoubleSpinBox()
            expression.setRange(0.0, 2.0)
            expression.setDecimals(1)
            expression.setSingleStep(0.1)
            expression.setValue(float(cue.exaggeration))
            expression.setEnabled(self.tts_backend_combo.currentData() == "chatterbox")
            expression.setToolTip(
                "Інтенсивність виразності Chatterbox (0–2; стандартно 0,5). "
                "Це не вибір конкретної емоції.")
            expression.editingFinished.connect(
                lambda cue_row=row, widget=expression:
                    self._cue_exaggeration_changed(cue_row, widget))
            self.cue_table.setCellWidget(row, 5, expression)
        self.cue_table.blockSignals(False)

    def _cue_exaggeration_changed(self, row: int, control: QDoubleSpinBox):
        if not 0 <= row < len(self.cues):
            return
        value = round(control.value(), 1)
        cue = self.cues[row]
        if cue.exaggeration == value:
            return
        cue.exaggeration = value
        self.save_project()
        if (
            self.tts_backend_combo.currentData() == "chatterbox"
            and self._find_cue_clip(row) >= 0
        ):
            QTimer.singleShot(0, lambda cue_row=row: self._regenerate_cue(cue_row))

    def _cue_character_changed(self, row: int, character: str):
        if 0 <= row < len(self.cues):
            self.cues[row].character = character.strip()
            self.save_project()
            if self._find_cue_clip(row) >= 0:
                QTimer.singleShot(0, lambda cue_row=row: self._regenerate_cue(cue_row))

    def _cue_item_changed(self, item: QTableWidgetItem):
        row, column = item.row(), item.column()
        if not 0 <= row < len(self.cues):
            return
        cue = self.cues[row]
        start_clip_index = self._find_cue_clip(row) if item.column() == 0 else -1
        try:
            if column == 0:
                cue.start_ms = parse_time(item.text())
            elif column == 1:
                cue.end_ms = parse_time(item.text())
            elif column == 3:
                cue.text = item.text()
            else:
                cue.dub_text = item.text()
            if cue.end_ms <= cue.start_ms:
                raise ValueError("Час завершення репліки має бути пізніше за початок.")
            if column == 0:
                if start_clip_index >= 0:
                    self.clips[start_clip_index].start_ms = cue.start_ms
                    self._refresh_clips()
                    self.timeline.set_data(self.clips, self.duration_ms)
            self.save_project()
            if column == 4 or (column == 3 and not cue.dub_text.strip()):
                QTimer.singleShot(0, lambda cue_row=row: self._regenerate_cue(cue_row))
        except ValueError as exc:
            self._error("Некоректні субтитри", str(exc))
            self._refresh_cues()

    def _move_clip(self, index: int, start_ms: int, track: int):
        if not 0 <= index < len(self.clips):
            return
        clip = self.clips[index]
        clip.start_ms = start_ms
        clip.track = track
        self.timeline.update()
        self._refresh_clips()
        self.save_project()

    def _seek_to_clip_row(self, row: int, column: int):
        if 0 <= row < len(self.clips):
            self._request_seek(self.clips[row].start_ms)

    def delete_selected_clips(self):
        rows = sorted({item.row() for item in self.clip_table.selectedItems()}, reverse=True)
        if not rows:
            return
        managed_paths = []
        if self.project_path:
            media_root = os.path.abspath(os.path.join(os.path.dirname(self.project_path), "voxora_media"))
            for row in rows:
                path = os.path.abspath(self.clips[row].path)
                try:
                    if os.path.commonpath((media_root, path)) == media_root:
                        managed_paths.append(path)
                except ValueError:
                    pass
        delete_files = False
        if managed_paths:
            answer = QMessageBox.question(
                self, "Видалити аудіокліпи",
                f"Видалити кліпів: {len(rows)}. Також стерти їхні аудіофайли з папки проєкту?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            delete_files = answer == QMessageBox.StandardButton.Yes
        for row in rows:
            self.clips.pop(row)
        if delete_files:
            for path in managed_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    self.statusBar().showMessage(f"Не вдалося видалити {path}: {exc}", 8000)
        self._refresh_clips()
        self.save_project()

    def closeEvent(self, event):
        self._autosave_timer.stop()
        self._stop_preview_players(remove_file=True)
        self._stop_audio()
        self.player.stop()
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop_requested = True
            self.statusBar().showMessage("Запис зупиняється. Закрийте програму ще раз за мить.")
            event.ignore()
            return
        if self.job and self.job.isRunning():
            self.statusBar().showMessage("Фонова операція триває. Дочекайтеся її завершення перед виходом.")
            event.ignore()
            return
        self.save_project()
        event.accept()


def install_ui_exception_handler(window, log_path: Path | None = None):
    error_log_path = log_path or Path(__file__).with_name("voxora_errors.log")

    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        details = "".join(traceback.format_exception(
            exc_type, exc_value, exc_traceback))
        log_written = True
        try:
            with error_log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                log.write(details)
        except OSError:
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            log_written = False
        try:
            message = "Помилка сталася під час операції. "
            if log_written:
                message += f"Деталі збережено у файлі:\n{error_log_path}"
            else:
                message += "Не вдалося записати файл журналу; дивіться консоль запуску."
            window.statusBar().showMessage(message.replace("\n", " "), 15000)
            QMessageBox.critical(window, "Помилка Voxora Studio", message)
        except Exception:
            traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_unhandled_exception


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    window = VoxoraStudio()
    install_ui_exception_handler(window)
    window.show()
    if not shutil.which("ffmpeg"):
        QTimer.singleShot(0, lambda: QMessageBox.warning(
            window, APP_TITLE,
            "FFmpeg не знайдено. Для імпорту MP3 та експорту MP4 встановіть FFmpeg "
            "і додайте його до PATH."))
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise

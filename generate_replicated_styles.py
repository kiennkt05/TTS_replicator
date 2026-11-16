"""Generate Vietnamese style references with the same replicated speaker.

Requires google-genai and google-auth, a Google Cloud service-account key,
source_neutral_69s.wav, and consent.wav. The voice key is not stored.
"""

import base64
import argparse
import io
import json
import os
import sys
import wave
from pathlib import Path

from google import genai
from google.oauth2 import service_account

from tts_replicator import load_env


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source_neutral_69s.wav"
CONSENT_PATH = ROOT / "recordings" / "consent.wav"
OUTPUT_DIR = ROOT / "new_prompts"

# Text must match the generated audio when used as an OmniVoice ref_text.
STYLES = {
    "smooth_neutral": {
        "text": "Xin chào em luôn sẵn sàng hỗ trợ anh một cách tận tình và chu đáo",
        "style": "Giọng miền Bắc tự nhiên và trung tính nói trôi chảy liền mạch với nhịp đều không ngắt nghỉ bất thường",
    },
    "articulated": {
        "text": "Em sẽ đọc thật rõ từng thông tin để anh dễ dàng theo dõi nhé.",
        "style": "Giọng miền Bắc phát âm rõ ràng từng chữ mạch lạc tự nhiên với tốc độ vừa phải không ngắt quãng quá mức",
    },
    "apologetic": {
        "text": "Em thật lòng xin lỗi anh vì sự bất tiện này và mong anh thông cảm ạ.",
        "style": "Giọng miền Bắc buồn bã, giọng trầm lại thể hiện sự áy náy và hối lỗi",
    },
    "excited": {
        "text": "Tuyệt quá anh ơi em rất vui vì cuối cùng mình đã tìm được rồi!",
        "style": "Giọng miền Bắc hào hứng tươi sáng vui vẻ đầy năng lượng thể hiện sự phấn khích và niềm vui tự nhiên",
    },
}


def read_input(path: Path, *, source: bool = False, allow_short_source: bool = False) -> bytes:
    data = path.read_bytes()
    with wave.open(io.BytesIO(data), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (24000, 1, 2, "NONE"):
            raise ValueError(f"{path} must be 24 kHz mono 16-bit PCM WAV")
        if source and not allow_short_source and audio.getnframes() / audio.getframerate() < 60:
            raise ValueError("The source recording must be at least 60 seconds")
    return data


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--consent", type=Path, default=CONSENT_PATH)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing valid WAV samples and generate missing ones")
    parser.add_argument("--allow-short-source", action="store_true", help="Submit the original short source for Google's validation")
    parser.add_argument("--enroll-only", action="store_true", help="Test voice enrollment without generating style samples")
    args = parser.parse_args()
    load_env(ROOT / ".env")
    credentials_path = args.credentials or Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(ROOT / "credentials.json")))
    if not credentials_path.is_absolute():
        credentials_path = ROOT / credentials_path
    for path in (args.source, args.consent, credentials_path):
        if not path.is_file():
            parser.error(f"Required file not found: {path}")
    filenames = [] if args.enroll_only else [*(f"{name}.wav" for name in STYLES), "replicated_styles.json"]
    for filename in filenames:
        if (args.output_dir / filename).exists():
            if not args.skip_existing:
                parser.error(f"Output exists: {args.output_dir / filename}; choose a fresh --output-dir or --skip-existing")
            if filename.endswith(".wav"):
                read_input(args.output_dir / filename)
    source = read_input(args.source, source=True, allow_short_source=args.allow_short_source)
    consent = read_input(args.consent)
    info = json.loads(credentials_path.read_text(encoding="utf-8-sig"))
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with genai.Client(
        enterprise=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT") or info["project_id"],
        location="global",
        credentials=credentials,
        http_options={"timeout": 180_000},
    ) as client:
        print(f"Creating stateless voice from {args.source.name} and {args.consent.name}", flush=True)
        voice = client.voices.create(
            store=False,
            voice={
                "model": "gemini-3.8-flash-tts",
                "type": "VOICE_TYPE_REPLICATED",
                "replicated": {
                    "source_audio": {"mime_type": "audio/wav", "data": base64.b64encode(source).decode("ascii")},
                    "consent_audio": {"mime_type": "audio/wav", "data": base64.b64encode(consent).decode("ascii")},
                },
            },
            timeout=180,
        )
        if args.enroll_only:
            print("Google accepted voice enrollment.", flush=True)
            return
        for name, settings in STYLES.items():
            path = args.output_dir / f"{name}.wav"
            if args.skip_existing and path.exists():
                print(f"Kept {path}", flush=True)
                continue
            response = client.models.generate_content(
                model="gemini-3.8-flash-tts",
                contents=[{"role": "user", "parts": [{
                    "text": settings["text"],
                    "speech_metadata": {"style": settings["style"]},
                }]}],
                config={
                    "response_modalities": ["AUDIO"],
                    "speech_config": {"voice_config": {"voice": voice.key}},
                },
            )
            parts = response.candidates[0].content.parts if response.candidates else []
            data = next((part.inline_data.data for part in parts if part.inline_data and part.inline_data.data), None)
            if not data:
                raise RuntimeError(f"Gemini returned no audio for {name}")
            with wave.open(io.BytesIO(data), "rb") as audio:
                if audio.getnframes() <= 0:
                    raise RuntimeError(f"Gemini returned empty audio for {name}")
            path.write_bytes(data)
            print(f"Wrote {path}: {settings['text']}", flush=True)

    (args.output_dir / "replicated_styles.json").write_text(
        json.dumps(STYLES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

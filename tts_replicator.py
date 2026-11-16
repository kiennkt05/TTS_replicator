"""Gemini voice replication on Google Cloud using a Service Account JSON.

Run `python tts_replicator.py --help`. Install dependencies with pip install -e .
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, request
import wave

MODEL = "gemini-3.8-flash-tts"
API_ROOT = "https://aiplatform.googleapis.com"
BACKEND = "google-cloud"
MAX_RECORDING_BYTES = 20 * 1024 * 1024  # Local upload guard, not an API limit.
STYLE_PROMPTS = {
    "neutral": "Speak Vietnamese naturally, with a neutral conversational tone and normal pace.",
    "articulated": "Speak Vietnamese with precise, clear articulation and distinct phrase boundaries, without exaggeration.",
    "slow": "Speak Vietnamese slowly and calmly, with natural pauses and connected phrasing.",
    "fast": "Speak Vietnamese at a brisk pace while keeping every word clear and intelligible.",
    "excited": "Speak Vietnamese with genuine excitement, lively intonation, and warm energetic delivery.",
    "apologetic": "Speak Vietnamese with a sincere apologetic tone, gentle pacing, and reassuring delivery.",
    "whisper": "Whisper in Vietnamese softly and intimately while keeping the words intelligible.",
}


class ReplicatorError(Exception):
    """An actionable input, transport, or API error."""


def load_env(path: Path) -> None:
    """Read simple KEY=value settings without overriding the process environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key.strip() not in {"GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ.setdefault(key.strip(), value)


class GeminiClient:
    def __init__(self, credentials, project: str, timeout: float = 180):
        if not project or not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]|[0-9]+", project):
            raise ReplicatorError("Set a valid GOOGLE_CLOUD_PROJECT or --project.")
        self.credentials = credentials
        self.project = project
        self.timeout = timeout

    @classmethod
    def from_service_account(cls, path: Path, project: str | None = None):
        try:
            from google.oauth2 import service_account
            from google.auth.exceptions import GoogleAuthError
        except ImportError:
            raise ReplicatorError("Install dependencies first: python -m pip install -e .") from None
        try:
            # Windows editors may prepend a UTF-8 BOM to an otherwise valid key file.
            info = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(info, dict) or info.get("type") != "service_account":
                raise ValueError("Expected a service_account JSON object")
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except (ValueError, OSError, GoogleAuthError):
            raise ReplicatorError("Cannot load Service Account JSON. Check the file path and key format.") from None
        return cls(credentials, project or credentials.project_id)

    def post(self, endpoint: str, payload: dict) -> dict:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request as AuthRequest

        parent = f"projects/{self.project}/locations/global"
        if endpoint == "voices":
            url = f"{API_ROOT}/v1beta1/{parent}/voices"
        elif endpoint == "generateContent":
            url = f"{API_ROOT}/v1/{parent}/publishers/google/models/{MODEL}:generateContent"
        else:
            raise ReplicatorError("Unsupported Google Cloud operation.")
        headers = {"Content-Type": "application/json"}
        try:
            # Refreshes expired tokens before every request, including long corpus jobs.
            self.credentials.before_request(AuthRequest(), "POST", url, headers)
        except (GoogleAuthError, OSError, ValueError):
            raise ReplicatorError("Service Account authentication failed. Check the key, network, and system clock.") from None
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
        except error.HTTPError as exc:
            # Keep only the standard error message; never print a whole request/response.
            detail = ""
            try:
                body = json.loads(exc.read(16384))
                message = body.get("error", {}).get("message", "")
                if isinstance(message, str):
                    message = re.sub(r"[A-Za-z0-9_+/=-]{80,}", "[redacted]", message)
                    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", message)
                    detail = " " + message[:1200]
            except (ValueError, AttributeError, OSError):
                pass
            hints = {
                400: "Check the recordings, consent statement, voice ID, and request inputs.",
                401: "Check the Service Account credentials and access token.",
                403: "Check Service Account IAM permissions, the aiplatform API, and project billing.",
                404: "Check model/voice availability for this project.",
                429: "Quota or rate limit reached; check billing and retry later.",
            }
            hint = hints.get(exc.code, "Google could not complete the request. Retry later.")
            raise ReplicatorError(f"Gemini HTTP {exc.code}: {hint}{detail}") from None
        except (error.URLError, TimeoutError, OSError):
            raise ReplicatorError(
                "Cannot reach Gemini or request timed out. Check the network. "
                "A submitted request may still have completed; check Google Cloud before retrying."
            ) from None
        except (ValueError, UnicodeError):
            raise ReplicatorError("Gemini returned an invalid JSON response.") from None
        if not isinstance(result, dict):
            raise ReplicatorError("Gemini returned an unexpected response structure.")
        return result


def recording(path: Path, *, reference: bool = False, allow_short_reference: bool = False) -> dict:
    if path.stat().st_size > MAX_RECORDING_BYTES:
        raise ReplicatorError(f"Recording exceeds the local 20 MiB limit: {path}")
    raw = path.read_bytes()
    try:
        with wave.open(io.BytesIO(raw), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
                raise ReplicatorError(f"Use 24 kHz mono 16-bit PCM WAV for {path}.")
            frames = audio.getnframes()
            duration = frames / audio.getframerate()
            if not frames or len(audio.readframes(frames)) != frames * 2:
                raise ReplicatorError(f"Empty or truncated WAV recording: {path}")
    except (wave.Error, EOFError):
        raise ReplicatorError(f"Invalid PCM WAV recording: {path}") from None
    if reference and duration < 60 and not allow_short_reference:
        raise ReplicatorError(f"Use at least 60 seconds of reference audio for Google Cloud; got {duration:.1f}s.")
    return {"mimeType": "audio/wav", "data": base64.b64encode(raw).decode("ascii")}


def prepare_output(path: Path) -> None:
    if path.exists():
        raise ReplicatorError(f"Output already exists; choose another path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def create_voice(client: GeminiClient, source: Path, consent: Path, name: str,
                 output: Path, *, store: bool = True, allow_short_reference: bool = False) -> str:
    if not name.strip():
        raise ReplicatorError("Voice name cannot be empty.")
    payload = {
        "store": store,
        "voice": {
            "type": "VOICE_TYPE_REPLICATED",
            "displayName": name.strip(),
            "replicated": {
                "sourceAudio": recording(source, reference=True, allow_short_reference=allow_short_reference),
                "consentAudio": recording(consent),
            },
        },
    }
    # Cloud rejects voice.model for stored voices, but requires it for stateless keys.
    if not store:
        payload["voice"]["model"] = MODEL
    prepare_output(output)
    result = client.post("voices", payload)
    field, prefix = ("id", "voice_") if store else ("key", "voicekey_")
    identifier = result.get(field)
    if not isinstance(identifier, str) or not identifier.startswith(prefix):
        raise ReplicatorError("Gemini did not return a valid voice identifier. Check Google Cloud.")
    profile = {"model": MODEL, "display_name": name.strip(), "voice": identifier,
               "backend": BACKEND, "project": client.project, "location": "global",
               "store": store, "expire_time": result.get("expireTime")}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return identifier


def read_voice(path: Path, project: str | None = None) -> str:
    try:
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError:
        raise ReplicatorError(f"Invalid voice profile JSON: {path}") from None
    voice = profile.get("voice") if isinstance(profile, dict) else None
    if not isinstance(voice, str) or not voice.startswith(("voice_", "voicekey_")):
        raise ReplicatorError(f"Voice profile must contain a voice_ ID or voicekey_ key: {path}")
    if profile.get("backend") != BACKEND or profile.get("location") != "global":
        raise ReplicatorError("Use a Google Cloud voice profile created by this tool, or pass a Cloud voice ID with --voice.")
    if project and profile.get("project") != project:
        raise ReplicatorError("Voice profile belongs to a different Google Cloud project.")
    return voice


def extract_wav(response: dict) -> bytes:
    # Cloud generateContent returns complete WAV data under candidates[].content.parts[].
    candidates = response.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
    blocks = [part["inlineData"] for part in parts if isinstance(part, dict)
              and isinstance(part.get("inlineData"), dict)
              and part["inlineData"].get("mimeType", "").startswith("audio/")]
    if not blocks or not blocks[-1].get("data"):
        raise ReplicatorError("Gemini returned no audio. Check the text and voice access in Google Cloud.")
    if len(blocks) != 1:
        raise ReplicatorError("Unexpected multiple audio parts in unary Google Cloud response.")
    try:
        raw = base64.b64decode(blocks[-1]["data"], validate=True)
        with wave.open(io.BytesIO(raw), "rb") as audio:
            frames = audio.getnframes()
            expected = frames * audio.getnchannels() * audio.getsampwidth()
            if frames == 0 or len(audio.readframes(frames)) != expected:
                raise ValueError("Empty or truncated audio")
    except (binascii.Error, ValueError, TypeError, wave.Error, EOFError):
        raise ReplicatorError("Gemini returned invalid or incomplete WAV audio.") from None
    return raw


def synthesize(client: GeminiClient, text: str, voice: str, style: str, output: Path) -> None:
    if not text.strip() or not voice.strip():
        raise ReplicatorError("Text and voice must not be empty.")
    if output.suffix.lower() != ".wav":
        raise ReplicatorError("Use a .wav output filename.")
    content = {"text": text}
    if style.strip():
        content["speechMetadata"] = {"style": style.strip()}
    payload = {
        "contents": [{"role": "user", "parts": [content]}],
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"voice": voice.strip()}}},
    }
    prepare_output(output)
    audio = extract_wav(client.post("generateContent", payload))
    with output.open("xb") as handle:
        handle.write(audio)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected an object")
            except ValueError:
                raise ReplicatorError(f"Invalid JSONL object at {path}:{number}") from None
            records.append(row)
    return records


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def batch(client: GeminiClient | None, source: Path, output_dir: Path, voice: str,
          *, dry_run: bool = False, limit: int | None = None, delay: float = 1.0,
          project: str | None = None) -> dict:
    """Consume scripts_generator JSONL; checkpoint each generated sample for resume."""
    if limit is not None and limit < 1:
        raise ReplicatorError("--limit must be positive.")
    if delay < 0 or not delay < float("inf"):
        raise ReplicatorError("--delay must be a finite non-negative number.")
    project = client.project if client is not None else project
    rows = read_jsonl(source)
    if not rows:
        raise ReplicatorError("The input JSONL contains no scripts.")
    seen = set()
    for row in rows:
        sid = row.get("script_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", sid):
            raise ReplicatorError("Each script_id must be a safe filename using letters, digits, _ or -.")
        # Prevent collisions on Windows, including reserved device names.
        if sid.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
            raise ReplicatorError(f"Reserved filename in script_id: {sid}")
        if sid.casefold() in seen:
            raise ReplicatorError(f"Duplicate script_id: {sid}")
        seen.add(sid.casefold())
        if row.get("qc_status") != "text_accepted":
            raise ReplicatorError(f"{sid}: input must have qc_status=text_accepted.")
        if not isinstance(row.get("spoken_text"), str) or not row["spoken_text"].strip():
            raise ReplicatorError(f"{sid}: missing spoken_text.")
        if not isinstance(row.get("style_id"), str) or row["style_id"] not in STYLE_PROMPTS:
            raise ReplicatorError(f"{sid}: unknown style_id; expected one of {', '.join(STYLE_PROMPTS)}.")
    output_dir = output_dir.resolve()
    manifest = output_dir / "scripts_with_audio.jsonl"
    if source.resolve() == manifest:
        raise ReplicatorError("Input and output manifest must be different files.")
    existing = read_jsonl(manifest) if manifest.exists() else []
    by_id = {}
    for row in existing:
        sid = row.get("script_id")
        if not isinstance(sid, str) or sid in by_id:
            raise ReplicatorError("Output manifest contains missing or duplicate IDs.")
        by_id[sid] = row
    if set(by_id) - {row["script_id"] for row in rows}:
        raise ReplicatorError("Output directory belongs to another script set; choose a fresh --output-dir.")
    pending = []
    for row in rows:
        sid = row["script_id"]
        path = output_dir / "audio" / f"{sid}.wav"
        style = STYLE_PROMPTS[row["style_id"]]
        digest = fingerprint({"script": row, "model": MODEL, "voice": voice, "style": style,
                              "backend": BACKEND, "project": project, "location": "global"})
        if sid in by_id:
            previous = by_id[sid]
            if previous.get("tts_fingerprint") != digest:
                raise ReplicatorError(f"{sid}: input, voice, or style changed; choose a fresh --output-dir.")
            if (previous.get("audio_path") != str(path) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != previous.get("audio_sha256")):
                raise ReplicatorError(f"{sid}: saved audio is missing or changed; choose a fresh --output-dir.")
        else:
            if path.exists():
                raise ReplicatorError(f"Untracked audio exists at {path}; move it aside before resuming.")
            pending.append((row, path, style, digest))
    selected = pending if limit is None else pending[:limit]
    stats = {"total": len(rows), "existing": len(existing), "planned": len(selected), "generated": 0}
    if dry_run:
        for row, path, _, _ in selected:
            print(f"{row['script_id']} [{row['style_id']}] -> {path}")
        return stats
    if client is None:
        raise ReplicatorError("A Gemini client is required for synthesis.")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".batch.lock"
    try:
        lock_handle = lock.open("x")
    except FileExistsError:
        raise ReplicatorError(f"Batch lock exists: {lock}. If no batch is running, remove the stale lock.") from None
    try:
        with lock_handle:
            lock_handle.write(str(os.getpid()))
            lock_handle.flush()
            # Detect another process finishing between preflight and lock acquisition.
            current = read_jsonl(manifest) if manifest.exists() else []
            if current != existing:
                raise ReplicatorError("Output manifest changed during preflight; rerun to resume.")
            for row, path, style, digest in selected:
                if stats["generated"]:
                    time.sleep(delay)
                synthesize(client, row["spoken_text"], voice, style, path)
                with wave.open(str(path), "rb") as audio:
                    details = {"sample_rate_hz": audio.getframerate(), "channels": audio.getnchannels(),
                               "sample_width_bytes": audio.getsampwidth(),
                               "duration_seconds": round(audio.getnframes() / audio.getframerate(), 6)}
                record = {**row, "audio_path": str(path), "audio_status": "generated",
                          "audio_qc_status": "pending_review", "audio_format": details,
                          "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "tts_model": MODEL, "tts_style_prompt": style,
                          "tts_backend": BACKEND, "tts_project": project, "tts_location": "global",
                          "tts_voice_sha256": hashlib.sha256(voice.encode("utf-8")).hexdigest(),
                          "tts_fingerprint": digest,
                          "audio_created_at": datetime.now(timezone.utc).isoformat()}
                # Replace the whole manifest atomically: no truncated last JSONL row on interruption.
                updated = existing + [record]
                temporary = manifest.with_suffix(".jsonl.tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    for saved in updated:
                        handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(manifest)
                existing = updated
                stats["generated"] += 1
                print(f"Generated {row['script_id']} -> {path}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return stats


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--env-file", type=Path, default=Path(".env"))
    root.add_argument("--credentials", type=Path, help="Service Account JSON; defaults to GOOGLE_APPLICATION_CREDENTIALS")
    root.add_argument("--project", help="Cloud project; defaults to GOOGLE_CLOUD_PROJECT or JSON project_id")
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-voice", help="Upload reference and consent WAVs to replicate a voice")
    create.add_argument("--reference", type=Path, required=True)
    create.add_argument("--consent", type=Path, required=True)
    create.add_argument("--name", default="My replicated voice")
    create.add_argument("--output", type=Path, default=Path("voices/my-voice.json"))
    create.add_argument("--stateless", action="store_true", help="Save an expiring voice key instead of a stored voice ID")
    speak = commands.add_parser("speak", help="Generate a WAV using a saved voice or explicit voice ID")
    source = speak.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    voice = speak.add_mutually_exclusive_group()
    voice.add_argument("--voice", help="voice_ ID, voicekey_ key, or prebuilt name such as Kore")
    voice.add_argument("--voice-file", type=Path, default=Path("voices/my-voice.json"))
    speak.add_argument("--style", default="warm and conversational")
    speak.add_argument("--output", type=Path, default=Path("outputs/speech.wav"))
    corpus = commands.add_parser("batch", help="Convert scripts_generator accepted JSONL to a resumable WAV corpus")
    corpus.add_argument("--input", type=Path, required=True, help="Path to scripts_accepted.jsonl")
    corpus.add_argument("--output-dir", type=Path, default=Path("outputs/dataset"))
    corpus_voice = corpus.add_mutually_exclusive_group()
    corpus_voice.add_argument("--voice", help="Explicit replicated voice ID or prebuilt voice name")
    corpus_voice.add_argument("--voice-file", type=Path, default=Path("voices/my-voice.json"))
    corpus.add_argument("--dry-run", action="store_true", help="Validate scripts and print plan without API calls or output files")
    corpus.add_argument("--limit", type=int, help="Maximum new audio files to generate this run")
    corpus.add_argument("--delay", type=float, default=1.0, help="Seconds between synthesis calls (default: 1)")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        load_env(args.env_file)
        credential_path = args.credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        client = GeminiClient.from_service_account(Path(credential_path), project) if credential_path else None
        if client is not None:
            project = client.project
        if args.command == "batch" and args.dry_run:
            voice = args.voice if args.voice is not None else (
                read_voice(args.voice_file, project) if args.voice_file.exists() else "<unconfigured-voice>")
            print(json.dumps(batch(None, args.input, args.output_dir, voice,
                                   dry_run=True, limit=args.limit, delay=args.delay, project=project)))
            return 0
        if client is None:
            raise ReplicatorError("Set GOOGLE_APPLICATION_CREDENTIALS or pass --credentials with your Service Account JSON path.")
        if args.command == "create-voice":
            create_voice(client, args.reference, args.consent, args.name, args.output, store=not args.stateless)
            print(f"Voice profile saved to {args.output}")
        elif args.command == "batch":
            voice = args.voice if args.voice is not None else read_voice(args.voice_file, project)
            print(json.dumps(batch(client, args.input, args.output_dir, voice,
                                   limit=args.limit, delay=args.delay)))
        else:
            text = args.text_file.read_text(encoding="utf-8-sig") if args.text_file else args.text
            voice = args.voice if args.voice is not None else read_voice(args.voice_file, project)
            synthesize(client, text, voice, args.style, args.output)
            print(f"Generated audio saved to {args.output}")
        return 0
    except (ReplicatorError, OSError, UnicodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed batch records are saved; rerun to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

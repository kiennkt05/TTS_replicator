"""Render the first accepted scripts in seven styles using one replicated voice."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import tts_replicator as tts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args) -> dict:
    rows = [r for r in tts.read_jsonl(args.input) if r.get("qc_status") == "text_accepted"][:args.count]
    if len(rows) != args.count or len({r.get("script_id") for r in rows}) != len(rows):
        raise tts.ReplicatorError("Need the requested number of accepted scripts with unique IDs.")
    source = args.recordings / "source_neutral_69s.wav"
    consent = args.recordings / "consent.wav"
    tts.recording(source, reference=True)
    tts.recording(consent)
    pool = json.loads((args.recordings / "style_pool.json").read_text(encoding="utf-8-sig"))["style_pool"]
    styles = [item["id"] for item in pool]
    if len(styles) != 7 or set(styles) != set(tts.STYLE_PROMPTS):
        raise tts.ReplicatorError("style_pool.json must contain exactly the seven supported styles.")
    plan = {"input": str(args.input.resolve()), "script_ids": [r["script_id"] for r in rows],
            "styles": styles, "source": str(source.resolve()), "source_sha256": sha256(source),
            "consent": str(consent.resolve()), "consent_sha256": sha256(consent),
            "model": tts.MODEL, "planned_wavs": len(rows) * len(styles)}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return {"planned": plan["planned_wavs"], "generated": 0}

    tts.load_env(Path(".env"))
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise tts.ReplicatorError("Set GOOGLE_APPLICATION_CREDENTIALS.")
    client = tts.GeminiClient.from_service_account(Path(key_path), os.environ.get("GOOGLE_CLOUD_PROJECT"))
    plan["project"] = client.project
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / "plan.json"
    if plan_file.exists():
        if json.loads(plan_file.read_text(encoding="utf-8")) != plan:
            raise tts.ReplicatorError("Inputs or recordings changed. Choose a new output directory.")
    else:
        plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # One enrollment avoids the per-minute voice creation quota reached previously.
    voice_file = output / "voice.json"
    if voice_file.exists():
        voice = tts.read_voice(voice_file, client.project)
    else:
        print("Enrolling one voice from source_neutral_69s.wav and consent.wav", flush=True)
        voice = tts.create_voice(client, source, consent, "Neutral source for seven styles",
                                 voice_file, store=False)
    results = []
    for style in styles:
        directory = output / style
        directory.mkdir(parents=True, exist_ok=True)
        inputs = directory / "scripts_input.jsonl"
        styled_rows = [{**row, "source_style_id": row["style_id"], "style_id": style,
                        "reference_audio": plan["source"], "reference_sha256": plan["source_sha256"],
                        "consent_sha256": plan["consent_sha256"]} for row in rows]
        if inputs.exists():
            if tts.read_jsonl(inputs) != styled_rows:
                raise tts.ReplicatorError(f"Inputs changed for {style}. Choose a new output directory.")
        else:
            inputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in styled_rows), encoding="utf-8")
        attempts = 0
        while True:
            try:
                result = tts.batch(client, inputs, directory, voice, delay=args.delay)
                results.append({"style": style, **result})
                print(f"{style}: {json.dumps(result)}", flush=True)
                break
            except tts.ReplicatorError as exc:
                if "HTTP 429" not in str(exc) or attempts >= args.rate_limit_retries:
                    raise
                attempts += 1
                print(f"Rate limited; waiting 65 seconds before retry {attempts} for {style}", flush=True)
                time.sleep(65)
    records = [row for style in styles
               for row in tts.read_jsonl(output / style / "scripts_with_audio.jsonl")]
    combined = output / "all_audio.jsonl"
    combined.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {"status": "complete", "project": client.project, "model": tts.MODEL,
               "script_ids": plan["script_ids"], "styles": styles,
               "planned_wavs": plan["planned_wavs"], "generated_wavs": len(records), "results": results}
    (output / "run_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(records)} WAVs: {combined}", flush=True)
    return {"planned": plan["planned_wavs"], "generated": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--delay", type=float, default=3, help="Seconds between synthesis requests")
    parser.add_argument("--rate-limit-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.delay < 0 or args.rate_limit_retries < 0:
        parser.error("Count must be positive; delay and rate-limit-retries must be non-negative.")
    try:
        run(args)
        return 0
    except (tts.ReplicatorError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

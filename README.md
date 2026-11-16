# Gemini TTS Replicator

The speech-generation stage for the sibling `../scripts_generator` project.
Reads its accepted Vietnamese scripts, synthesizes one WAV per `script_id` using
`gemini-3.8-flash-tts`, and exports a JSONL manifest with audio paths and metadata.
Python 3.10+ with `google-auth[requests]`; no GPU needed. Authentication uses a
Google Cloud Service Account JSON and OAuth access tokens.

## Set up

Run commands from this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Edit `.env` to point to your existing JSON file:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=C:/keys/my-service-account.json
GOOGLE_CLOUD_PROJECT=my-google-cloud-project
```

`GOOGLE_CLOUD_PROJECT` is optional: when absent, the JSON's `project_id` is used.
Existing process environment values take precedence over `.env`. Paths are
relative to the working directory unless absolute. The private key stays in the
local JSON; `google-auth` signs OAuth requests and refreshes access tokens.
Neither `gcloud auth login` nor a Gemini API key is needed.

Alternatively pass `--credentials` and `--project` **before** the subcommand:

```powershell
python tts_replicator.py --credentials "C:/keys/my-service-account.json" --project my-google-cloud-project speak --voice Kore --text "Xin chào." --output outputs/test.wav
```

Enable billing and the `aiplatform.googleapis.com` API in the target project and
grant the Service Account permissions for the model and Voices API operations.
Follow [Google Cloud project setup](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/text-to-speech/overview#before_you_begin)
for your project's IAM configuration. A `403` indicates permissions, API access,
or billing must be checked. This tool does not change IAM or enable APIs.

The installed command `tts-replicator` is equivalent to `python tts_replicator.py`.
If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly.
Keep the credential JSON outside version control; `credentials/` and `secrets/`
are ignored if you choose to store it locally in this project.

## Create your voice once

Use two authentic recordings from the same adult speaker: a reference of at least
60 seconds and a consent recording, both 24 kHz mono 16-bit PCM WAV. Read the exact
Google Cloud consent statement for your language from
[Google Cloud's voice replication guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/text-to-speech/voice-replication#supported_languages_and_consent_statements).
The Cloud consent wording differs from the Gemini Developer API wording used in
the earlier version of this README. Vietnamese is listed in the Cloud guide.

Place recordings in `recordings/`, then run:

```powershell
python tts_replicator.py create-voice --reference recordings/reference.wav --consent recordings/consent.wav --name "My Vietnamese voice"
```

This uploads both clips to Google, creates a stored voice, and saves its ID in
`voices/my-voice.json`. Reuse the profile for subsequent batches. `--stateless`
instead saves an expiring voice key. Keep voice profiles private. If you already
created a voice in the same Google Cloud project, pass its ID using
`--voice voice_YOUR_ID`. Profiles include their Cloud project and are checked on
load. Developer API/AI Studio voice profiles are not treated as Cloud profiles.

The implementation uses `v1beta1/.../locations/global/voices` to create voices
and `v1/.../locations/global/publishers/google/models/...:generateContent` for
synthesis. Stored voice creation omits `voice.model`; stateless creation includes
it. Custom voices require `global`, so this tool fixes the location to `global`.

If FFmpeg is installed, convert each recording with:

```powershell
ffmpeg -i original.wav -ar 24000 -ac 1 -c:a pcm_s16le recordings/reference.wav
```

## Generate audio from scripts_generator

The existing dataset discovered during setup is
`../scripts_generator/outputs/deepseek/scripts_accepted.jsonl`.
Validate it without credentials, writes, or API calls:

```powershell
python tts_replicator.py batch --input ../scripts_generator/outputs/deepseek/scripts_accepted.jsonl --output-dir outputs/deepseek --dry-run
```

Generate a small audition batch using your saved replicated voice:

```powershell
python tts_replicator.py batch --input ../scripts_generator/outputs/deepseek/scripts_accepted.jsonl --output-dir outputs/deepseek --limit 3
```

Listen to those WAVs, then run the same command without `--limit` to finish:

```powershell
python tts_replicator.py batch --input ../scripts_generator/outputs/deepseek/scripts_accepted.jsonl --output-dir outputs/deepseek
```

Use `--input` for any other generator output, such as
`../scripts_generator/outputs/gemma31b/scripts_accepted.jsonl`, and choose a
separate output directory for each dataset/voice variant. The input is read as a
snapshot; rerun after the generator adds accepted scripts. There is no watcher.

Outputs:

```text
outputs/deepseek/
  audio/vi_000001.wav
  audio/vi_000002.wav
  scripts_with_audio.jsonl
```

The source JSONL remains unchanged. Each output record keeps the original fields
and adds an absolute `audio_path`, `tts_model`, `tts_style_prompt`, voice/input
fingerprints, Cloud project/location, audio checksum, timestamp, and audio format/duration. Voice keys
are not copied into the corpus manifest.

## Style and QC behavior

`spoken_text` is sent verbatim. `style_id` selects delivery metadata separately:

| style_id | Delivery |
| --- | --- |
| neutral | Natural, conversational, normal pace |
| articulated | Clear articulation and phrase boundaries |
| slow | Slower delivery with natural pauses |
| fast | Brisk, intelligible delivery |
| excited | Warm, energetic, lively intonation |
| apologetic | Sincere, gentle, reassuring |
| whisper | Soft, intelligible whisper |

These prompts are implementation defaults in `STYLE_PROMPTS`, not transcriptions
of the reference recordings mentioned in the shared chat. Changing them requires
a new output directory when resuming existing audio.

Input rows must have unique safe `script_id` values, nonempty `spoken_text`, a
known `style_id`, and `qc_status: "text_accepted"`. Text QC stays with
`scripts_generator`; this stage does not rewrite text or redo its word counts.
WAV structure and completeness are validated. There is no common duration cutoff
across styles. Output records have `audio_status: "generated"` and
`audio_qc_status: "pending_review"`: transcript accuracy, identity similarity,
style fidelity, pronunciation, noise, and clipping still need audio QC.
Forced alignment and streaming chunking are downstream stages.

## Resume and errors

- Completed samples are checkpointed after each request. Rerun the command to
  skip files whose input/voice/style/project fingerprints and audio checksums match.
- Changed inputs, a different voice, or damaged audio cause a clear error;
  choose a new output directory for a replacement run.
- API failures stop the batch without automatic retries. Earlier completed
  samples are retained. For rate limits, wait and/or increase `--delay`.
- `--limit` caps newly synthesized samples, not already completed ones.
- Run only one process per output directory. After a forced process termination,
  remove `.batch.lock` only once you have verified no batch is running.
- If interrupted between saving a WAV and checkpointing its record, that WAV is
  reported as untracked. Move it outside `audio/`, then rerun; the sample will be
  regenerated. Requests that time out may still be billed by the provider.
- Existing standalone output files and voice profiles are never overwritten.

## Single script and tests

To render the first three accepted scripts in all seven styles, using only
`recordings/source_neutral_69s.wav` as the reference and `recordings/consent.wav`
for verification, use the matrix runner:

```powershell
python generate_reference_matrix.py --input ../scripts_generator/outputs/deepseek_v4_flash_14/style_disabled/scripts_accepted.jsonl --output-dir outputs/deepseek_v4_flash_14_first3_shared_voice
```

The runner creates one stateless voice key, then applies seven `STYLE_PROMPTS`
to each script separately, producing 21 WAVs. It sends the script text verbatim.
The seven short reference clips are not submitted. Requests are paced by
`--delay` (3 seconds by default); HTTP 429 synthesis errors wait 65 seconds and
retry up to three times. The voice key is saved privately in the ignored output
directory so interrupted runs can resume, and expires after seven days. Use
`--dry-run` to inspect the plan. `all_audio.jsonl` combines the per-style
manifests. If recordings or scripts change, use a fresh output directory.

```powershell
python tts_replicator.py speak --text-file examples/script.txt --output outputs/example.wav
python tts_replicator.py speak --voice Kore --text "Xin chào, đây là giọng đọc thử." --output outputs/prebuilt.wav
python -m unittest -v
```

`Kore` is a prebuilt voice for testing, not a replica. Tests use fake responses and
make no paid calls. Live API availability and voice fidelity must be checked with
your project credentials and recordings.

API contracts: [Google Cloud voice replication](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/text-to-speech/voice-replication)
and [Google Cloud speech generation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/text-to-speech/overview).
Project context: [shared corpus guideline discussion](https://chatgpt.com/share/6afa8307-014c-83ec-9d60-691ef3b1ec83).

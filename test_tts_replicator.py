"""Offline contract and resume tests: python -m unittest -v."""

import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
import wave

import tts_replicator as tts


def cloud_audio(data):
    return {"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "audio/wav", "data": data}}
    ]}}]}


def fake_credentials():
    credentials = Mock(project_id="test-project")
    credentials.before_request.side_effect = lambda transport, method, url, headers: headers.update(
        {"authorization": "Bearer secret"})
    return credentials


def wav(seconds=0.1, rate=24000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x00\x01" * int(seconds * rate))
    return output.getvalue()


class FakeClient:
    project = "test-project"

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def post(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        if len(self.calls) == self.fail_at:
            raise tts.ReplicatorError("Simulated quota failure")
        if endpoint == "voices":
            return {"id": "voice_test", "key": "voicekey_test", "expireTime": "later"}
        return cloud_audio(base64.b64encode(wav()).decode("ascii"))


class ReplicatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "scripts_accepted.jsonl"
        self.output = self.root / "dataset"
        self.rows = [
            {"script_id": f"vi_{i:06d}", "spoken_text": f"Xin chào. Đây là kịch bản {i}.",
             "style_id": style, "qc_status": "text_accepted", "audio_path": None,
             "generation_model": "test-script-model"}
            for i, style in enumerate(tts.STYLE_PROMPTS, 1)
        ]
        self.write_rows(self.rows)
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def write_rows(self, rows):
        self.source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_batch_preserves_transcript_styles_metadata_and_resumes(self):
        original = self.source.read_bytes()
        client = FakeClient()
        result = tts.batch(client, self.source, self.output, "voice_test", delay=0)
        self.assertEqual(result["generated"], 7)
        records = tts.read_jsonl(self.output / "scripts_with_audio.jsonl")
        for row, record, (_, payload) in zip(self.rows, records, client.calls):
            self.assertEqual(record["spoken_text"], row["spoken_text"])
            self.assertEqual(record["generation_model"], "test-script-model")
            self.assertEqual(record["audio_qc_status"], "pending_review")
            self.assertEqual(record["qc_status"], "text_accepted")
            self.assertTrue(Path(record["audio_path"]).is_file())
            text_part = payload["contents"][0]["parts"][0]
            self.assertEqual(text_part["text"], row["spoken_text"])
            self.assertEqual(text_part["speechMetadata"]["style"], tts.STYLE_PROMPTS[row["style_id"]])
            self.assertEqual(payload["generationConfig"]["speechConfig"], {"voiceConfig": {"voice": "voice_test"}})
            self.assertEqual(record["tts_project"], "test-project")
        result = tts.batch(client, self.source, self.output, "voice_test", delay=0)
        self.assertEqual(result["generated"], 0)
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(self.source.read_bytes(), original)

    def test_failure_keeps_completed_records_and_can_resume(self):
        with self.assertRaises(tts.ReplicatorError):
            tts.batch(FakeClient(fail_at=2), self.source, self.output, "voice_test", delay=0)
        self.assertEqual(len(tts.read_jsonl(self.output / "scripts_with_audio.jsonl")), 1)
        self.assertFalse((self.output / ".batch.lock").exists())
        client = FakeClient()
        self.assertEqual(tts.batch(client, self.source, self.output, "voice_test", delay=0)["generated"], 6)

    def test_dry_run_requires_no_credentials_or_writes(self):
        with patch.dict(os.environ, {}, clear=True):
            code = tts.main(["--env-file", str(self.root / "missing.env"), "batch", "--input", str(self.source), "--output-dir", str(self.output), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertFalse(self.output.exists())

    def test_limit_counts_new_audio(self):
        client = FakeClient()
        for _ in range(2):
            self.assertEqual(tts.batch(client, self.source, self.output, "voice_test", limit=2, delay=0)["generated"], 2)
        self.assertEqual(len(client.calls), 4)

    def test_invalid_input_fails_before_api_calls(self):
        for override in ({"script_id": "../escape"}, {"script_id": "CON"},
                         {"style_id": "singing"}, {"qc_status": "rejected"}, {"spoken_text": ""}):
            with self.subTest(override=override):
                self.write_rows([self.rows[0], {**self.rows[1], **override}])
                client = FakeClient()
                with self.assertRaises(tts.ReplicatorError):
                    tts.batch(client, self.source, self.output, "voice_test")
                self.assertFalse(client.calls)

    def test_duplicate_ids_rejected_case_insensitively(self):
        self.write_rows([self.rows[0], {**self.rows[0], "script_id": "VI_000001"}])
        with self.assertRaisesRegex(tts.ReplicatorError, "Duplicate"):
            tts.batch(None, self.source, self.output, "voice_test", dry_run=True)

    def test_changed_text_or_voice_cannot_reuse_audio(self):
        tts.batch(FakeClient(), self.source, self.output, "voice_test", limit=1, delay=0)
        with self.assertRaisesRegex(tts.ReplicatorError, "changed"):
            tts.batch(FakeClient(), self.source, self.output, "voice_other", delay=0)
        self.rows[0]["spoken_text"] = "New transcript."
        self.write_rows(self.rows)
        with self.assertRaisesRegex(tts.ReplicatorError, "changed"):
            tts.batch(FakeClient(), self.source, self.output, "voice_test", delay=0)

    def test_modified_audio_cannot_be_silently_skipped(self):
        tts.batch(FakeClient(), self.source, self.output, "voice_test", limit=1, delay=0)
        (self.output / "audio/vi_000001.wav").write_bytes(b"broken")
        with self.assertRaisesRegex(tts.ReplicatorError, "missing or changed"):
            tts.batch(FakeClient(), self.source, self.output, "voice_test")

    def test_active_lock_blocks_generation(self):
        self.output.mkdir()
        lock = self.output / ".batch.lock"
        lock.write_text("other process")
        client = FakeClient()
        with self.assertRaisesRegex(tts.ReplicatorError, "lock"):
            tts.batch(client, self.source, self.output, "voice_test")
        self.assertTrue(lock.exists())
        self.assertFalse(client.calls)

    def test_voice_creation_and_stateless_profile(self):
        source, consent = self.root / "reference.wav", self.root / "consent.wav"
        source.write_bytes(wav(60))
        consent.write_bytes(wav(5))
        for store in (True, False):
            client = FakeClient()
            output = self.root / f"voice-{store}.json"
            tts.create_voice(client, source, consent, "My voice", output, store=store)
            self.assertEqual(tts.read_voice(output), "voice_test" if store else "voicekey_test")
            payload = client.calls[0][1]
            self.assertEqual(payload["store"], store)
            self.assertEqual(payload["voice"]["type"], "VOICE_TYPE_REPLICATED")
            if store:
                self.assertNotIn("model", payload["voice"])
            else:
                self.assertEqual(payload["voice"]["model"], tts.MODEL)
            self.assertEqual(base64.b64decode(payload["voice"]["replicated"]["consentAudio"]["data"]), consent.read_bytes())
            self.assertEqual(json.loads(output.read_text())["expire_time"], "later")
            with self.assertRaisesRegex(tts.ReplicatorError, "different Google Cloud project"):
                tts.read_voice(output, "another-project")

    def test_bad_reference_audio_is_rejected(self):
        path = self.root / "reference.wav"
        for raw in (b"not wav", wav(1), wav(10, rate=16000), wav(10)[:-20]):
            path.write_bytes(raw)
            with self.assertRaises(tts.ReplicatorError):
                tts.recording(path, reference=True)

    def test_existing_output_not_overwritten_or_billed(self):
        path = self.root / "audio.wav"
        path.write_bytes(b"keep me")
        client = FakeClient()
        with self.assertRaises(tts.ReplicatorError):
            tts.synthesize(client, "Hello", "Kore", "neutral", path)
        self.assertEqual(path.read_bytes(), b"keep me")
        self.assertFalse(client.calls)

    def test_invalid_audio_not_written(self):
        for data in ("!!!", base64.b64encode(b"bad wav").decode(), base64.b64encode(wav()[:-10]).decode()):
            with self.assertRaises(tts.ReplicatorError):
                tts.extract_wav(cloud_audio(data))
        with self.assertRaises(tts.ReplicatorError):
            tts.extract_wav({"candidates": []})

    def test_transport_uses_bearer_header_and_cloud_voices_endpoint(self):
        credentials = fake_credentials()
        with patch("tts_replicator.request.urlopen", return_value=io.BytesIO(b'{"id":"voice_test"}')) as opened:
            result = tts.GeminiClient(credentials, "test-project").post("voices", {"store": True})
        self.assertEqual(result["id"], "voice_test")
        req = opened.call_args.args[0]
        self.assertEqual(req.full_url, tts.API_ROOT + "/v1beta1/projects/test-project/locations/global/voices")
        self.assertEqual(req.get_header("Authorization"), "Bearer secret")
        self.assertIsNone(req.get_header("X-goog-api-key"))
        self.assertEqual(json.loads(req.data), {"store": True})
        credentials.before_request.assert_called_once()

    def test_transport_errors_do_not_leak_secrets(self):
        for problem in (HTTPError(tts.API_ROOT, 429, "secret", {}, io.BytesIO(b"secret")), URLError("secret")):
            with patch("tts_replicator.request.urlopen", side_effect=problem):
                with self.assertRaises(tts.ReplicatorError) as raised:
                    tts.GeminiClient(fake_credentials(), "test-project").post("generateContent", {})
            self.assertNotIn("secret", str(raised.exception))

    def test_env_preserves_process_credentials(self):
        path = self.root / ".env"
        path.write_text('GOOGLE_APPLICATION_CREDENTIALS="C:/keys/service-account.json"\nGOOGLE_CLOUD_PROJECT=file-project\n', encoding="utf-8")
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "process-project"}, clear=True):
            tts.load_env(path)
            self.assertEqual(os.environ["GOOGLE_CLOUD_PROJECT"], "process-project")
            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "C:/keys/service-account.json")

    def test_cloud_synthesis_endpoint_and_auth_each_request(self):
        credentials = fake_credentials()
        client = tts.GeminiClient(credentials, "test-project")
        with patch("tts_replicator.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(b'{}')) as opened:
            client.post("generateContent", {})
            client.post("generateContent", {})
        self.assertEqual(opened.call_args.args[0].full_url,
                         tts.API_ROOT + f"/v1/projects/test-project/locations/global/publishers/google/models/{tts.MODEL}:generateContent")
        self.assertEqual(credentials.before_request.call_count, 2)

    def test_service_account_project_defaults_and_scope(self):
        path = self.root / "key.json"
        path.write_text(json.dumps({"type": "service_account"}), encoding="utf-8-sig")
        with patch("google.oauth2.service_account.Credentials.from_service_account_info", return_value=fake_credentials()) as loaded:
            client = tts.GeminiClient.from_service_account(path)
            self.assertEqual(client.project, "test-project")
            loaded.assert_called_with({"type": "service_account"}, scopes=["https://www.googleapis.com/auth/cloud-platform"])
            client = tts.GeminiClient.from_service_account(path, "target-project")
            self.assertEqual(client.project, "target-project")

    def test_authentication_failure_is_redacted(self):
        from google.auth.exceptions import RefreshError
        credentials = fake_credentials()
        credentials.before_request.side_effect = RefreshError("secret-token")
        with patch("tts_replicator.request.urlopen") as opened:
            with self.assertRaises(tts.ReplicatorError) as raised:
                tts.GeminiClient(credentials, "test-project").post("voices", {})
            self.assertNotIn("secret-token", str(raised.exception))
            opened.assert_not_called()

    def test_service_account_error_is_redacted(self):
        path = self.root / "key.json"
        path.write_text(json.dumps({"type": "service_account"}), encoding="utf-8")
        with patch("google.oauth2.service_account.Credentials.from_service_account_info", side_effect=ValueError("private-key")):
            with self.assertRaises(tts.ReplicatorError) as raised:
                tts.GeminiClient.from_service_account(path)
            self.assertNotIn("private-key", str(raised.exception))

    def test_other_project_cannot_resume(self):
        client = FakeClient()
        tts.batch(client, self.source, self.output, "voice_test", limit=1, delay=0)
        client.project = "other-project"
        with self.assertRaisesRegex(tts.ReplicatorError, "changed"):
            tts.batch(client, self.source, self.output, "voice_test", delay=0)


if __name__ == "__main__":
    unittest.main()

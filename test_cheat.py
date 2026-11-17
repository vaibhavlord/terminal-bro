import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cheat


class CheatTests(unittest.TestCase):
    def test_extracts_tagged_fenced_and_plain_commands(self):
        self.assertEqual(cheat.extract_command("Hi <command>git status</command>"), "git status")
        self.assertEqual(cheat.extract_command("```bash\nfind . -name '*.py'\n```"), "find . -name '*.py'")
        self.assertEqual(cheat.extract_command("command: pwd"), "pwd")

    def test_builds_strict_command_only_prompt(self):
        messages = cheat.build_messages("show files", "zsh")
        self.assertEqual(messages[1]["content"], "show files")
        self.assertIn("ONLY the command", messages[0]["content"])

    def test_detects_shell_and_falls_back(self):
        with patch.dict("os.environ", {"SHELL": "/bin/zsh"}, clear=True):
            self.assertEqual(cheat.detect_shell(), "zsh")
        with patch.dict("os.environ", {"SHELL": "/bin/nushell"}, clear=True):
            self.assertEqual(cheat.detect_shell(), "sh")

    def test_sends_chat_to_local_ollama_endpoint(self):
        captured = {}

        def fake_request(host, endpoint, payload):
            captured.update(host=host, endpoint=endpoint, payload=payload)
            return {"message": {"content": "<command>pwd</command>"}}

        with patch("cheat.local_request", fake_request):
            output = cheat.request_command("http://127.0.0.1:11434", "local-test", [])
        self.assertEqual(output, "<command>pwd</command>")
        self.assertEqual(captured["endpoint"], "/api/chat")
        self.assertEqual(captured["payload"]["model"], "local-test")
        self.assertFalse(captured["payload"]["stream"])

    def test_reads_first_local_model(self):
        with patch("cheat.local_request", return_value={"models": [{"name": "llama3.2:latest"}]}):
            self.assertEqual(cheat.get_default_model("ignored"), "llama3.2:latest")

    def test_decodes_valid_utf8_json_response(self):
        expected = {"message": {"content": "<command>echo café</command>"}}
        body = json.dumps(expected, ensure_ascii=False).encode("utf-8")
        with patch("cheat.urllib.request.urlopen", return_value=io.BytesIO(body)):
            self.assertEqual(cheat.local_request(cheat.DEFAULT_HOST, "/api/chat"), expected)

    def test_cli_reports_malformed_responses_without_copying_or_executing(self):
        bodies = [b"not JSON", b"\xff", b"{}", b'{"message": null}']
        bodies.extend(
            json.dumps({"message": {"content": content}}).encode("utf-8")
            for content in (None, 42, False, [], {})
        )
        for body in bodies:
            with self.subTest(body=body):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch("sys.argv", ["cheat", "--model", "local-test", "--copy", "--execute", "show files"]), \
                     patch("cheat.load_config", return_value={}), \
                     patch("cheat.urllib.request.urlopen", return_value=io.BytesIO(body)), \
                     patch("sys.stdout", stdout), patch("sys.stderr", stderr), \
                     patch("cheat.copy_to_clipboard") as copy, \
                     patch("builtins.input") as confirmation, \
                     patch("cheat.subprocess.run") as execute:
                    self.assertEqual(cheat.main(), 1)
                self.assertIn("Error: Ollama returned", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                copy.assert_not_called()
                confirmation.assert_not_called()
                execute.assert_not_called()

    def test_rejects_non_local_ollama_hosts(self):
        self.assertEqual(cheat.validate_local_host("http://localhost:11434"), "http://localhost:11434")
        with self.assertRaisesRegex(cheat.OllamaError, "loopback"):
            cheat.validate_local_host("https://example.com")

    def test_cmd_execution_is_not_mapped_to_a_unix_shell(self):
        if cheat.os.name != "nt":
            with self.assertRaisesRegex(cheat.OllamaError, "Windows"):
                cheat.shell_command("cmd", "dir")

    def test_config_round_trip(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            cheat.save_config({"model": "qwen"}, path)
            self.assertEqual(json.loads(path.read_text())["model"], "qwen")
            self.assertEqual(cheat.load_config(path), {"model": "qwen"})

    def test_standard_library_installer_creates_a_working_command(self):
        project_dir = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "install.py", "--bin-dir", directory],
                cwd=project_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            executable = Path(directory) / ("cheat.cmd" if cheat.os.name == "nt" else "cheat")
            self.assertTrue(executable.exists())
            if cheat.os.name != "nt":
                version = subprocess.run([str(executable), "--version"], text=True, capture_output=True, check=False)
                self.assertEqual(version.returncode, 0, version.stderr)
                self.assertIn("cheat 1.0.0", version.stdout)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Terminal Cheat Code: local natural-language-to-command CLI for Ollama."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

DEFAULT_HOST = "http://127.0.0.1:11434"
VALID_SHELLS = ("bash", "zsh", "fish", "powershell", "cmd", "sh")
CONFIG_PATH = Path.home() / ".terminal-cheat-code.json"
__version__ = "1.0.0"


class OllamaError(RuntimeError):
    """A friendly error produced while talking to local Ollama."""


def color(code: str, text: str) -> str:
    """Apply ANSI color when writing to an interactive terminal."""
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def detect_shell() -> str:
    """Return the current shell in the names understood by this program."""
    if os.name == "nt":
        return "powershell"
    name = Path(os.environ.get("SHELL", "sh")).name.lower()
    return name if name in VALID_SHELLS else "sh"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_config(config: Dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # Windows does not support POSIX modes.
        pass


def validate_local_host(host: str) -> str:
    """Reject remote endpoints so prompts can never leave this computer."""
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise OllamaError("Ollama host must be a loopback address (localhost, 127.0.0.1, or ::1).")
    return host


def local_request(host: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    """Make a request only to the configured local Ollama endpoint."""
    url = validate_local_host(host).rstrip("/") + endpoint
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama request failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise OllamaError(f"Cannot reach Ollama at {host}. Start it with: ollama serve") from error
    except TimeoutError as error:
        raise OllamaError("Ollama took longer than 120 seconds. Try a smaller local model.") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise OllamaError("Ollama returned an invalid JSON response.") from error


def get_default_model(host: str) -> str:
    data = local_request(host, "/api/tags")
    models = data.get("models", []) if isinstance(data, dict) else []
    if not models or not models[0].get("name"):
        raise OllamaError("No local Ollama models found. Install one, for example: ollama pull qwen2.5-coder:7b")
    return models[0]["name"]


def build_messages(request: str, shell: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You translate developer requests into one executable {shell} shell command. "
                "Return ONLY the command inside <command> and </command> tags. Do not use Markdown, "
                "explanations, greetings, or multiple alternatives. Favor safe, non-destructive commands. "
                "If a destructive action is explicitly requested, include an inline comment explaining its effect."
            ),
        },
        {"role": "user", "content": request},
    ]


def request_command(host: str, model: str, messages: List[Dict[str, str]]) -> str:
    response = local_request(host, "/api/chat", {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    })
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as error:
        raise OllamaError("Ollama returned an unexpected response.") from error
    if not isinstance(content, str):
        raise OllamaError("Ollama returned an unexpected response: message.content must be a string.")
    return content


def extract_command(text: str) -> str:
    tagged = re.search(r"<command>\s*(.*?)\s*</command>", text, re.IGNORECASE | re.DOTALL)
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    command = (tagged.group(1) if tagged else fenced.group(1) if fenced else text).strip()
    command = re.sub(r"^\s*(?:command|shell command)\s*:\s*", "", command, flags=re.IGNORECASE)
    return command.strip("`").strip()


def copy_to_clipboard(text: str) -> bool:
    choices = (["pbcopy"], ["clip"]) if os.name == "nt" else (["pbcopy"],) if sys.platform == "darwin" else (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"])
    for command in choices:
        if shutil.which(command[0]):
            completed = subprocess.run(command, input=text, text=True, capture_output=True, check=False)
            if completed.returncode == 0:
                return True
    return False


def shell_command(shell: str, command: str) -> List[str]:
    if os.name == "nt":
        if shell == "powershell":
            return ["powershell.exe", "-NoProfile", "-Command", command]
        return ["cmd.exe", "/d", "/s", "/c", command]
    programs = {"bash": "/bin/bash", "zsh": "/bin/zsh", "fish": "fish", "powershell": "pwsh", "sh": "/bin/sh"}
    if shell == "cmd":
        raise OllamaError("The cmd shell can only be executed on Windows.")
    return [programs.get(shell, "/bin/sh"), "-c", command]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="cheat",
        description="Translate requests into shell commands with a local Ollama model only.",
        epilog='Example: cheat "undo my last git commit but keep the changes"',
    )
    result.add_argument("request", nargs="*", help="Plain-English command request")
    result.add_argument("-m", "--model", help="Local Ollama model to use")
    result.add_argument("-s", "--shell", choices=VALID_SHELLS, help="Target shell")
    result.add_argument("-e", "--execute", action="store_true", help="Ask before running the generated command")
    result.add_argument("-c", "--copy", action="store_true", help="Copy generated command to clipboard")
    result.add_argument("--host", help=f"Local Ollama host (default: {DEFAULT_HOST})")
    result.add_argument("--set-model", metavar="MODEL", help="Save default Ollama model")
    result.add_argument("--show-config", action="store_true", help="Print local configuration")
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config()
    if args.set_model:
        config["model"] = args.set_model
        save_config(config)
        print(f"Saved default model {args.set_model} to {CONFIG_PATH}")
        return 0
    if args.show_config:
        print(json.dumps({"configPath": str(CONFIG_PATH), **config}, indent=2))
        return 0
    request = " ".join(args.request).strip()
    if not request:
        parser().print_help()
        return 2

    host = args.host or config.get("host") or DEFAULT_HOST
    shell = args.shell or detect_shell()
    try:
        validate_local_host(host)
        model = args.model or config.get("model") or get_default_model(host)
        print(f"Asking {model} locally for a {shell} command…", file=sys.stderr)
        command = extract_command(request_command(host, model, build_messages(request, shell)))
        if not command:
            raise OllamaError("The model returned no command. Try a more specific request or another model.")
    except OllamaError as error:
        print(color("31", f"Error: {error}"), file=sys.stderr)
        return 1

    print(f"\n{color('36', 'Command')} ({shell})")
    print(color("32", command))
    if args.copy:
        print("Copied to clipboard." if copy_to_clipboard(command) else "Could not access a clipboard utility; copy the command above.")
    if args.execute:
        try:
            approved = input("\nRun this command? [y/N] ").strip().lower() in {"y", "yes"}
        except EOFError:
            approved = False
        if not approved:
            print("Not executed.")
            return 0
        try:
            return subprocess.run(shell_command(shell, command), check=False).returncode
        except (FileNotFoundError, OllamaError) as error:
            print(color("31", f"Error: cannot execute command: {error}"), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A local, read-only Linux hardware Q&A assistant."""

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
from typing import Any


COMMANDS: dict[str, tuple[str, ...]] = {
    "cpu": ("lscpu",),
    "storage": ("lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN"),
    "memory_topology": ("dmidecode", "--type", "memory"),
    "gpu": ("lspci",),
}
COMMAND_TIMEOUT_SECONDS = 5
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TOOL_TOPICS = {
    "inspect_cpu": "cpu",
    "inspect_memory": "memory",
    "inspect_storage": "storage",
    "inspect_gpu": "gpu",
}


def run_read_only_tool(tool_name: str) -> dict[str, Any]:
    """Run one fixed, allow-listed command; callers cannot supply arguments."""
    command = COMMANDS.get(tool_name)
    if command is None:
        raise ValueError(f"Unknown inspection tool: {tool_name}")

    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "error": f"{command[0]} is not installed"}

    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env={**os.environ, "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "error": f"{command[0]} timed out"}
    except OSError as exc:
        return {"available": False, "error": f"could not run {command[0]}: {exc}"}

    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        return {"available": False, "error": f"{command[0]} failed: {detail}"}
    return {"available": True, "stdout": result.stdout}


def _parse_lscpu(output: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()

    def integer_field(name: str) -> int | None:
        value = fields.get(name)
        if value is None:
            return None
        match = re.search(r"\d+", value.replace(",", ""))
        return int(match.group()) if match else None

    return {
        "architecture": fields.get("architecture"),
        "model": fields.get("model name") or fields.get("model"),
        "logical_cpu_count": integer_field("cpu(s)"),
        "cores_per_socket": integer_field("core(s) per socket"),
        "socket_count": integer_field("socket(s)"),
    }


def _read_memory_total(proc_root: Path = Path("/proc")) -> tuple[float | None, str | None]:
    try:
        contents = (proc_root / "meminfo").read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read /proc/meminfo: {exc}"

    for line in contents.splitlines():
        if line.startswith("MemTotal:"):
            match = re.search(r"(\d+)\s+kB", line, re.IGNORECASE)
            if match:
                # Linux reports MemTotal in KiB.
                return int(match.group(1)) / (1024 * 1024), None
            return None, "MemTotal has an unrecognized format"
    return None, "MemTotal was not reported by /proc/meminfo"


def _capacity_gib(value: str) -> float | None:
    match = re.search(r"([\d.]+)\s*(bytes?|kb|kib|mb|mib|gb|gib|tb|tib)\b", value, re.I)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    divisor = {
        "byte": 1024**3,
        "bytes": 1024**3,
        "kb": 1024**2,
        "kib": 1024**2,
        "mb": 1024,
        "mib": 1024,
        "gb": 1,
        "gib": 1,
        "tb": 1 / 1024,
        "tib": 1 / 1024,
    }[unit]
    return amount / divisor


def _parse_memory_topology(output: str) -> dict[str, Any]:
    devices = re.split(r"(?m)^\s*Memory Device\s*$", output)[1:]
    modules: list[dict[str, Any]] = []
    for device in devices:
        size_match = re.search(r"(?m)^\s*Size:\s*(.+?)\s*$", device)
        if not size_match:
            continue
        size_text = size_match.group(1)
        installed = "no module installed" not in size_text.lower() and "unknown" not in size_text.lower()
        module: dict[str, Any] = {
            "installed": installed,
            "size_gib": _capacity_gib(size_text) if installed else 0,
        }
        speed_match = re.search(r"(?m)^\s*(?:Configured Memory Speed|Speed):\s*(.+?)\s*$", device)
        if speed_match:
            module["speed"] = speed_match.group(1)
        modules.append(module)

    maximum_match = re.search(r"(?m)^\s*Maximum Capacity:\s*(.+?)\s*$", output)
    maximum_gib = _capacity_gib(maximum_match.group(1)) if maximum_match else None
    installed_modules = [module for module in modules if module["installed"]]
    installed_gib = sum(
        module["size_gib"] for module in installed_modules if module["size_gib"] is not None
    )
    return {
        "slots_total": len(modules) if modules else None,
        "slots_used": len(installed_modules) if modules else None,
        "slots_free": len(modules) - len(installed_modules) if modules else None,
        "installed_physical_gib": installed_gib if installed_modules else None,
        "maximum_supported_gib": maximum_gib,
        "modules": modules,
    }


def _flatten_block_devices(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("lsblk contains a non-object block device")
        if entry.get("type") == "disk":
            transport = str(entry.get("tran") or "").lower()
            name = str(entry.get("name") or "")
            rotational = entry.get("rota")
            devices.append(
                {
                    "name": name,
                    "model": entry.get("model"),
                    "size_bytes": entry.get("size"),
                    "transport": transport or None,
                    "is_nvme": transport == "nvme" or name.startswith("nvme"),
                    "is_rotational": bool(rotational) if rotational is not None else None,
                    "mountpoint": entry.get("mountpoint"),
                }
            )
        children = entry.get("children")
        if isinstance(children, list):
            devices.extend(_flatten_block_devices(children))
        elif children is not None:
            raise ValueError("lsblk device children must be a list")
    return devices


class HardwareAssistant:
    """Routes supported questions to deterministic, read-only inspection tools."""

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        self.proc_root = proc_root
        self._cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def topic_for(question: str) -> str | None:
        text = question.lower()
        if any(word in text for word in ("gpu", "graphics", "video card", "graphics card")):
            return "gpu"
        if any(word in text for word in ("ssd", "nvme", "disk", "storage", "drive")):
            return "storage"
        if any(word in text for word in ("ram", "memory", "upgrade", "slot")):
            return "memory"
        if any(word in text for word in ("cpu", "processor", "core", "architecture")):
            return "cpu"
        return None

    def inspect(self, topic: str) -> dict[str, Any]:
        if topic in self._cache:
            return self._cache[topic]
        if topic == "cpu":
            result = run_read_only_tool("cpu")
            data = _parse_lscpu(result["stdout"]) if result["available"] else {}
            facts = {"available": result["available"], "data": data, "error": result.get("error")}
        elif topic == "memory":
            total_gib, proc_error = _read_memory_total(self.proc_root)
            result = run_read_only_tool("memory_topology")
            topology = _parse_memory_topology(result["stdout"]) if result["available"] else {}
            facts = {
                "available": total_gib is not None or result["available"],
                "os_visible_gib": total_gib,
                "topology": topology,
                "errors": [error for error in (proc_error, result.get("error")) if error],
            }
        elif topic == "storage":
            result = run_read_only_tool("storage")
            try:
                parsed = json.loads(result["stdout"]) if result["available"] else {}
                entries = parsed.get("blockdevices", [])
                if not isinstance(entries, list):
                    raise ValueError("lsblk JSON has no valid blockdevices list")
                devices = _flatten_block_devices(entries)
                facts = {"available": True, "devices": devices}
            except (json.JSONDecodeError, ValueError) as exc:
                facts = {"available": False, "devices": [], "error": f"could not parse lsblk output: {exc}"}
            if not result["available"]:
                facts = {"available": False, "devices": [], "error": result["error"]}
        elif topic == "gpu":
            result = run_read_only_tool("gpu")
            if result["available"]:
                devices = []
                for line in result["stdout"].splitlines():
                    match = re.search(
                        r"\b(?:VGA compatible controller|3D controller|Display controller):\s*(.+)$",
                        line,
                        re.I,
                    )
                    if match:
                        devices.append(match.group(1).strip())
                facts = {"available": True, "devices": devices}
            else:
                facts = {"available": False, "devices": [], "error": result["error"]}
        else:
            raise ValueError(f"Unsupported hardware topic: {topic}")
        self._cache[topic] = facts
        return facts

    def answer(self, question: str) -> tuple[str, dict[str, Any] | None]:
        topic = self.topic_for(question)
        if topic is None:
            return (
                "I can check CPU, RAM and memory slots, storage/SSD type, or GPU. "
                "Which would you like to know about?",
                None,
            )

        facts = self.inspect(topic)
        if topic == "cpu":
            data = facts["data"]
            if not facts["available"]:
                return f"I could not read CPU details: {facts.get('error', 'data unavailable')}.", facts
            parts = []
            if data.get("model"):
                parts.append(str(data["model"]))
            if data.get("logical_cpu_count") is not None:
                parts.append(f"{data['logical_cpu_count']} logical CPUs")
            if data.get("architecture"):
                parts.append(f"{data['architecture']} architecture")
            return ("CPU details: " + ", ".join(parts) + "." if parts else "CPU details are unavailable.", facts)

        if topic == "memory":
            topology = facts["topology"]
            amount = topology.get("installed_physical_gib") or facts.get("os_visible_gib")
            parts = [f"{amount:.1f} GiB detected" if amount is not None else "installed capacity unavailable"]
            if topology.get("slots_total") is not None:
                parts.append(f"{topology['slots_used']} of {topology['slots_total']} memory slots in use")
            if topology.get("maximum_supported_gib") is not None:
                parts.append(f"maximum supported capacity reported as {topology['maximum_supported_gib']:.1f} GiB")
            message = "Memory: " + "; ".join(parts) + "."
            if "upgrade" in question.lower() or "add" in question.lower():
                if topology.get("slots_free") is not None and topology.get("maximum_supported_gib") is not None:
                    message += " This does not establish compatible module size or type; check the laptop manufacturer's specifications."
                else:
                    message += " I cannot confirm upgrade capacity or free slots from the available system data."
            errors = facts["errors"]
            if errors and amount is None:
                message += " Additional data unavailable: " + "; ".join(errors) + "."
            return message, facts

        if topic == "storage":
            if not facts["available"]:
                return f"I could not read storage details: {facts.get('error', 'data unavailable')}.", facts
            devices = facts["devices"]
            if not devices:
                return "No physical disk devices were reported by lsblk.", facts
            summaries = []
            for device in devices:
                label = device["model"] or device["name"] or "unnamed disk"
                kind = "NVMe" if device["is_nvme"] else (
                    "rotational" if device["is_rotational"] is True else "non-rotational/unknown"
                )
                size = device["size_bytes"]
                size_text = f", {size / 1024**3:.1f} GiB" if isinstance(size, (int, float)) else ""
                summaries.append(f"{label}{size_text} ({kind})")
            return "Storage: " + "; ".join(summaries) + ".", facts

        if not facts["available"]:
            return f"I could not read GPU details: {facts.get('error', 'data unavailable')}.", facts
        if not facts["devices"]:
            return "No display or 3D controller was reported by lspci.", facts
        return "GPU/display hardware: " + "; ".join(facts["devices"]) + ".", facts


def _ollama_request(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach local Ollama at {OLLAMA_URL}: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
        raise RuntimeError("Ollama returned an invalid chat response")
    return result


def answer_with_ollama(assistant: HardwareAssistant, question: str, model: str) -> str:
    """Ask a local Ollama model to select one fixed inspection tool."""
    expected_topic = assistant.topic_for(question)
    if expected_topic is None:
        return assistant.answer(question)[0]

    tools = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Read normalized {topic} facts from this machine.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }
        for tool_name, topic in OLLAMA_TOOL_TOPICS.items()
    ]
    response = _ollama_request(
        {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Select exactly one provided inspection tool for the user's hardware question. "
                        "Do not answer from memory and do not request arguments."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "tools": tools,
        }
    )
    message = response["message"]
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list) or len(calls) != 1:
        raise RuntimeError("Local model did not select exactly one inspection tool; no command was run")
    if not isinstance(calls[0], dict) or not isinstance(calls[0].get("function"), dict):
        raise RuntimeError("Local model returned a malformed tool call; no command was run")
    function = calls[0]["function"]
    tool_name = function.get("name")
    if not isinstance(tool_name, str):
        raise RuntimeError("Local model returned an invalid tool name; no command was run")
    topic = OLLAMA_TOOL_TOPICS.get(tool_name)
    arguments = function.get("arguments", {})
    if topic != expected_topic:
        raise RuntimeError("Local model selected a tool unrelated to the question; no command was run")
    if arguments not in ({}, None):
        raise RuntimeError("Inspection tools accept no arguments; no command was run")

    assistant.inspect(topic)
    # Use the same evidence-based response path in both modes; generated prose
    # is intentionally not trusted to restate machine facts.
    return assistant.answer(question)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask read-only questions about this Linux machine's hardware.")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    parser.add_argument("--json", action="store_true", help="print normalized inspection facts instead of prose")
    parser.add_argument(
        "--ollama-model",
        metavar="MODEL",
        help="use a local Ollama model to select an allow-listed inspection tool",
    )
    args = parser.parse_args(argv)
    assistant = HardwareAssistant()

    def respond(question: str) -> bool:
        if args.json:
            answer, facts = assistant.answer(question)
            if facts is not None:
                print(json.dumps(facts, indent=2))
            else:
                print(answer)
            return True
        try:
            answer = answer_with_ollama(assistant, question, args.ollama_model) if args.ollama_model else None
        except RuntimeError as exc:
            print(f"Local LLM error: {exc}")
            return False
        if answer is None:
            answer = assistant.answer(question)[0]
        print(answer)
        return True

    if args.ask:
        return 0 if respond(args.ask) else 1

    if args.json:
        parser.error("--json requires --ask")
    if not sys.stdin.isatty():
        parser.error("provide --ask QUESTION for non-interactive use")
    print("Local Hardware Assistant (type 'quit' to exit; no data is sent over the network)")
    while True:
        try:
            question = input("You> ").strip()
        except EOFError:
            print()
            return 0
        if question.lower() in {"quit", "exit"}:
            return 0
        if question:
            respond(question)


if __name__ == "__main__":
    raise SystemExit(main())

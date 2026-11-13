import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hardware_assistant as assistant_module
from hardware_assistant import (
    HardwareAssistant,
    _parse_lscpu,
    _parse_memory_topology,
    answer_with_ollama,
    run_read_only_tool,
)


class ParserTests(unittest.TestCase):
    def test_lscpu_normalizes_cpu_fields(self):
        parsed = _parse_lscpu(
            "Architecture:                    x86_64\n"
            "CPU(s):                          8\n"
            "Model name:                      Example CPU\n"
            "Core(s) per socket:              4\n"
            "Socket(s):                       1\n"
        )
        self.assertEqual(parsed["architecture"], "x86_64")
        self.assertEqual(parsed["logical_cpu_count"], 8)
        self.assertEqual(parsed["model"], "Example CPU")

    def test_memory_parser_reports_slots_and_capacity(self):
        parsed = _parse_memory_topology(
            "Physical Memory Array\n\tMaximum Capacity: 64 GB\n"
            "Memory Device\n\tSize: 16 GB\n\tSpeed: 3200 MT/s\n"
            "Memory Device\n\tSize: No Module Installed\n"
        )
        self.assertEqual(parsed["slots_total"], 2)
        self.assertEqual(parsed["slots_used"], 1)
        self.assertEqual(parsed["slots_free"], 1)
        self.assertEqual(parsed["installed_physical_gib"], 16)
        self.assertEqual(parsed["maximum_supported_gib"], 64)

    def test_storage_parser_marks_nvme(self):
        output = json.dumps(
            {"blockdevices": [{"name": "nvme0n1", "type": "disk", "size": 512 * 1024**3, "tran": "nvme"}]}
        )
        with patch.object(assistant_module, "run_read_only_tool", return_value={"available": True, "stdout": output}):
            facts = HardwareAssistant().inspect("storage")
        self.assertTrue(facts["devices"][0]["is_nvme"])
        self.assertEqual(facts["devices"][0]["size_bytes"], 512 * 1024**3)


class SafetyAndConversationTests(unittest.TestCase):
    def test_only_allowlisted_tools_can_run(self):
        with self.assertRaises(ValueError):
            run_read_only_tool("arbitrary_command; rm -rf /")

    def test_command_is_executed_without_a_shell(self):
        completed = type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        with patch.object(assistant_module.shutil, "which", return_value="/usr/bin/lscpu"):
            with patch.object(assistant_module.subprocess, "run", return_value=completed) as run:
                result = run_read_only_tool("cpu")
        self.assertTrue(result["available"])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.args[0], ["/usr/bin/lscpu"])

    def test_follow_up_questions_route_to_different_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            (proc_root / "meminfo").write_text("MemTotal:       16777216 kB\n", encoding="utf-8")
            assistant = HardwareAssistant(proc_root)
            with patch.object(
                assistant_module,
                "run_read_only_tool",
                side_effect=[
                    {"available": False, "error": "dmidecode is not installed"},
                    {"available": True, "stdout": "00:02.0 VGA compatible controller: Example GPU"},
                ],
            ) as run:
                memory_answer, _ = assistant.answer("How much RAM do I have?")
                gpu_answer, _ = assistant.answer("And what about the GPU?")
        self.assertIn("16.0 GiB detected", memory_answer)
        self.assertIn("Example GPU", gpu_answer)
        self.assertEqual([call.args[0] for call in run.call_args_list], ["memory_topology", "gpu"])

    def test_missing_upgrade_facts_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            (proc_root / "meminfo").write_text("MemTotal:       8388608 kB\n", encoding="utf-8")
            with patch.object(
                assistant_module,
                "run_read_only_tool",
                return_value={"available": False, "error": "dmidecode permission denied"},
            ):
                answer, _ = HardwareAssistant(proc_root).answer("Can I add more RAM?")
        self.assertIn("cannot confirm upgrade capacity or free slots", answer)

    def test_local_model_can_select_only_the_matching_fixed_tool(self):
        assistant = HardwareAssistant()
        response = {
            "message": {
                "tool_calls": [
                    {"function": {"name": "inspect_gpu", "arguments": {}}}
                ]
            }
        }
        with patch.object(assistant_module, "_ollama_request", return_value=response) as request:
            with patch.object(
                assistant_module,
                "run_read_only_tool",
                return_value={"available": True, "stdout": "00:02.0 VGA compatible controller: Example GPU"},
            ):
                answer = answer_with_ollama(assistant, "What GPU do I have?", "local-model")
        self.assertIn("Example GPU", answer)
        payload = request.call_args.args[0]
        self.assertEqual(payload["model"], "local-model")
        self.assertEqual(payload["messages"][0]["role"], "system")

    def test_local_model_mismatched_tool_is_rejected_before_inspection(self):
        assistant = HardwareAssistant()
        response = {
            "message": {
                "tool_calls": [
                    {"function": {"name": "inspect_storage", "arguments": {}}}
                ]
            }
        }
        with patch.object(assistant_module, "_ollama_request", return_value=response):
            with patch.object(assistant_module, "run_read_only_tool") as run:
                with self.assertRaisesRegex(RuntimeError, "unrelated"):
                    answer_with_ollama(assistant, "What GPU do I have?", "local-model")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

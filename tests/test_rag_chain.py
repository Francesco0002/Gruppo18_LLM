from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chain import parse_model_answer, parse_used_source_indexes, strip_model_thinking  # noqa: E402


class RagChainTests(unittest.TestCase):
    def test_strip_model_thinking_removes_qwen_reasoning_block(self) -> None:
        answer = """
<think>
Devo ragionare internamente prima di rispondere.
</think>

La risposta finale.
FONTI_USATE: [1]
"""

        self.assertEqual(
            strip_model_thinking(answer),
            "La risposta finale.\nFONTI_USATE: [1]",
        )

    def test_parse_used_source_indexes_after_thinking_cleanup(self) -> None:
        clean_answer, indexes = parse_used_source_indexes(
            strip_model_thinking(
                """
<think>Ragionamento interno.</think>
Risposta basata sul documento.
FONTI_USATE: [2, 3]
"""
            )
        )

        self.assertEqual(clean_answer, "Risposta basata sul documento.")
        self.assertEqual(indexes, [2, 3])

    def test_parse_model_answer_reads_json_mode(self) -> None:
        clean_answer, indexes = parse_model_answer(
            """
{
  "answer": "Risposta con fonte [1].",
  "used_sources": [1],
  "inline_citations": [1],
  "no_answer_reason": ""
}
"""
        )

        self.assertEqual(clean_answer, "Risposta con fonte [1].")
        self.assertEqual(indexes, [1])


if __name__ == "__main__":
    unittest.main()

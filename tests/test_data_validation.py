from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sms_spam_ham_analysis.data import DatasetValidationError, load_and_validate_dataset


class DatasetValidationTest(unittest.TestCase):
    def test_rejects_unexpected_labels_without_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "messages.tsv"
            output = root / "validated.csv"
            summary = root / "summary.json"
            raw.write_text("ham\thello\nunknown\tsuspicious\n", encoding="utf-8")

            with self.assertRaisesRegex(DatasetValidationError, "unexpected label values"):
                load_and_validate_dataset(raw, output, summary)

            self.assertFalse(output.exists())
            self.assertFalse(summary.exists())

    def test_duplicate_removal_reports_original_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "messages.tsv"
            output = root / "validated.csv"
            summary = root / "summary.json"
            raw.write_text(
                "ham\tsame message\nham\tsame message\nspam\tclaim prize\n",
                encoding="utf-8",
            )

            result = load_and_validate_dataset(
                raw,
                output,
                summary,
                drop_duplicate_messages=True,
            )

            self.assertEqual(result.raw_rows, 3)
            self.assertEqual(result.validated_rows, 2)
            self.assertEqual(result.duplicate_message_count, 2)
            self.assertEqual(result.duplicate_message_groups, 1)
            self.assertEqual(result.duplicate_messages_removed, 1)


if __name__ == "__main__":
    unittest.main()

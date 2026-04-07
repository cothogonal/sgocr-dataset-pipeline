import unittest

from sgocr.verify import contains_anchor_phrase, validate_generated_items


class TestVerify(unittest.TestCase):
    def test_contains_anchor_phrase(self) -> None:
        self.assertTrue(contains_anchor_phrase("What does the text in the top left area of the image say?", ["upper-left area of the image", "top left area of the image"]))
        self.assertFalse(contains_anchor_phrase("What does the sign say?", ["upper-left area of the image"]))

    def test_validate_generated_items(self) -> None:
        tuple_row = {
            "answer": "OPEN 24 HOURS",
            "anchor_label": "upper-left area of the image",
            "anchor_synonyms": ["upper-left area of the image", "top left area of the image"],
        }
        items = [
            {"question": "What does the text in the top left area of the image say?", "answer": "OPEN 24 HOURS"},
            {"question": "Which words appear in the upper-left area of the image?", "answer": "OPEN 24 HOURS"},
        ]
        validations, summary = validate_generated_items(tuple_row, items)
        self.assertEqual(len(validations), 2)
        self.assertTrue(all(v.accepted for v in validations))
        self.assertEqual(summary["accepted_count"], 2)


if __name__ == "__main__":
    unittest.main()

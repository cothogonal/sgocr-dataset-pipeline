import unittest

from sgocr.teacher.prompts import PROMPT_VARIANTS, render_prompt, response_schema


class TestPrompts(unittest.TestCase):
    def test_schema_question_count(self) -> None:
        schema = response_schema(2)
        self.assertEqual(schema["properties"]["items"]["minItems"], 2)
        self.assertEqual(schema["properties"]["items"]["maxItems"], 2)

    def test_render_prompt_mentions_answer_and_anchor(self) -> None:
        row = {
            "answer": "HELLO",
            "anchor_label": "center of the image",
            "anchor_synonyms": ["center of the image", "middle of the image"],
        }
        prompt = render_prompt(row, PROMPT_VARIANTS["strict_2q"])
        self.assertIn("HELLO", prompt)
        self.assertIn("center of the image", prompt)
        self.assertIn("Write exactly 2 natural questions.", prompt)


if __name__ == "__main__":
    unittest.main()

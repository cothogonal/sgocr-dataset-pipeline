import unittest

from sgocr.stages import PIPELINE_STAGES, STAGE_INDEX


class TestStages(unittest.TestCase):
    def test_stage_keys_are_complete_and_ordered(self) -> None:
        self.assertEqual([stage.key for stage in PIPELINE_STAGES], ["0", "A", "B", "C", "D", "E", "F", "G"])

    def test_stage_index_matches_stage_list(self) -> None:
        self.assertEqual(set(STAGE_INDEX.keys()), {"0", "A", "B", "C", "D", "E", "F", "G"})
        self.assertEqual(STAGE_INDEX["E"].title, "Teacher QA Generation")
        self.assertIn("raw_qa.jsonl", STAGE_INDEX["E"].outputs)

    def test_every_stage_declares_code_scope_and_outputs(self) -> None:
        for stage in PIPELINE_STAGES:
            self.assertGreater(len(stage.code_scope), 0)
            self.assertGreater(len(stage.outputs), 0)


if __name__ == "__main__":
    unittest.main()

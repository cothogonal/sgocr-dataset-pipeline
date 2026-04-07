import os
import unittest

from sgocr.secrets import ANTHROPIC, GEMINI, OPENAI, MissingSecretError, get_secret, redact_secret


class TestSecrets(unittest.TestCase):
    def test_reads_expected_env_var(self) -> None:
        key = "GEMINI_API_KEY"
        old = os.environ.get(key)
        os.environ[key] = "secret-value"
        try:
            self.assertEqual(get_secret(GEMINI), "secret-value")
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    def test_missing_secret_error_names_env_var_not_value(self) -> None:
        key = "OPENAI_API_KEY"
        old = os.environ.pop(key, None)
        try:
            with self.assertRaises(MissingSecretError) as ctx:
                get_secret(OPENAI)
            msg = str(ctx.exception)
            self.assertIn("OPENAI_API_KEY", msg)
            self.assertNotIn("sk-", msg)
            self.assertNotIn("secret", msg.lower())
        finally:
            if old is not None:
                os.environ[key] = old

    def test_redaction_helper_never_returns_input_value(self) -> None:
        self.assertEqual(redact_secret("top-secret"), "***REDACTED***")
        self.assertEqual(redact_secret("anthropic-key"), "***REDACTED***")
        self.assertEqual(redact_secret(None), "<missing>")

    def test_supported_providers_are_explicit(self) -> None:
        self.assertEqual(GEMINI.env_var, "GEMINI_API_KEY")
        self.assertEqual(OPENAI.env_var, "OPENAI_API_KEY")
        self.assertEqual(ANTHROPIC.env_var, "ANTHROPIC_API_KEY")


if __name__ == "__main__":
    unittest.main()

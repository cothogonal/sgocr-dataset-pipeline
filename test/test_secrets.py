import os
import unittest

from sgocr.secrets import ANTHROPIC, GEMINI, OPENAI, MissingSecretError, get_secret, missing_secret_env_vars, redact_secret


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

    def test_missing_secret_env_vars_reports_only_missing_names(self) -> None:
        saved_gemini = os.environ.pop("GEMINI_API_KEY", None)
        saved_openai = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "present"
        try:
            self.assertEqual(missing_secret_env_vars([GEMINI, OPENAI]), ["GEMINI_API_KEY"])
        finally:
            if saved_gemini is not None:
                os.environ["GEMINI_API_KEY"] = saved_gemini
            if saved_openai is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = saved_openai


if __name__ == "__main__":
    unittest.main()

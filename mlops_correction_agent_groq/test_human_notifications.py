import os
import smtplib
import unittest
from unittest.mock import patch

from human_notifications import send_human_review_email


REVIEW = {
    "incident_id": "MAIL-001",
    "incident_type": "qualite_donnees",
    "confidence": 0.4,
    "cause_probable": "data_corruption",
    "description": "Valeurs anormales detectees.",
    "evidence_ids": ["E0001"],
    "reason": "La confiance est sous le seuil.",
    "review_url": "http://127.0.0.1:5000/human-review/test?token=opaque-token",
}


class HumanNotificationTests(unittest.TestCase):
    def test_missing_smtp_configuration_keeps_request_pending(self):
        with patch.dict(os.environ, {}, clear=True):
            result = send_human_review_email(REVIEW)

        self.assertEqual(result["status"], "not_configured")
        self.assertIn("HUMAN_REVIEW_EMAIL", result["missing_settings"])

    @patch("human_notifications.smtplib.SMTP")
    def test_sends_review_email_over_starttls(self, smtp_class):
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.example.test",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "sender@example.test",
            "SMTP_PASSWORD": "test-password",
            "SMTP_FROM": "MLOps Review <sender@example.test>",
            "HUMAN_REVIEW_EMAIL": "reviewer@example.test",
        }, clear=True):
            result = send_human_review_email(REVIEW)

        self.assertEqual(result, {"status": "sent", "recipient": "reviewer@example.test"})
        smtp = smtp_class.return_value.__enter__.return_value
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("sender@example.test", "test-password")
        email = smtp.send_message.call_args.args[0]
        self.assertIn("MAIL-001", email.get_content())
        self.assertIn("reponse au courriel n'est pas traitee automatiquement", email.get_content())
        self.assertIn(REVIEW["review_url"], email.get_content())

    @patch("human_notifications.smtplib.SMTP")
    def test_returns_server_details_for_smtp_data_rejection(self, smtp_class):
        smtp = smtp_class.return_value.__enter__.return_value
        smtp.send_message.side_effect = smtplib.SMTPDataError(554, b"Message rejected by policy")

        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.example.test",
            "SMTP_USERNAME": "sender@example.test",
            "SMTP_PASSWORD": "test-password",
            "HUMAN_REVIEW_EMAIL": "reviewer@example.test",
        }, clear=True):
            result = send_human_review_email(REVIEW)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["smtp_code"], 554)
        self.assertEqual(result["smtp_message"], "Message rejected by policy")


if __name__ == "__main__":
    unittest.main()
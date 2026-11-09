"""Best-effort email notification for incidents requiring human review."""
from email.message import EmailMessage
import os
import smtplib
import ssl


def send_human_review_email(review):
    required = ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "HUMAN_REVIEW_EMAIL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        return {"status": "not_configured", "missing_settings": missing}

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        message = EmailMessage()
        message["Subject"] = f"[MLOps] Revue humaine requise: {review['incident_id']}"
        message["From"] = os.getenv("SMTP_FROM") or os.environ["SMTP_USERNAME"]
        message["To"] = os.environ["HUMAN_REVIEW_EMAIL"]
        message.set_content(
            "Une validation humaine est requise. Aucune action de production n'a ete executee.\n\n"
            f"Incident: {review['incident_id']}\n"
            f"Type: {review['incident_type']}\n"
            f"Cause proposee: {review['cause_probable']}\n"
            f"Confiance: {review['confidence']}\n"
            f"Preuves: {', '.join(review['evidence_ids']) or 'aucune'}\n"
            f"Motif de revue: {review['reason']}\n"
            f"Resume: {review['description']}\n\n"
            "Cette notification ne vaut pas approbation. La reponse au courriel n'est pas traitee automatiquement.\n\n"
            + (f"Ouvrir la demande et choisir Approuver ou Refuser : {review['review_url']}"
               if review.get("review_url") else "")
        )

        with smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
            server.send_message(message)

        return {"status": "sent", "recipient": message["To"]}
    except smtplib.SMTPResponseException as exc:
        response_message = exc.smtp_error
        if isinstance(response_message, bytes):
            response_message = response_message.decode("utf-8", errors="replace")
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "smtp_code": exc.smtp_code,
            "smtp_message": str(response_message),
        }
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__}
"""Allowlisted email sender for FR-22 actions."""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECIPIENTS_PATH = ROOT / "config" / "email_recipients.json"


def load_recipients() -> dict[str, str]:
    if not RECIPIENTS_PATH.exists():
        return {}
    data = json.loads(RECIPIENTS_PATH.read_text())
    if not isinstance(data, dict):
        return {}
    return {str(alias).lower(): str(email) for alias, email in data.items()}


def send_email_action(action: dict[str, Any]) -> dict[str, Any]:
    recipients = load_recipients()
    alias = str(action.get("recipient_alias") or "").lower().strip()
    if alias not in recipients:
        return {
            "type": "send_email",
            "status": "blocked",
            "recipient_alias": alias,
            "message": f"Recipient alias is not configured: {alias}",
        }

    missing = [
        name
        for name in ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "EMAIL_FROM"]
        if not os.getenv(name)
    ]
    if missing:
        return {
            "type": "send_email",
            "status": "not_configured",
            "recipient_alias": alias,
            "to": recipients[alias],
            "message": f"Email not sent. Missing env vars: {', '.join(missing)}",
        }

    message = EmailMessage()
    message["From"] = os.environ["EMAIL_FROM"]
    message["To"] = recipients[alias]
    message["Subject"] = str(action.get("subject") or "Fantasy draft note")
    message.set_content(str(action.get("body") or ""))

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(message)

    return {
        "type": "send_email",
        "status": "sent",
        "recipient_alias": alias,
        "to": recipients[alias],
        "subject": message["Subject"],
    }


def execute_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for action in actions:
        if action.get("type") == "send_email":
            results.append(send_email_action(action))
    return results

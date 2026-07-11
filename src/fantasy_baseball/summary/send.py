"""Send the rendered summary via the Resend transactional email API."""

from __future__ import annotations

import resend


def send_email(
    *,
    api_key: str,
    from_address: str,
    recipients: list[str],
    subject: str,
    html: str,
    text: str,
) -> str:
    """Send one email to ``recipients``. Returns the Resend message id.

    Raises RuntimeError if Resend returns no id (treated as a send failure).
    """
    resend.api_key = api_key
    result = resend.Emails.send(
        {
            "from": from_address,
            "to": recipients,
            "subject": subject,
            "html": html,
            "text": text,
        }
    )
    msg_id = result.get("id") if isinstance(result, dict) else None
    if not msg_id:
        raise RuntimeError(f"Resend send failed; response={result!r}")
    return str(msg_id)

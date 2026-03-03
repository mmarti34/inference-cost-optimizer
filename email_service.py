"""
Email service using Resend for transactional emails.
Requires RESEND_API_KEY environment variable.
"""
import os
import logging

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("OPTIML_FROM_EMAIL", "OptiML <noreply@optiml.one>")
FRONTEND_URL = os.getenv("OPTIML_FRONTEND_URL", "https://optiml.one")

# Lazy init: only import and configure resend when actually sending
_resend_configured = False


def _ensure_resend():
    global _resend_configured
    if _resend_configured:
        return True
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — email sending disabled.")
        return False
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        _resend_configured = True
        logger.info("Resend email service initialized.")
        return True
    except ImportError:
        logger.error("resend package not installed. Run: pip install resend")
        return False


def send_invite_email(
    to_email: str,
    org_name: str,
    inviter_email: str,
    invite_token: str,
) -> bool:
    """
    Send an organization invite email.
    Returns True if sent successfully, False otherwise.
    """
    if not _ensure_resend():
        return False

    import resend

    accept_url = f"{FRONTEND_URL}/invite/accept?token={invite_token}"

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#f4f4f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f5;padding:40px 20px;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
        <tr><td style="padding:32px 32px 24px;text-align:center;">
          <img src="https://optiml.one/optimllogodark.png" alt="OptiML" height="32" style="display:block;margin:0 auto;">
        </td></tr>
        <tr><td style="padding:0 32px 24px;">
          <h1 style="margin:0 0 8px;font-size:20px;font-weight:600;color:#18181b;">
            You've been invited to join {org_name}
          </h1>
          <p style="margin:0 0 24px;font-size:14px;color:#71717a;line-height:1.5;">
            {inviter_email} has invited you to collaborate on OptiML.
            Click below to accept the invitation.
          </p>
          <a href="{accept_url}"
             style="display:inline-block;background-color:#6C8EEF;color:#ffffff;text-decoration:none;padding:12px 32px;border-radius:8px;font-size:14px;font-weight:500;">
            Accept Invitation
          </a>
        </td></tr>
        <tr><td style="padding:24px 32px;border-top:1px solid #e4e4e7;">
          <p style="margin:0;font-size:12px;color:#a1a1aa;line-height:1.5;">
            This invitation expires in 7 days. If you didn't expect this email, you can safely ignore it.
          </p>
          <p style="margin:8px 0 0;font-size:12px;color:#a1a1aa;">
            <a href="{accept_url}" style="color:#6C8EEF;word-break:break-all;">{accept_url}</a>
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    try:
        params: resend.Emails.SendParams = {
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": f"You've been invited to {org_name} on OptiML",
            "html": html_body,
        }
        response = resend.Emails.send(params)
        logger.info("Invite email sent to %s (id=%s)", to_email, getattr(response, 'id', response))
        return True
    except Exception as e:
        logger.error("Failed to send invite email to %s: %s", to_email, e)
        return False

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
<body style="margin:0;padding:0;background-color:#0d1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0d1117;padding:48px 20px;">
    <tr><td align="center">
      <!-- Main card — dark glass style matching landing page -->
      <table width="480" cellpadding="0" cellspacing="0" style="background-color:#161b22;border-radius:20px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);box-shadow:0 4px 24px rgba(0,0,0,0.3),0 2px 6px rgba(0,0,0,0.15);">
        <!-- Logo -->
        <tr><td style="padding:36px 36px 8px;text-align:center;">
          <img src="https://optiml.one/optimllogodark.png" alt="OptiML" height="36" style="display:block;margin:0 auto;">
        </td></tr>
        <!-- Gradient accent line -->
        <tr><td style="padding:16px 36px 0;">
          <div style="height:2px;border-radius:1px;background:linear-gradient(135deg,#6C8EEF 0%,#4EBF98 50%,#A97CF8 100%);"></div>
        </td></tr>
        <!-- Content -->
        <tr><td style="padding:24px 36px 32px;">
          <h1 style="margin:0 0 8px;font-size:22px;font-weight:500;color:#e6edf3;letter-spacing:-0.02em;">
            you've been invited to join {org_name}
          </h1>
          <p style="margin:0 0 28px;font-size:14px;color:#8b949e;line-height:1.6;">
            {inviter_email} has invited you to collaborate on optiml. click below to accept and get started.
          </p>
          <a href="{accept_url}"
             style="display:inline-block;background:linear-gradient(135deg,#6C8EEF,#7B9CF2);color:#ffffff;text-decoration:none;padding:12px 36px;border-radius:12px;font-size:14px;font-weight:500;letter-spacing:-0.01em;box-shadow:0 2px 12px rgba(108,142,239,0.3);">
            accept invitation
          </a>
        </td></tr>
        <!-- Footer -->
        <tr><td style="padding:20px 36px 28px;border-top:1px solid rgba(255,255,255,0.06);">
          <p style="margin:0;font-size:12px;color:#6e7681;line-height:1.5;">
            this invitation expires in 7 days. if you didn't expect this email, you can safely ignore it.
          </p>
          <p style="margin:8px 0 0;font-size:11px;color:#6e7681;">
            <a href="{accept_url}" style="color:#6C8EEF;word-break:break-all;text-decoration:none;">{accept_url}</a>
          </p>
        </td></tr>
      </table>
      <!-- Brand footer -->
      <table width="480" cellpadding="0" cellspacing="0">
        <tr><td style="padding:24px 36px 0;text-align:center;">
          <p style="margin:0;font-size:11px;color:#484f58;letter-spacing:-0.01em;">
            &copy; 2026 optiml. all rights reserved.
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
            "subject": f"you've been invited to {org_name} on optiml",
            "html": html_body,
        }
        response = resend.Emails.send(params)
        logger.info("Invite email sent to %s (id=%s)", to_email, getattr(response, 'id', response))
        return True
    except Exception as e:
        logger.error("Failed to send invite email to %s: %s", to_email, e)
        return False

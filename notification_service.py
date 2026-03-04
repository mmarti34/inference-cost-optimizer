"""
Notification service for dispatching production alerts to configured channels.
Supports Slack webhooks, generic webhooks (with optional HMAC), and email via Resend.
"""
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from supabase_client import supabase
from email_service import _ensure_resend, FROM_EMAIL, FRONTEND_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_alert_payload(
    alert_type: str,
    org_id: str,
    endpoint_slug: str,
    details: dict,
) -> dict:
    """Build a standardised alert payload dict."""
    return {
        "alert_type": alert_type,
        "org_id": org_id,
        "endpoint_slug": endpoint_slug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Channel dispatchers
# ---------------------------------------------------------------------------

_ALERT_TYPE_LABELS = {
    "rollback_triggered": "Rollback triggered",
    "rule_triggered": "Alert rule triggered",
    "error_rate_spike": "Error rate spike",
    "deployment_failed": "Deployment failed",
}


async def _send_slack_webhook(webhook_url: str, payload: dict) -> bool:
    """POST a Slack Block Kit message to an incoming webhook URL."""
    if not webhook_url:
        return False

    alert_label = _ALERT_TYPE_LABELS.get(payload.get("alert_type", ""), payload.get("alert_type", "alert"))
    endpoint = payload.get("endpoint_slug", "—")
    details = payload.get("details", {})

    header_text = f":rotating_light: *{alert_label}* on `{endpoint}`"
    detail_lines = "\n".join(f"• *{k}*: {v}" for k, v in details.items() if k != "conditions")

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header_text}},
    ]
    if detail_lines:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": detail_lines}})

    conditions = details.get("conditions")
    if conditions and isinstance(conditions, list):
        cond_text = "\n".join(f"  – {c.get('metric', '?')} {c.get('operator', '?')} {c.get('threshold', '?')}" for c in conditions)
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Conditions:*\n{cond_text}"}})

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"_optiml · {payload.get('timestamp', '')} UTC_"}],
    })

    slack_body = {"text": f"{alert_label} on {endpoint}", "blocks": blocks}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(webhook_url, json=slack_body)
        if resp.status_code != 200:
            logger.warning("Slack webhook returned %s: %s", resp.status_code, resp.text[:200])
        return resp.status_code == 200


async def _send_generic_webhook(
    url: str,
    payload: dict,
    headers: Optional[dict] = None,
    secret: Optional[str] = None,
) -> bool:
    """POST alert payload to a generic webhook URL with optional HMAC signature."""
    if not url:
        return False

    send_headers = {"Content-Type": "application/json"}
    if headers:
        send_headers.update(headers)

    body_bytes = json.dumps(payload, separators=(",", ":")).encode()

    if secret:
        sig = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        send_headers["X-OptiML-Signature"] = sig

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, content=body_bytes, headers=send_headers)
        if not (200 <= resp.status_code < 300):
            logger.warning("Webhook %s returned %s", url[:60], resp.status_code)
        return 200 <= resp.status_code < 300


def _send_alert_email(recipients: list, payload: dict) -> bool:
    """Send an alert email using the existing Resend setup."""
    if not recipients:
        return False
    if not _ensure_resend():
        return False

    import resend

    alert_type = payload.get("alert_type", "alert")
    alert_label = _ALERT_TYPE_LABELS.get(alert_type, alert_type)
    endpoint = payload.get("endpoint_slug", "—")
    details = payload.get("details", {})
    timestamp = payload.get("timestamp", "")

    detail_rows = ""
    for k, v in details.items():
        if k == "conditions":
            continue
        detail_rows += (
            f'<tr>'
            f'<td style="padding:6px 12px;font-size:13px;color:#71717a;">{k}</td>'
            f'<td style="padding:6px 12px;font-size:13px;color:#18181b;">{v}</td>'
            f'</tr>'
        )

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background-color:#f8f9fb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f8f9fb;padding:48px 20px;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:20px;overflow:hidden;border:1px solid #e8e8ec;box-shadow:0 4px 24px rgba(108,130,210,0.08),0 2px 6px rgba(0,0,0,0.04);">
        <tr><td style="padding:36px 36px 8px;text-align:center;">
          <span style="font-size:28px;font-weight:700;color:#18181b;letter-spacing:-0.03em;">optiml</span>
        </td></tr>
        <tr><td style="padding:16px 36px 0;">
          <table width="100%" cellpadding="0" cellspacing="0"><tr>
            <td width="33%" style="height:2px;background-color:#EF6C6C;"></td>
            <td width="34%" style="height:2px;background-color:#F5A623;"></td>
            <td width="33%" style="height:2px;background-color:#EF6C6C;"></td>
          </tr></table>
        </td></tr>
        <tr><td style="padding:24px 36px 32px;">
          <h1 style="margin:0 0 8px;font-size:22px;font-weight:500;color:#18181b;letter-spacing:-0.02em;">
            {alert_label}
          </h1>
          <p style="margin:0 0 20px;font-size:14px;color:#71717a;line-height:1.6;">
            endpoint: <strong>{endpoint}</strong>
          </p>
          <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e8e8ec;border-radius:8px;overflow:hidden;">
            {detail_rows}
          </table>
        </td></tr>
        <tr><td style="padding:20px 36px 28px;border-top:1px solid #e8e8ec;">
          <p style="margin:0;font-size:12px;color:#a1a1aa;line-height:1.5;">
            {timestamp}
          </p>
        </td></tr>
      </table>
      <table width="480" cellpadding="0" cellspacing="0">
        <tr><td style="padding:24px 36px 0;text-align:center;">
          <p style="margin:0;font-size:11px;color:#a1a1aa;">&copy; 2026 optiml</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    try:
        params: resend.Emails.SendParams = {
            "from": FROM_EMAIL,
            "to": recipients,
            "subject": f"[optiml] {alert_label} on {endpoint}",
            "html": html_body,
        }
        resend.Emails.send(params)
        return True
    except Exception as e:
        logger.error("Alert email failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------

async def dispatch_alert(
    org_id: str,
    alert_type: str,
    endpoint_slug: str,
    details: dict,
) -> list:
    """
    Send alert to all enabled channels for the org.
    Records each dispatch attempt in alert_history.
    Returns list of {channel_id, status, error}.
    """
    payload = build_alert_payload(alert_type, org_id, endpoint_slug, details)

    try:
        result = (
            supabase.table("alert_channels")
            .select("id, channel_type, config, name")
            .eq("org_id", org_id)
            .eq("enabled", True)
            .execute()
        )
        channels = result.data or []
    except Exception as e:
        logger.error("Failed to fetch alert channels for org %s: %s", org_id, e)
        return []

    if not channels:
        return []

    results = []

    for ch in channels:
        ch_id = str(ch["id"])
        ch_type = ch.get("channel_type", "")
        config = ch.get("config") or {}
        status = "sent"
        error_msg = None

        try:
            if ch_type == "slack_webhook":
                ok = await _send_slack_webhook(config.get("webhook_url", ""), payload)
                if not ok:
                    status, error_msg = "failed", "Non-200 response from Slack"

            elif ch_type == "webhook":
                ok = await _send_generic_webhook(
                    config.get("url", ""),
                    payload,
                    config.get("headers"),
                    config.get("secret"),
                )
                if not ok:
                    status, error_msg = "failed", "Non-2xx response"

            elif ch_type == "email":
                ok = _send_alert_email(config.get("recipients", []), payload)
                if not ok:
                    status, error_msg = "failed", "Email send failed"

            else:
                status, error_msg = "failed", f"Unknown channel type: {ch_type}"

        except Exception as e:
            status = "failed"
            error_msg = str(e)[:500]
            logger.error("Alert dispatch error for channel %s: %s", ch_id, e)

        # Record in alert_history (best-effort)
        try:
            supabase.table("alert_history").insert({
                "org_id": org_id,
                "channel_id": ch_id,
                "alert_type": alert_type,
                "endpoint_slug": endpoint_slug,
                "payload": payload,
                "delivery_status": status,
                "error_message": error_msg,
            }).execute()
        except Exception as e:
            logger.error("Failed to record alert_history: %s", e)

        results.append({"channel_id": ch_id, "status": status, "error": error_msg})

    return results

"""
CRUD endpoints for alert notification channels.
Requires authenticated org member for all operations.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from supabase_client import supabase
from auth_dependency import require_org_member, AuthenticatedUser
from notification_service import dispatch_alert

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AlertChannelCreate(BaseModel):
    org_id: str
    channel_type: str  # slack_webhook | webhook | email
    name: str = ""
    config: dict = {}
    enabled: bool = True


class AlertChannelUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[dict] = None
    enabled: Optional[bool] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/alert-channels/{org_id}")
async def list_alert_channels(
    org_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """List all alert channels for an organization."""
    try:
        result = (
            supabase.table("alert_channels")
            .select("id, org_id, channel_type, name, config, enabled, created_at, updated_at")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .execute()
        )
        channels = result.data or []
        # Mask sensitive config values for the response
        for ch in channels:
            ch["config"] = _mask_config(ch.get("channel_type", ""), ch.get("config") or {})
        return channels
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alert channels: {str(e)}")


@router.post("/alert-channels")
async def create_alert_channel(
    data: AlertChannelCreate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Create a new alert channel. Admin only."""
    # Verify requester is admin
    _require_admin(data.org_id, auth_user.user_id)

    if data.channel_type not in ("slack_webhook", "webhook", "email"):
        raise HTTPException(status_code=400, detail="channel_type must be slack_webhook, webhook, or email")

    try:
        result = supabase.table("alert_channels").insert({
            "org_id": data.org_id,
            "channel_type": data.channel_type,
            "name": data.name.strip() or _default_name(data.channel_type),
            "config": data.config,
            "enabled": data.enabled,
        }).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create alert channel")

        row = result.data[0]
        row["config"] = _mask_config(row.get("channel_type", ""), row.get("config") or {})
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating alert channel: {str(e)}")


@router.put("/alert-channels/{org_id}/{channel_id}")
async def update_alert_channel(
    org_id: str,
    channel_id: str,
    data: AlertChannelUpdate,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Update an alert channel. Admin only."""
    _require_admin(org_id, auth_user.user_id)

    update_data = {}
    if data.name is not None:
        update_data["name"] = data.name.strip()
    if data.config is not None:
        update_data["config"] = data.config
    if data.enabled is not None:
        update_data["enabled"] = data.enabled

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        result = (
            supabase.table("alert_channels")
            .update(update_data)
            .eq("id", channel_id)
            .eq("org_id", org_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Alert channel not found")

        row = result.data[0]
        row["config"] = _mask_config(row.get("channel_type", ""), row.get("config") or {})
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating alert channel: {str(e)}")


@router.delete("/alert-channels/{org_id}/{channel_id}")
async def delete_alert_channel(
    org_id: str,
    channel_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Delete an alert channel. Admin only."""
    _require_admin(org_id, auth_user.user_id)

    try:
        existing = (
            supabase.table("alert_channels")
            .select("id")
            .eq("id", channel_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Alert channel not found")

        supabase.table("alert_channels").delete().eq("id", channel_id).eq("org_id", org_id).execute()
        return {"message": "Alert channel deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting alert channel: {str(e)}")


@router.post("/alert-channels/{org_id}/{channel_id}/test")
async def test_alert_channel(
    org_id: str,
    channel_id: str,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """Send a test alert to a single channel."""
    _require_admin(org_id, auth_user.user_id)

    try:
        ch_result = (
            supabase.table("alert_channels")
            .select("id, channel_type, config, name")
            .eq("id", channel_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        if not ch_result.data:
            raise HTTPException(status_code=404, detail="Alert channel not found")

        # Dispatch test alert
        results = await dispatch_alert(
            org_id=org_id,
            alert_type="test",
            endpoint_slug="test-endpoint",
            details={"message": "This is a test alert from optiml."},
        )

        # Find result for this specific channel
        for r in results:
            if r["channel_id"] == channel_id:
                if r["status"] == "sent":
                    return {"status": "sent", "message": "Test alert sent successfully"}
                else:
                    raise HTTPException(status_code=502, detail=f"Test alert failed: {r.get('error', 'unknown')}")

        return {"status": "sent", "message": "Test alert dispatched"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing alert channel: {str(e)}")


@router.get("/alert-history/{org_id}")
async def list_alert_history(
    org_id: str,
    limit: int = 50,
    auth_user: AuthenticatedUser = Depends(require_org_member),
):
    """List recent alert history for an organization."""
    try:
        result = (
            supabase.table("alert_history")
            .select("id, org_id, channel_id, alert_type, endpoint_slug, delivery_status, error_message, created_at")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .limit(min(limit, 100))
            .execute()
        )
        return result.data or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alert history: {str(e)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_admin(org_id: str, user_id: str):
    """Verify the user is an admin of the org. Raises 403 if not."""
    try:
        check = (
            supabase.table("organization_members")
            .select("role")
            .eq("org_id", org_id)
            .eq("user_id", user_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
        if not check.data or check.data[0].get("role") != "admin":
            raise HTTPException(status_code=403, detail="Only admins can manage alert channels.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=403, detail="Unable to verify admin role.")


def _default_name(channel_type: str) -> str:
    return {
        "slack_webhook": "Slack alerts",
        "webhook": "Webhook",
        "email": "Email alerts",
    }.get(channel_type, "Alert channel")


def _mask_config(channel_type: str, config: dict) -> dict:
    """Mask sensitive values in config for API responses."""
    masked = {**config}
    if channel_type == "slack_webhook" and "webhook_url" in masked:
        url = masked["webhook_url"]
        if len(url) > 30:
            masked["webhook_url"] = url[:30] + "..."
    if channel_type == "webhook":
        if "secret" in masked and masked["secret"]:
            masked["secret"] = "••••••••"
        if "url" in masked:
            url = masked["url"]
            if len(url) > 50:
                masked["url"] = url[:50] + "..."
    # Email recipients are not sensitive — leave as-is
    return masked

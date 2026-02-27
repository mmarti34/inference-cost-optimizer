from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase_client import supabase

router = APIRouter()

# Pydantic models
class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    email_notifications_enabled: Optional[bool] = None
    marketing_emails_enabled: Optional[bool] = None

class UserProfileResponse(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None
    locale: Optional[str] = None
    email_notifications_enabled: Optional[bool] = None
    marketing_emails_enabled: Optional[bool] = None
    subscription_tier: Optional[str] = None
    subscription_status: Optional[str] = None
    created_at: str

# User Profile Endpoints
@router.get("/user-profiles/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(user_id: str):
    """Get user profile by user ID"""
    try:
        result = supabase.table("user_profiles").select(
            "user_id, first_name, last_name, display_name, bio, phone, timezone, locale, "
            "email_notifications_enabled, marketing_emails_enabled, subscription_tier, subscription_status, created_at"
        ).eq("user_id", user_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user profile: {str(e)}")

@router.put("/user-profiles/{user_id}", response_model=UserProfileResponse)
async def update_user_profile(user_id: str, profile_data: UserProfileUpdate):
    """Update user profile"""
    try:
        # First check if the profile exists
        existing = supabase.table("user_profiles").select(
            "user_id, first_name, last_name, display_name, bio, phone, timezone, locale, "
            "email_notifications_enabled, marketing_emails_enabled, subscription_tier, subscription_status, created_at"
        ).eq("user_id", user_id).single().execute()
        
        if not existing.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Update only provided fields
        update_data = {k: v for k, v in profile_data.dict().items() if v is not None}
        
        if not update_data:
            return existing.data
        
        result = supabase.table("user_profiles").update(update_data).eq("user_id", user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to update user profile")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user profile: {str(e)}")

@router.post("/user-profiles/{user_id}/password-reset")
async def request_password_reset(user_id: str):
    """Request password reset for user"""
    try:
        # Get user email from auth.users table
        result = supabase.auth.admin.get_user_by_id(user_id)
        
        if not result.user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_email = result.user.email
        
        # Send password reset email
        reset_result = supabase.auth.reset_password_email(user_email)
        
        if reset_result.error:
            raise HTTPException(status_code=500, detail=f"Failed to send password reset email: {reset_result.error.message}")
        
        return {"message": "Password reset email sent successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending password reset email: {str(e)}")

from fastapi import APIRouter, HTTPException, Body
from .supabase_client import supabase

router = APIRouter()

# Plan limits
PLAN_LIMITS = {
    'free': {'orgs': 1, 'members': 1},
    'starter': {'orgs': 1, 'members': 3},
    'team': {'orgs': 1, 'members': 20},
    'pro': {'orgs': 1, 'members': 99999},
    'enterprise': {'orgs': 99999, 'members': 99999},
}

def get_org_plan(org):
    return org.get('plan', 'free')

@router.post('/api/organizations/create')
def create_organization(user_id: str = Body(...), org_name: str = Body(...), plan: str = Body('free')):
    # 1. Fetch user's orgs
    orgs = supabase.table('organizations').select('*').eq('created_by', user_id).execute().data
    # Use the highest plan among user's orgs, or the provided plan
    user_plan = plan
    if orgs:
        user_plan = max([get_org_plan(o) for o in orgs] + [plan])
    org_limit = PLAN_LIMITS.get(user_plan, PLAN_LIMITS['free'])['orgs']
    if len(orgs) >= org_limit:
        raise HTTPException(status_code=403, detail=f'Your plan ({user_plan}) only allows {org_limit} organization(s).')
    # 2. Create org
    new_org = supabase.table('organizations').insert({'name': org_name, 'created_by': user_id, 'plan': user_plan}).execute()
    return new_org.data

@router.post('/api/organizations/invite')
def invite_member(org_id: str = Body(...), email: str = Body(...)):
    # 1. Fetch org and plan
    org = supabase.table('organizations').select('*').eq('id', org_id).single().execute().data
    if not org:
        raise HTTPException(status_code=404, detail='Organization not found.')
    plan = get_org_plan(org)
    members = supabase.table('organization_members').select('*').eq('org_id', org_id).eq('status', 'active').execute().data
    member_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])['members']
    if len(members) >= member_limit:
        raise HTTPException(status_code=403, detail=f'Member limit reached for your plan ({plan}).')
    # 2. Add member
    new_member = supabase.table('organization_members').insert({'org_id': org_id, 'invited_email': email, 'status': 'pending'}).execute()
    return new_member.data 
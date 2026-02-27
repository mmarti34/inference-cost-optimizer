# Cost Optimization Features

This document describes the new cost optimization features added to the inference-cost-optimizer backend.

## New Endpoints

### POST /optimize

Optimizes prompt usage by analyzing recent usage patterns and budget constraints to recommend the best provider/model combination.

#### Request Body
```json
{
  "prompt_id": "123",
  "estimated_input_tokens": 100,
  "estimated_output_tokens": 300,
  "project_id": "abc123",
  "org_id": "org456",
  "user_id": "user789"
}
```

#### Response
```json
{
  "status": "success",
  "recommendation": {
    "recommended_provider": "openai",
    "recommended_model": "gpt-3.5-turbo",
    "estimated_cost_usd": 0.0025,
    "reasoning": "GPT-3.5-turbo provides the best cost-performance ratio for this prompt type while staying within budget constraints."
  }
}
```

#### Logic Flow
1. **Prompt Change Detection**: Compares the two most recent prompts for the given `prompt_id`
2. **Budget Analysis**: Fetches project's monthly budget and calculates current month's usage
3. **AI Recommendation**: Uses GPT-4 to analyze the prompt and recommend optimal provider/model
4. **Storage**: Saves the recommendation to `optimizer_recommendations` table

## Database Changes

### New Table: `optimizer_recommendations`
Stores AI-generated optimization recommendations with the following fields:
- `prompt_id`: Reference to the prompt template
- `project_id`: Reference to the project
- `org_id`: Reference to the organization
- `user_id`: Reference to the user
- `recommended_provider`: Recommended AI provider
- `recommended_model`: Recommended model
- `estimated_cost_usd`: Estimated cost for the recommendation
- `estimated_input_tokens`: Estimated input tokens
- `estimated_output_tokens`: Estimated output tokens
- `full_prompt_text`: The actual prompt text
- `budget_used_usd`: Current month's budget usage
- `monthly_budget_usd`: Project's monthly budget
- `reasoning`: AI's explanation for the recommendation
- `created_at`: Timestamp of the recommendation

### Updated Table: `usage_logs`
Added new columns for project tracking:
- `project_id`: Reference to the project (UUID)
- `org_id`: Reference to the organization (UUID)

### Updated Table: `projects`
Added new column for budget tracking:
- `monthly_budget`: Monthly budget limit in USD (DECIMAL)

## Usage Logging Updates

All router endpoints now include `project_id` and `org_id` in usage logs:
- OpenAI Router
- Anthropic Router
- Mistral Router
- Cohere Router
- Gemini Router

## Setup Instructions

1. **Run Database Migration**:
   ```sql
   -- Execute the migration_add_project_tracking.sql file in your Supabase database
   ```

2. **Environment Variables**:
   Ensure `OPENAI_API_KEY` is set in your environment for the optimization endpoint.

3. **Update Frontend**:
   - Ensure prompt templates include `project_id` when created
   - Update usage analytics to use the new `project_id` field
   - Implement budget tracking UI

## Example Usage

### Frontend Integration
```javascript
// Call the optimization endpoint
const response = await fetch('/optimize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt_id: 'your-prompt-id',
    estimated_input_tokens: 150,
    estimated_output_tokens: 500,
    project_id: 'your-project-id',
    org_id: 'your-org-id',
    user_id: 'your-user-id'
  })
});

const result = await response.json();
console.log('Optimization recommendation:', result.recommendation);
```

### Budget Tracking
The system automatically tracks:
- Monthly budget usage per project
- Cost optimization recommendations
- Historical usage patterns

## Benefits

1. **Cost Optimization**: AI-powered recommendations for the most cost-effective provider/model combinations
2. **Budget Management**: Real-time tracking of project spending against monthly budgets
3. **Project Isolation**: Usage logs are now tied to specific projects for better organization
4. **Historical Analysis**: Store optimization recommendations for future reference and analysis

## Error Handling

The optimization endpoint handles various error scenarios:
- Prompt template not found (404)
- Project not found (404)
- OpenAI API failures (500)
- Invalid request data (400)

## Performance Considerations

- Database indexes are created for optimal query performance
- Optimization recommendations are cached in the database
- Monthly budget calculations are performed efficiently using date filters 
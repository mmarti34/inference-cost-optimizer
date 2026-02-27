-- Migration to add is_dynamic field to prompt_templates table
-- This allows distinguishing between static and dynamic prompts for optimization

-- Add is_dynamic column to prompt_templates table
ALTER TABLE prompt_templates 
ADD COLUMN IF NOT EXISTS is_dynamic BOOLEAN DEFAULT FALSE;

-- Add comment to explain the new column
COMMENT ON COLUMN prompt_templates.is_dynamic IS 'Whether this prompt template allows dynamic user input (true) or is static (false)';

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_prompt_templates_is_dynamic ON prompt_templates(is_dynamic);

-- Update existing prompts to be static by default
UPDATE prompt_templates SET is_dynamic = FALSE WHERE is_dynamic IS NULL; 
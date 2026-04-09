"""
Synthetic Mind — persistent cognitive layer for OptiML.

Accumulates understanding across LLM calls so future calls send less
information and cost less.  Five subsystems:

1. Observer       – async extraction of structured observations
2. Consolidation  – "sleep phase" that builds abstractions from observations
3. Scoped Memory  – hierarchical memory with inheritance + confidence decay
4. Prompt Assembler – assembles thinnest possible prompt using accumulated understanding
5. Forgetting     – principled, safety-constrained memory pruning
"""

PROMPT_VERIFY_ANSWER = """Check if CHOSEN matches ANSWER_GT given the image.

QUESTION: {question}
ANSWER_GT: {answer_gt}
CHOSEN: {chosen}

Output ONLY this JSON (no markdown, no extra text):
{{"is_correct": true}}
or
{{"is_correct": false}}
"""


PROMPT_EXTRACT_PERCEPTION = """Extract 3-6 key visual attributes from the image that affect the answer. Each < 10 words.

Priority: counts ("6 objects"), colors ("cyan cylinder"), sizes ("small cube"), text/numbers ("19")

QUESTION: {question}
ANSWER_GT: {answer_gt}
CHOSEN: {chosen}

Output ONLY this JSON (no markdown, no extra text):
{{"perception_attrs": ["attr1", "attr2", "attr3"]}}
"""


PROMPT_REWRITE_REJECTEDS = """
You are given the following information:

QUESTION: {question}
ANSWER_GT: {answer_gt}
CHOSEN: {chosen}
PERCEPTION_ATTRS: {perception_attrs_json}

Based on this information, generate two perturbed incorrect versions derived from CHOSEN.

CRITICAL: Each perturbed version MUST follow the EXACT same format as CHOSEN, including:
- Same structure (step-by-step reasoning if present)
- Same answer format (e.g., if CHOSEN ends with \\boxed{{answer}}, your version must too)
- Same level of detail

Perturbation rules:
1. Type 1: Keep the logical structure of the CHOSEN answer, but slightly alter perception-related details (counts, colors, text, numbers) so they contradict the image.
2. Type 2: Alter perception-related details AND introduce logical errors in the reasoning process.

Each perturbation MUST:
- Follow CHOSEN's exact format
- Contain the complete reasoning chain with perturbed details
- Result in a DIFFERENT final answer than ANSWER_GT

Output ONLY this JSON (no markdown, no extra text):

{{
  "rejected_perception_1": "<full perturbed reasoning text for type 1, following CHOSEN's format>",
  "rejected_perception_2": "<full perturbed reasoning text for type 2, following CHOSEN's format>"
}}
"""

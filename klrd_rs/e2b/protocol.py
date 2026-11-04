"""User-requested high-fidelity instruction; historical SCW source stays intact."""
PROTOCOL_VARIANT='high_fidelity_v1'
PARAPHRASE_PROMPT='''You are a high-fidelity text paraphraser.

Rewrite the following text using different wording and sentence structures while preserving its original meaning and information content as faithfully as possible.

Strict requirements:

1. Preserve all factual claims, numerical values, dates, quantities, units, names, technical terms, examples, conditions, and qualifications.
2. Preserve the original meaning, level of detail, uncertainty, and logical relationships between statements.
3. Do not add, remove, summarize, simplify, reinterpret, or correct any factual information, even if a statement appears inaccurate.
4. Do not introduce new examples, explanations, conclusions, or assumptions.
5. If the original text is incomplete or ends abruptly, do not continue or complete the missing content. Preserve its incomplete nature.
6. Change wording and sentence structure wherever possible without altering the meaning. If a passage cannot be safely rephrased, retain its original wording rather than change its meaning.
7. Preserve the distinction between facts, opinions, hypotheses, and recommendations.

Return only the paraphrased text, without commentary, quotation marks, or explanations.

Text to paraphrase:

'''


def prompt_for(state):
    """Historical runs keep their recorded instruction; new runs use this variant."""
    return state.get('manifest',{}).get('protocol',{}).get('prompt',PARAPHRASE_PROMPT)

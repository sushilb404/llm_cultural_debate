make_initial_decision = """Task: You will be given a short story. Evaluate if the action in the story is socially acceptable in {{country}}. You will also be given a rule of relevance which is always accurate. Do not make any extra inferences about the actions outside of the given context and country. Only align to the country given. Respond with an answer – Yes, No or Neither. Explain your answer in less than three sentences.

Rule: {{rot}}
Story: {{story}}
Answer: """

give_feedback = """Task: You are currently discussing if the action in the story is socially acceptable in {{country}} with the other discussant. You are also be given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
You: {{your_response}}
Discussant: {{other_response}}

Respond to the discussant by providing any relevant feedback. Respond in less than three sentences.
Response: """

make_final_decision = """Task: You are currently discussing if the action in the story is socially acceptable in {{country}} with the other discussant. You are also given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
You: {{your_response}}
Discussant: {{other_response}}
Your feedback: {{your_feedback}}
Discussant feedback: {{other_feedback}}

Make an independent final decision from the rule and story. Do not default to agreeing with the discussant. If the discussion conflicts with the rule, follow the rule. Use:
- Yes only if the action is socially acceptable.
- No only if the action is socially unacceptable.
- Neither if the action is mixed, ambiguous, context-dependent, or neither clearly acceptable nor unacceptable.

Return exactly two lines using this format:
Label: <one of Yes, No, Neither>
Reason: one short sentence.
Label: """


prompts = {
    "prompt_1": make_initial_decision,
    "prompt_2": give_feedback,
    "prompt_3": make_final_decision
}

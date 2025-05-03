from llama_index.core.prompts import RichPromptTemplate

prompt_temple = RichPromptTemplate(
    """
    Context: {{context}}
    ---
    Task: {{task}}
    ---
    Role: {{role}}
    ---
    Please focus on Subject: {{subject}} / Subcategory: {{subcategory}}
    ---
    Difficulty Level: treat me as a {{difficulty_level}}
    ---
    Output Structure: {{output}}
    ---
    Avoid these questions: {{avoid}}
    """
)

# General
task = "Need you to create a boolean question. If your answers are false, please explain why. If your answers are true, please provide further insight."
context = "I am preparing for a technical interview. I think it is a good idea to reinforce my knowledge by doing examinations regularly."
role = "Act as a hiring manager for a position that I am applying for. You have an interview with me where you're going to ask me related questions to evaluate if I am a potential candidate."
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_support_message(risk_level, issues, mood_score=None, stress_level=None):
    prompt = f"""
You are a calm and supportive student wellness assistant.
You do not diagnose.
You do not give medical advice.
Use a respectful, non-alarming tone.
Keep total response under 150 words.
Do not repeat the risk level.
Avoid dramatic language.
If format is not followed, regenerate properly.

Risk Level: {risk_level}
Issues: {issues}

Return response STRICTLY in this format:

Support_Message:
<2-3 sentences>

Suggestions:
1. <short actionable suggestion>
2. <short actionable suggestion>

Escalation:
<one line about when to seek professional help>
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        response_text = response.text

        # Hybrid Escalation Logic
        force_escalation = False

        if risk_level == "High":
            force_escalation = True

        if mood_score is not None and mood_score <= 2:
            force_escalation = True

        if stress_level is not None and stress_level >= 9:
            force_escalation = True

        if force_escalation:
            response_text += "\n\nIf you are in India, you may contact Kiran Mental Health Helpline: 1800-599-0019."

        return response_text

    except Exception:
        return "We are currently unable to generate advice. Please take a short break and consider speaking to someone you trust."


if __name__ == "__main__":
    print(generate_support_message(  ## only for testing, not part of the final product
        "Moderate",
        "Low sleep and high stress",
        mood_score=3,
        stress_level=7
    ))
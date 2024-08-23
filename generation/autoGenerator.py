from dotenv import load_dotenv
from openai import OpenAI
import os
import torch

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key=openai_api_key
)

# generate a code snippet based on given prompt
def generate_snippet(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in Python and PyTorch. Generate Python code that strictly follows the user prompt. Return only raw code, no comments, no formatting, no blank lines."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# validate whether a generated code snippet runs or crashes
def validate_snippet(snippet):
    return


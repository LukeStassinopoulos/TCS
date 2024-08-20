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
            {"role": "system", "content": "You are an expert in Python programming and PyTorch. Your task is to generate Python code snippets that use PyTorch APIs, specifically designed for fuzz testing. Focus on generating code that explores edge cases, potential errors, and unusual usage patterns in PyTorch functions. Ensure the code is well-structured and ready for validation."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message

# validate whether a generated code snippet runs or crashes
def validate_snippet(snippet):
    return


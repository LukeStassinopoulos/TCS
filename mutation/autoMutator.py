from dotenv import load_dotenv
from openai import OpenAI
import os
import torch

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key=openai_api_key
)

# mutates a given program
def mutate():
    return


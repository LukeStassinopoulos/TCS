from generation import autoGenerator
from mutation import autoMutator

def main():
    with open('gpt_prompt.txt', 'r') as prompt_file:
        prompt_structure = prompt_file.read()
    with open('test_data.txt', 'r') as api_file:
        api_signatures = api_file.readlines()

    for api_signature in api_signatures:
        api_signature = api_signature.strip()
        prompt = prompt_structure.format(api_signature)
        snippet = autoGenerator.generate_snippet(prompt)
        print(f"{snippet}\n")

if __name__ == "__main__":
    main()

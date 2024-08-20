from generation import autoGenerator
from mutation import autoMutator

def main():
    print(autoGenerator.generate_snippet("Generate a PyTorch code snippet that tests an edge case when creating a tensor with an unusual shape."))

if __name__ == "__main__":
    main()
import torch

def main():
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device('cuda')
        x = torch.randn(3, 3, device=device)
        print("Tensor on GPU:", x)
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    main()

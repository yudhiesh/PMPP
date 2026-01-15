from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
from PIL import Image
import torchvision.transforms.functional as TF

CUDA_FILE_PATH = Path("rgb_to_grayscale.cu")


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """Load an image and convert to a float tensor on GPU.

    Returns tensor of shape (H, W, C) with values in [0, 1]
    """
    image = Image.open(image_path).convert("RGB")
    # Convert to tensor: (C, H, W) with values in [0, 1]
    tensor = TF.to_tensor(image)
    # Rearrange to (H, W, C) for the CUDA kernel
    tensor = tensor.permute(1, 2, 0).contiguous()
    # Move to GPU
    tensor = tensor.cuda()
    return tensor


def save_grayscale_tensor(tensor: torch.Tensor, output_path: str):
    """Save a grayscale tensor as an image.

    Expects tensor of shape (H, W) with values in [0, 1]
    """
    # Clamp values to valid range
    tensor = tensor.clamp(0, 1)
    # Convert to PIL Image
    tensor_cpu = (tensor * 255).byte().cpu()
    image = Image.fromarray(tensor_cpu.numpy(), mode="L")
    image.save(output_path)
    print(f"Saved grayscale image to {output_path}")


def compile_cuda_code(cuda_file_path: Path):
    # Read CUDA source
    cuda_source = cuda_file_path.read_text()

    # C++ declaration for the function
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    # Compile and load the CUDA extension
    print("Compiling CUDA kernel...")
    module = load_inline(
        name="rgb_to_grayscale_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grayscale"],
        verbose=True,
    )
    print("Compilation complete!")
    return module


def main(cuda_file_path: Path, input_image_path: str, output_image_path: str):
    # Load input image
    print(f"Loading image from {input_image_path}...")
    input_tensor = load_image_as_tensor(input_image_path)
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    module = compile_cuda_code(cuda_file_path)

    # Run the CUDA kernel
    print("Running grayscale conversion...")
    output_tensor = module.rgb_to_grayscale(input_tensor)
    print(f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

    # Save the output
    save_grayscale_tensor(output_tensor, output_image_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <input_image> [output_image]")
        print("Example: python main.py photo.jpg grayscale_output.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "grayscale_output.jpg"

    main(CUDA_FILE_PATH, input_path, output_path)

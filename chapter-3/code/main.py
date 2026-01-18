from pathlib import Path
from typing import Literal
import torch
from torch.utils.cpp_extension import load_inline
from PIL import Image
import torchvision.transforms.functional as TF
import argparse


CUDA_FILE_PATH = Path("main.cu")

CPP_SOURCE = """
torch::Tensor process_image(torch::Tensor image, const std::string& kernel_name, int param);
"""

SUPPORTED_KERNELS = Literal["grayscale", "blur"]


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """Load an image and convert to a float tensor on GPU.
    Returns tensor of shape (H, W, C) with values in [0, 1]
    """
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image)
    tensor = tensor.permute(1, 2, 0).contiguous()
    tensor = tensor.cuda()
    return tensor


def save_tensor_as_image(tensor: torch.Tensor, output_path: str):
    """Save a tensor as an image.
    Supports both grayscale (H, W) and RGB (H, W, C) tensors.
    """
    tensor = tensor.clamp(0, 1)
    tensor_cpu = (tensor * 255).byte().cpu()

    if tensor.dim() == 2:
        # Grayscale
        image = Image.fromarray(tensor_cpu.numpy(), mode="L")
    elif tensor.dim() == 3 and tensor.shape[2] == 3:
        # RGB
        image = Image.fromarray(tensor_cpu.numpy(), mode="RGB")
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    image.save(output_path)
    print(f"Saved image to {output_path}")


def compile_cuda_code(cuda_file_path: Path, cpp_source: str):
    """Compile and load the CUDA extension."""
    cuda_source = cuda_file_path.read_text()

    print("Compiling CUDA kernel...")
    module = load_inline(
        name="image_processing",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["process_image"],
        verbose=True,
    )
    print("Compilation complete!")
    return module


def main(
    cuda_file_path: Path,
    cpp_source: str,
    input_image_path: str,
    output_image_path: str,
    kernel_name: str,
    kernel_param: int = 1,
):
    # Load input image
    print(f"Loading image from {input_image_path}...")
    input_tensor = load_image_as_tensor(input_image_path)
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # Compile CUDA code
    module = compile_cuda_code(cuda_file_path, cpp_source)

    # Run the selected kernel
    print(f"Running '{kernel_name}' kernel (param={kernel_param})...")
    output_tensor = module.process_image(input_tensor, kernel_name, kernel_param)
    print(f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

    # Save the output
    save_tensor_as_image(output_tensor, output_image_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process images using CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.jpg -o grayscale.jpg -k grayscale
  python main.py input.jpg -o blurred.jpg -k blur -p 3
        """,
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        default="output.jpg",
        help="Output image path (default: output.jpg)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        default="grayscale",
        help=f"Kernel to apply (default: grayscale)",
    )
    parser.add_argument(
        "-p",
        "--param",
        type=int,
        default=1,
        help="Kernel parameter (e.g., blur radius, default: 1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        cuda_file_path=CUDA_FILE_PATH,
        cpp_source=CPP_SOURCE,
        input_image_path=args.input,
        output_image_path=args.output,
        kernel_name=args.kernel,
        kernel_param=args.param,
    )

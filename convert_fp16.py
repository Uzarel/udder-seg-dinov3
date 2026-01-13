import argparse
import onnx
from onnxconverter_common import float16

def convert_fp32_to_fp16(
    input_onnx_path: str,
    output_onnx_path: str,
    keep_io_types: bool = True
) -> None:
    """
    Convert an FP32 ONNX model to FP16.

    Args:
        input_onnx_path: Path to the FP32 ONNX model.
        output_onnx_path: Path where the FP16 ONNX model will be saved.
        keep_io_types: If True, keeps model inputs/outputs in FP32 for compatibility.
    """
    model = onnx.load(input_onnx_path)

    # Convert to FP16. By default, this converter tries to keep I/O types in FP32
    # when keep_io_types=True, which is often what you want for inference pipelines.
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types
    )

    onnx.save(fp16_model, output_onnx_path)
    print(f"Saved FP16 model to: {output_onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert FP32 ONNX model to FP16 ONNX model.")
    parser.add_argument("--input", "-i", required=True, help="Path to input FP32 ONNX model.")
    parser.add_argument("--output", "-o", required=True, help="Path to output FP16 ONNX model.")
    parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="Keep model inputs/outputs in FP32 (recommended)."
    )
    args = parser.parse_args()

    convert_fp32_to_fp16(
        input_onnx_path=args.input,
        output_onnx_path=args.output,
        keep_io_types=args.keep_io_types
    )

if __name__ == "__main__":
    main()

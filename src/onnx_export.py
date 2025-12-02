import torch.onnx

def export_onnx(model, dummy_input, model_name):
    """Export the model to ONNX format."""
    torch.onnx.export(
        model, dummy_input,
        f"checkpoints/{model_name.lower()}/best.onnx",
        input_names=["image"],
        output_names=["logits"],
        export_params=True,
        do_constant_folding=True,
        opset_version=11, # Forces decomposition of LayerNorm and other layers
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

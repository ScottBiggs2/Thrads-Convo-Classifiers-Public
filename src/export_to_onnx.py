"""
Export a fine-tuned Transformer model to ONNX using the Optimum library.
"""
import os
import json
import argparse
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_and_quantize_model(model_dir: str, output_dir: str, max_length: int):
    """
    Exports a trained Transformer model to ONNX and quantizes it using Optimum.
    
    Args:
        model_dir (str): The directory where the fine-tuned model and tokenizer are saved.
        output_dir (str): The directory to save the ONNX and quantized ONNX models.
        max_length (int): The maximum sequence length for the model input.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if there's an onnx_config.json and remove opset constraints
    onnx_config_path = os.path.join(model_dir, "onnx_config.json")
    if os.path.exists(onnx_config_path):
        print(f"Found onnx_config.json - backing it up and removing opset constraints...")
        os.rename(onnx_config_path, onnx_config_path + ".backup")
    
    # Export with explicit opset 18
    print(f"Exporting model from {model_dir} to ONNX using Optimum (opset 18)...")
    
    from optimum.exporters.onnx import main_export
    
    try:
        main_export(
            model_name_or_path=model_dir,
            output=output_dir,
            task="text-classification",
            opset=18,
            device="cpu",
        )
    except FileNotFoundError as e:
        # Optimum has a cleanup bug - check if export actually succeeded
        if "model.onnx.data" in str(e):
            onnx_path = os.path.join(output_dir, "model.onnx")
            onnx_data_path = os.path.join(output_dir, "model.onnx_data")
            
            if os.path.exists(onnx_path) and os.path.exists(onnx_data_path):
                print("Export completed successfully (ignoring Optimum cleanup bug)")
            else:
                print(f"ERROR: Export failed - ONNX files not found")
                if os.path.exists(onnx_config_path + ".backup"):
                    os.rename(onnx_config_path + ".backup", onnx_config_path)
                raise
        else:
            if os.path.exists(onnx_config_path + ".backup"):
                os.rename(onnx_config_path + ".backup", onnx_config_path)
            raise
    
    # Restore backup if we made one
    if os.path.exists(onnx_config_path + ".backup"):
        os.rename(onnx_config_path + ".backup", onnx_config_path)
    
    print(f"Float ONNX model saved to {output_dir}")
    
    # Quantize the ONNX model
    onnx_path = os.path.join(output_dir, "model.onnx")
    quantized_onnx_path = os.path.join(output_dir, "model.quant.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"ERROR: Expected ONNX model not found at {onnx_path}")
        return
    
    try:
        print(f"Applying dynamic quantization... Saving to {quantized_onnx_path}")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_onnx_path,
            weight_type=QuantType.QInt8
        )
        print("Quantization complete.")
    except Exception as q_e:
        print(f"Quantization failed: {q_e}")
        print(f"Float ONNX model is available at {onnx_path}.")
    
    # Copy over training config if it exists
    training_config_path = os.path.join(model_dir, 'training_config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)
    
    print(f"\n✓ Processing complete. Final artifacts are in {output_dir}")
    print(f"   - Float model: {onnx_path}")
    if os.path.exists(quantized_onnx_path):
        print(f"   - Quantized model: {quantized_onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Transformer model to ONNX using Optimum and quantize it.")
    parser.add_argument("--model-path", type=str, required=True, help="Directory of the fine-tuned model.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the ONNX models.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length (informational only).")
    
    args = parser.parse_args()
    export_and_quantize_model(args.model_path, args.output_path, args.max_length)
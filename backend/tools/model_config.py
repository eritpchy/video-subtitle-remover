import os
from backend.config import config, BASE_DIR
from backend.tools.common_tools import merge_big_file_if_not_exists
from backend.tools.constant import SubtitleDetectMode

class ModelConfig:
    def __init__(self):
        self.LAMA_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'big-lama')
        self.STTN_AUTO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn-det', 'sttn.pth')
        self.PROPAINTER_MODEL_DIR = os.path.join(BASE_DIR,'models', 'propainter')
        if config.subtitleDetectMode.value == SubtitleDetectMode.Fast:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR,'models', 'V4', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.Accurate:
            self.DET_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'V4', 'ch_det')
        else:
            raise ValueError(f"Invalid subtitle detect mode: {config.subtitleDetectMode.value}")

        merge_big_file_if_not_exists(self.LAMA_MODEL_DIR, 'bit-lama.pt')
        merge_big_file_if_not_exists(self.PROPAINTER_MODEL_DIR, 'ProPainter.pth')
        merge_big_file_if_not_exists(self.DET_MODEL_DIR, 'inference.pdiparams')
    
    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            onnx_model = paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir

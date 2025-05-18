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
        merge_big_file_if_not_exists(self.DET_MODEL_DIR, 'model.onnx', man_filename='model.onnx.csv')
    
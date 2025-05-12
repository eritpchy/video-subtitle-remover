import torch

from backend.config import tr

class HardwareAccelerator:

    # 类变量，用于存储单例实例
    _instance = None

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = HardwareAccelerator()
            cls._instance.initialize()
        return cls._instance

    def __init__(self):
        self.__cuda = False
        self.__dml = False
        self.__onnx_providers = []
        self.__enabled = True
        self.__device = None

    def initialize(self):
        self.load_directml_device()
        if self.__device == None:
            self.load_cuda_device()
        if self.__device == None:
            self.load_cuda_device()
        self.load_onnx_providers()

    def load_directml_device(self):
        try:
            import torch_directml
            self.__device = torch_directml.device(torch_directml.default_device())
            self.__dml = True
        except:
            pass

    def load_cuda_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            self.__cuda = True

    def load_onnx_providers(self):
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            for provider in available_providers:
                if provider in [
                    "CPUExecutionProvider"
                ]:
                    continue
                if provider not in [
                    "DmlExecutionProvider",         # DirectML，适用于 Windows GPU
                    "ROCMExecutionProvider",        # AMD ROCm
                    "MIGraphXExecutionProvider",    # AMD MIGraphX
                    "VitisAIExecutionProvider",     # AMD VitisAI，适用于 RyzenAI & Windows, 实测和DirectML性能似乎差不多
                    "OpenVINOExecutionProvider",    # Intel GPU
                    "MetalExecutionProvider",       # Apple macOS
                    "CoreMLExecutionProvider",      # Apple macOS
                    "CUDAExecutionProvider",        # Nvidia GPU
                ]:
                    print(tr['Main']['OnnxExectionProviderNotSupportedSkipped'].format(provider))
                    continue
                print(tr['Main']['OnnxExecutionProviderDetected'].format(provider))
                self.__onnx_providers.append(provider)
        except ModuleNotFoundError as e:
            print(tr['Main']['OnnxRuntimeNotInstall'])

    def load_cpu_device(self):
        return torch.device("cpu")

    def has_accelerator(self):
        if not self.__enabled:
            return False
        return self.__cuda or self.__dml or len(self.__onnx_providers) > 0

    @property
    def accelerator_name(self):
        if not self.__enabled:
            return "CPU"
        if self.__dml:
            return "DirectML"
        if self.__cuda:
            return "GPU"
        elif len(self.__onnx_providers) > 0:
            return ", ".join(self.__onnx_providers)
        else:
            return "CPU"

    @property
    def onnx_providers(self):
        if not self.__enabled:
            return []
        return self.__onnx_providers

    def has_cuda(self):
        if not self.__enabled:
            return False
        return self.__cuda
    
    def set_enabled(self, enable):
        self.__enabled = enable

    @property
    def device(self):
        return self.__device
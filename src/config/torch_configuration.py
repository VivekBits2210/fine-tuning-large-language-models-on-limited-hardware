import torch
import logging
logger = logging.getLogger(__name__)


class TorchConfiguration:
    def __init__(self, **kwargs):
        self.cudnn_benchmark = kwargs.get('cudnn_benchmark', True)
        self.should_enable_mem_efficient_sdp = kwargs.get('should_enable_mem_efficient_sdp', False)
        self.should_enable_math_sdp = kwargs.get('should_enable_math_sdp', False)

    def commit(self):
        if torch.backends.cuda.flash_sdp_enabled():
            logger.info(f"Flash attention is enabled!")

        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.cuda.empty_cache()
        torch.backends.cuda.enable_mem_efficient_sdp(self.should_enable_mem_efficient_sdp)
        torch.backends.cuda.enable_math_sdp(self.should_enable_math_sdp)

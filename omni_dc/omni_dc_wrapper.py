import os
import numpy as np
import torch
from torch.nn import functional as F
from omni_dc.model.ognidc import OGNIDC          # make sure this import works in your PYTHONPATH
import os
import numpy as np
import torch
from torch.nn import functional as F

from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class DefaultArgs:
     # Dataset
     dir_data: str = '../datasets/nyudepthv2_h5'
     train_data_name: str = 'NYU'
     val_data_name: str = 'NYU'
     split_json: str = '../data_json/nyu.json'
     benchmark_gen_split: str = 'test'
     benchmark_save_name: str = ''
     patch_height: int = 228
     patch_width: int = 304
     resize_height: int = 240
     resize_width: int = 320
     top_crop: int = 0
     depth_scale_multiplier: float = 1.0
     training_patch_size: int = -1
     data_normalize_median: int = 1
     mixed_dataset_total_length: int = -1
     precomputed_depth_data_path: str = ''
     precomputed_alignment_method: str = 'disparity'
     load_dav2: int = 0

     # Hardware
     seed: int = 43
     gpus: str = "0,1,2,3,4,5,6,7"
     port: str = '29500'
     tcp_port: int = 8080
     address: str = 'localhost'
     num_threads: int = 4
     multiprocessing: bool = False

     # Network
     model: str = 'OGNIDC'
     integration_alpha: float = 5.0
     spn_type: str = 'dyspn'
     prop_time: int = 6
     prop_kernel: int = 3
     preserve_input: bool = False
     affinity: str = 'TGASS'
     affinity_gamma: float = 0.5
     conf_prop: bool = True
     pred_depth: int = 0
     pred_context_feature: int = 1
     pred_confidence_input: int = 0
     GRU_iters: int = 1
     gru_context_dim: int = 64
     gru_hidden_dim: int = 64
     conf_min: float = 1.0
     optim_layer_scale_factor: float = 1.0
     optim_layer_input_clamp: float = 1.0
     backbone_mode: str = 'rgbd'
     backbone: str = 'cformer'
     training_depth_mask_out_rate: float = 0.0
     training_depth_mask_integ_depth: int = 0
     training_depth_random_shift_range: float = 0.0
     depth_activation_format: str = 'exp'
     backbone_output_downsample_rate: int = 4
     depth_downsample_method: str = 'min'
     whiten_sparse_depths: int = 0
     gru_internal_whiten_method: str = 'mean'

     # Training
     loss: str = '1.0*SeqL1+1.0*SeqL2'
     laplace_loss_min_beta: float = -2.0
     gmloss_scales: int = 4
     sequence_loss_decay: float = 0.9
     intermediate_loss_weight: float = 0.0
     start_epoch: int = 1
     epochs: int = 36
     milestones: List[int] = field(default_factory=lambda: [18, 24, 28, 43])
     opt_level: str = 'O0'
     pretrain: str = None
     resume: bool = False
     test_only: bool = False
     batch_size: int = 12
     max_depth: float = 10.0
     augment: bool = True
     test_augment: int = 0
     flip: int = 1
     random_rot_deg: float = 0.0
     train_depth_noise: str = '0.0'
     val_depth_noise: str = '0.0'
     train_depth_pattern: str = '500'
     train_sfm_max_dropout_rate: float = 0.0
     train_depth_velodyne_random_baseline: int = 1
     val_depth_pattern: str = '500'
     inference_pattern_type: str = 'random'
     num_pattern_types: int = 3
     backbone_pattern_condition_format: str = 'none'
     lidar_lines: int = 64
     test_crop: bool = False
     grad_clip: float = 1.0
     grad_format: str = 'grad'

     # Summary
     num_summary: int = 4

     # Optimizer
     lr: float = 0.001
     gamma: float = 0.5
     optimizer: str = 'ADAMW'
     momentum: float = 0.9
     betas: Tuple[float, float] = (0.9, 0.999)
     epsilon: float = 1e-8
     weight_decay: float = 0.01
     warm_up: bool = True
     num_resolution: int = 1
     multi_resolution_learnable_input_weights: int = 0
     multi_resolution_learnable_gradients_weights: str = 'learnable'

     # Logs
     log_dir: str = '../experiments/'
     print_freq: int = 1
     save: str = 'trial'
     save_full: bool = False
     save_result_only: bool = False
     save_pointcloud_visualization: bool = False
     save_uniformat_max_dataset_length: int = 800

from omni_dc.model.ognidc import OGNIDC          # make sure this import works in your PYTHONPATH

class DepthPredictor(torch.nn.Module):
    def __init__(self, checkpoint_path: str, da_path: str, device: str = "cuda"):
        super(DepthPredictor, self).__init__()
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        self.device = device

        # Set up args for OGNIDC
        args = DefaultArgs()
        args.load_dav2 = 1
        args.num_resolution = 3
        args.pred_confidence_input = 1
        args.multi_resolution_learnable_gradients_weights = "uniform"
        args.optim_layer_input_clamp = 1.0
        args.depth_activation_format = 'exp'
        args.max_depth = 300.0
        args.data_normalize_median = 1
        args.whiten_sparse_depths = 1
        args.backbone_mode = 'rgbd'
        args.da_path = da_path

        # 1. Construct network and load weights
        self.model = OGNIDC(args, device=self.device).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)["net"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[DepthPredictor] WARNING – missing keys: {missing}")
        if unexpected:
            print(f"[DepthPredictor] WARNING – unexpected keys: {unexpected}")
        self.model.eval().to(self.device)

        self.K = np.eye(3)
        self.K = torch.tensor(self.K, device=device).unsqueeze(0)

    @torch.inference_mode()
    def forward(self, rgb: np.ndarray, sparse_depth: np.ndarray, num_resolution: int = 4) -> np.ndarray:
        """
        Run the network on a single RGB image and return the depth map (H×W float32).

        Parameters
        ----------
        rgb : np.ndarray
            H-by-W-by-3, uint8 0-255 **or** float32 0-1.
        sparse_depth : np.ndarray
            The sparse depth data.
        K : np.ndarray
            The camera intrinsic matrix.
        num_resolution : int
            Keep in sync with --num_resolution used for training.

        Returns
        -------
        np.ndarray
            The predicted depth map.
        """
        # ---- 1. Preprocess RGB to tensor (N,3,H,W), float32 in [0,1] ----
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)  # HWC → CHW
            .unsqueeze(0)      # → NCHW
            .to(self.device)
        )

        # ---- 2. Pad so H, W are divisible by 4·2^(R-1) ----
        _, _, H, W = tensor.shape
        divisor = int(4 * 2 ** (num_resolution - 1))
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)
            sparse_depth = np.pad(sparse_depth, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        sample = {
            "rgb": tensor,
            "dep": torch.tensor(sparse_depth, device=self.device).unsqueeze(0).unsqueeze(0),
            "K": self.K,
            "pattern": torch.zeros((1,), device=self.device)
        }

        # ---- 3. Forward pass ----
        output_dict = self.model(sample)
        pred = output_dict["pred"][0, 0]  # (H', W')

        # ---- 4. Remove padding and return CPU numpy array ----
        if pad_h:
            pred = pred[:-pad_h, :]
        if pad_w:
            pred = pred[:, :-pad_w]

        return pred.cpu().numpy()

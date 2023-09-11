import numpy as np
import torch
from easydict import EasyDict as edict
import argparse
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
import torch.onnx


parser = argparse.ArgumentParser(description='HybrIK Demo')


parser.add_argument('--model-dir',
                    help='model path',
                    required=True,
                    type=str)
parser.add_argument('--batch-size',
                    help='batch size',
                    default=1,
                    type=int)
parser.add_argument('--output-path',
                    help='model output path with name: path/to/model.onnx',
                    required=True,
                    type=str)
parser.add_argument("--dynamic-batch",
                    help="set dynamic batch",
                    action="store_true")




opt = parser.parse_args()




cfg_file = 'configs/export_config.yaml'
CKPT = opt.model_dir
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})


transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])


hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)


prev_box = None
renderer = None
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

print('### Run Model...')
idx = 0

if opt.dynamic_batch:
    x = torch.randn(opt.batch_size, 3, 256, 256, requires_grad=True)
    torch.onnx.export(hybrik_model,
                  x,
                  opt.out_dir,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
else:
    x = torch.randn(opt.batch_size, 3, 256, 256, requires_grad=True)
    torch.onnx.export(hybrik_model,
                  x,
                  opt.out_dir,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'])



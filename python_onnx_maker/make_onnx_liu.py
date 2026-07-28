from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
import torch
class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        with torch.amp.autocast('cuda', enabled=True):
            disp = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
        return disp
cfg = OmegaConf.load("./weights/23-51-11/cfg.yaml")
cfg['vit_size'] = 'vitl'
cfg['height'] = 608 # 608 / 32 = 19 
cfg['width'] = 960 # 960 / 32 = 30
model = FoundationStereoOnnx(cfg)
ckpt = torch.load("./weights/23-51-11/model_best_bp2.pth", weights_only=False, map_location = 'cpu') # except model tensor, it also contains trainning metadata.
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()
left_img = torch.randn(1, 3, cfg.height, cfg.width).cuda().float()
right_img = torch.randn(1, 3, cfg.height, cfg.width).cuda().float()

torch.onnx.export(
        model,
        (left_img, right_img),
        '/home/liu4000/Desktop/FS_realsense/models/fs_960x608_iters32/fs.onnx',
        input_names = ['left', 'right'],
        output_names = ['disp'],
        dynamo = False,
    )
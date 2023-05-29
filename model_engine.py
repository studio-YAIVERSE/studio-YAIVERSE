"""
History
    - 230419 : MINSU , init
        - code skeleton
        - parsing (+ yaml format also !)
        - build get3d
    - 230423 : DONGHA , fix
        - DDP settings
"""
import os
import sys
import copy
import yaml
import torch

import dist_util
try:
    import dnnlib
except ImportError:
    sys.path.insert(0, os.path.abspath('../'))
    import dnnlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training.networks_get3d import GeneratorDMTETMesh
from addon import addon

class Engine(object):

    config: ...
    global_kwargs: ...
    G_kwargs: ...
    clip_kwargs: ...

    def __init__(self, config: dict, rank: int):
        self.rank = rank
        self.config = config
        self.device = torch.device('cuda', self.rank)
        self.parse()

    @classmethod
    def legacy_constructor(cls, config_path) -> "Engine":  # FIXME (DONGHA) - remove this
        self = object.__new__(cls)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.rank = 0
        self.device = torch.device(self.config['GLOBAL']['device'])
        self.parse()
        return self

    def parse(self):
        # setting : global configuration
        self.global_kwargs = dnnlib.EasyDict(self.config['GLOBAL'])

        # ref) get3d : train_3d.py ln251 - ln320
        # setting : GET3D configuration
        opts = dnnlib.EasyDict(self.config['GET3D'])
        # global
        G_kwargs = self.G_kwargs = dnnlib.EasyDict()
        G_kwargs.device = self.device
        G_kwargs.class_name = 'training.networks_get3d.GeneratorDMTETMesh'
        G_kwargs.img_resolution = opts.img_res  # // reformed
        G_kwargs.img_channels = opts.img_channels  # // reformed
        # mapping network
        G_kwargs.z_dim = opts.latent_dim
        G_kwargs.w_dim = opts.latent_dim
        G_kwargs.c_dim = opts.c_dim   # 0(=None) # NOTE : This can be used for class conditioning ... // reformed
        G_kwargs.mapping_kwargs = dnnlib.EasyDict()
        G_kwargs.mapping_kwargs.num_layers = 8
        # stylegan2 + tri-plane
        G_kwargs.use_style_mixing = opts.use_style_mixing
        G_kwargs.one_3d_generator = opts.one_3d_generator
        G_kwargs.dmtet_scale = opts.dmtet_scale
        G_kwargs.n_implicit_layer = opts.n_implicit_layer
        G_kwargs.feat_channel = opts.feat_channel
        G_kwargs.mlp_latent_channel = opts.mlp_latent_channel
        G_kwargs.deformation_multiplier = opts.deformation_multiplier
        G_kwargs.tri_plane_resolution = opts.tri_plane_resolution
        G_kwargs.n_views = opts.n_views
        G_kwargs.use_tri_plane = opts.use_tri_plane
        G_kwargs.tet_res = opts.tet_res
        # G_kwargs.tet_path = '../data/tets'
        # neural renderer
        G_kwargs.render_type = opts.render_type
        G_kwargs.data_camera_mode = opts.data_camera_mode
        # misc
        G_kwargs.fused_modconv_default = 'inference_only'

        # setting : NADA configuration
        clip_kwargs = self.clip_kwargs = dnnlib.EasyDict(self.config['NADA'])
        clip_kwargs.device = self.device

    def build_get3d_pair(self):
        G_ema: "GeneratorDMTETMesh" = dnnlib.util.construct_class_by_name(**self.G_kwargs) \
            .train() \
            .requires_grad_(False) \
            .to(self.device)

        G_ema = addon(G_ema)    # 0530

        assert self.global_kwargs['resume_pretrain'] != '', "ASSERTION : Specify pretrained GET3D model"

        if self.rank == 0:
            model_state_dict = torch.load(
                self.global_kwargs['resume_pretrain'],
                map_location=self.device
            )
            G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        dist_util.sync_params(G_ema.parameters(), src=0)
        dist_util.sync_params(G_ema.buffers(), src=0)

        G_ema_frozen: "GeneratorDMTETMesh" = copy.deepcopy(G_ema).eval()

        return G_ema, G_ema_frozen


# __main__ script for unit test

if __name__ == "__main__":

    # 0529 : Legacy code ... some errors may occur. Those were used for unit-test (debugging)

    engine = Engine.legacy_constructor('experiments/default.yaml')

    logger = dnnlib.util.Logger(file_name='log.txt', file_mode='a', should_flush=True)

    test_get3d, test_get3d_frozen = engine.build_get3d_pair()

    # 0. unit test: def build_get3d_pair()
    # print(test_get3d.synthesis)
    
    # 1-1. misc exp. (1)
    # modules() vs children() ?
    #triplane = test_get3d.synthesis.generator.tri_plane_synthesis.children()
    # print(hasattr(triplane, 'b4'), hasattr(triplane, 'b8'))
    # for cc in test_iter:
    #     for cc in child.children():
    #         print(hasattr(child, 'conv0'), hasattr(child, 'conv1'), hasattr(child, 'totex'), hasattr(child, 'togeo'))

    # 1-1. misc exp. (2)
    mlp = test_get3d.synthesis.generator.mlp_synthesis_tex

    print(mlp.layers)

        # for cc in child.children():
        #     print(hasattr(cc, 'layers'), hasattr(cc, 'layer0'), hasattr(cc, 'layer1'))
            # for ccc in cc.children():
                # print(ccc)
        # print(hasattr(child, 'conv0'), hasattr(child, 'conv1'), hasattr(child, 'totex'), hasattr(child, 'togeo'))
        # print(child.resolution)

    #1-2. unit test: def get_all_triplane_layers_dict()
    ut_tex, ut_geo = test_get3d.get_all_triplane_layers_dict()
    print(ut_tex)
    print(ut_geo)

    #1-3. unit test: def freeze_triplane_layers_()
    test_get3d.freeze_triplane_layers()
    # Fot test, uncomment def freeze_triplane_layers_() #---debug--- region
    
    #1-4. unit test : def unfreeze_triplane_layers_()
    dummy_idx_tex = [1, 3, 5, 7, 8]
    dummy_idx_geo = [1, 5, 9, 13, 17, 20, 21]
    test_get3d.unfreeze_triplane_layers(dummy_idx_tex, dummy_idx_geo)

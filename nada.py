"""
History
    - 230419 : MINSU , init
        - code skeleton
        - bring model_engine > build get3d (frozen, trainable)
    - 230420 : MINSU , fix
        - fix two functions for get3d forward pass
    - 230423 : DONGHA , fix
        - change __init__ (constructor) to be compatible with Engine argument
        - add wrap_ddp to support DDP learning
    - 230424 : MINSU , fix
        - (prev) fixed z --> (now) fixed w
        - (prev) unfreeze both tex. & geo. -> (now) unfreeze either tex. or geo. 
"""

import torch
from clip_loss import CLIPLoss
from functional import generate_custom, freeze_generator_layers, unfreeze_generator_layers


class YAIverseGAN(torch.nn.Module):

    def __init__(self, engine):
        super(YAIverseGAN, self).__init__()

        self.engine = engine
        self.device = self.engine.device

        self.generator_trainable, self.generator_frozen = self.engine.build_get3d_pair()

        # TODO (optional): selective freezing
        # self.generator_trainable.freeze_layers()
        # self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        # self.generator_trainable.train()

        # clip model & loss
        clip_kwargs = self.engine.clip_kwargs
        self.clip_loss_models = torch.nn.ModuleDict({
            model_name: CLIPLoss(
                self.device,
                lambda_direction=clip_kwargs['lambda_direction'],
                lambda_patch=clip_kwargs['lambda_patch'],
                lambda_global=clip_kwargs['lambda_global'],
                lambda_texture=clip_kwargs['lambda_texture'],
                lambda_manifold=clip_kwargs['lambda_manifold'],
                clip_model=model_name
            ) for model_name in clip_kwargs['clip_models']
        })

        self.clip_model_weights = {
            model_name: weight for model_name, weight in
            zip(clip_kwargs['clip_models'], clip_kwargs['clip_models_weight'])
        }

        # text
        self.source_text = clip_kwargs['source_text']
        self.target_text = clip_kwargs['target_text']

        # for layer freezing process
        self.auto_layer_k     = clip_kwargs['auto_layer_k']
        self.auto_layer_iters = clip_kwargs['auto_layer_iters']
        self.auto_layer_batch = clip_kwargs['auto_layer_batch']

        self.to(self.device)

    def get_loop_settings(self):
        g = self.engine.global_kwargs
        return (
            self.device, g.outdir, g.batch, g.vis_samples,
            g.sample_1st, g.sample_2nd, g.iter_1st, g.iter_2nd,
            self.engine.clip_kwargs.lr,
            g.output_interval, g.save_interval,
            self.engine.clip_kwargs.gradient_clip_threshold
    )
        
    def determine_opt_layers(self):
        """
        original code : return chosen layers : List[nn.Modules, nn.Modules, ...]
        our code      : return chosen layers idx : List[int, int, ...], List[int, int, ...]
                        * note that this returns two list for tex. and geo.
        """
        z_dim           = 512
        c_dim           = self.engine.G_kwargs['c_dim']
        sample_z_tex    = torch.randn(self.auto_layer_batch, z_dim, device=self.device)
        sample_z_geo    = torch.randn(self.auto_layer_batch, z_dim, device=self.device)

        with torch.no_grad():
            initial_w_tex_codes = self.generator_frozen.mapping(sample_z_tex, c_dim)        # (B, 9, 512)
            initial_w_geo_codes = self.generator_frozen.mapping_geo(sample_z_geo, c_dim)    # (B, 22, 512)

        w_tex_codes = torch.Tensor(initial_w_tex_codes.cpu().detach().numpy()).to(self.device)
        w_geo_codes = torch.Tensor(initial_w_geo_codes.cpu().detach().numpy()).to(self.device)

        w_tex_codes.requires_grad = True
        w_geo_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_tex_codes, w_geo_codes], lr=0.01)

        for _ in range(self.auto_layer_iters):
            generated_from_w, _ = generate_custom(self.generator_trainable, tex_z=w_tex_codes, geo_z=w_geo_codes, mode='layer') # (B, C, H, W)
            generated_from_w = generated_from_w[:, :-1, :, :]   # [RGB image, Silhouette] (B,4,H,W) -> [RGB image] (B,3,H,W)
            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_text) for model_name in self.clip_model_weights.keys()]
            w_loss = torch.sum(torch.stack(w_loss))
            
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
            
        layer_tex_weights = torch.abs(w_tex_codes - initial_w_tex_codes).mean(dim=-1).mean(dim=0)
        layer_geo_weights = torch.abs(w_geo_codes - initial_w_geo_codes).mean(dim=-1).mean(dim=0)

        cutoff = len(layer_tex_weights)

        chosen_layers_idx = torch.topk(torch.cat([layer_tex_weights, layer_geo_weights], dim=0), self.auto_layer_k)[1].cpu().numpy().tolist()
        chosen_layer_idx_tex = []
        chosen_layer_idx_geo = []
        for idx in chosen_layers_idx:
            if idx >= cutoff:
                chosen_layer_idx_geo.append(idx - cutoff)
            else:
                chosen_layer_idx_tex.append(idx)
        # 0422 ver
        # chosen_layer_idx_tex = torch.topk(layer_tex_weights, self.auto_layer_k)[1].cpu().numpy().tolist()
        # chosen_layer_idx_geo = torch.topk(layer_geo_weights, self.auto_layer_k)[1].cpu().numpy().tolist()

        return chosen_layer_idx_tex, chosen_layer_idx_geo

    def forward(        # modified for GET3D
            self,
            tex_z, geo_z,
            truncation_psi=1,
            # other args are not necessary
    ):
        c_dim = self.engine.G_kwargs['c_dim']
        batch = tex_z.shape[0]

        if self.training and self.auto_layer_iters > 0:
            unfreeze_generator_layers(self.generator_trainable, [], [])
            topk_idx_tex, topk_idx_geo = self.determine_opt_layers()
            freeze_generator_layers(self.generator_trainable)
            unfreeze_generator_layers(self.generator_trainable, topk_idx_tex, topk_idx_geo)

        w_geo = self.generator_frozen.mapping_geo(geo_z, c_dim)
        w_tex = self.generator_frozen.mapping(tex_z, c_dim)

        with torch.no_grad():
            frozen_img, _ = generate_custom(self.generator_frozen, tex_z=w_tex, geo_z=w_geo, c=c_dim, mode='nada')

        trainable_img, _ = generate_custom(self.generator_trainable, tex_z=w_tex, geo_z=w_geo, c=c_dim, mode='nada')

        input_dict = {
            "src_img": frozen_img[:, :-1],
            "target_img": trainable_img[:, :-1],
            "source_class": self.source_text,
            "target_class": self.target_text
        }

        clip_loss = torch.sum(
            torch.stack([
                self.clip_model_weights[model_name] * self.clip_loss_models[model_name](**input_dict)
                for model_name in self.clip_model_weights.keys()
            ]), dim=0
        )

        clip_loss = torch.mean(clip_loss.reshape(-1, batch), dim=0)
        return [frozen_img, trainable_img], clip_loss


# __main__ script for unit test
if __name__ == "__main__":

    import argparse
    import yaml
    from model_engine import Engine
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='experiments/default.yaml')
    args = parser.parse_args()

    rank = 0
    device = "cuda:0"

    ut_gan = YAIverseGAN(Engine(yaml.safe_load(args.config_path), rank))

    sample_z_geo = torch.randn(4, 512, device=device)
    sample_z_tex = torch.randn(4, 512, device=device)

    [img1, img2], loss = ut_gan(tex_z=sample_z_tex, geo_z=sample_z_geo)

    print(img1.shape)
    print(loss)
    print("DONE!")

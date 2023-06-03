# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torchvision.transforms import Resize
import torch.nn.functional as F 
import torch
import clip
import training.bgaugmentation as aug
# ----------------------------------------------------------------------------
class Loss:
    def accumulate_gradients(
            self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------
# Regulrarization loss for dmtet
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


# JUNGBIN (0508 ~)
class CLIPLoss(torch.nn.Module):
    def __init__(self, path, device):
        super(CLIPLoss, self).__init__()
        # FIXME : where is ViT-B/16 ??
        self.model, _ = clip.load(path, device=device)
        # FIXME : torchvision transform // 따로 normalize 할 필요는 없음. 이미 GAN 통과 후 tensor 라 (0, 1) 사이
        # self.upsample = torch.nn.Upsample(1120)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=5)
        self.preprocess = F.interpolate

    def forward_image_text(self, image, text):
        if type(image) != type(text): # if text is not tensor (string)
            text = clip.tokenize(text)
            text = self.model.encode_text(text)
        # image = self.avg_pool(self.upsample(image)) # FIXME : torchvision transform
        image = self.preprocess(image, size=224, mode='bicubic')

        image_embedding = self.model.encode_image(image)
        similarity = 1. - F.cosine_similarity(image_embedding, text, dim=1, eps=1e-8) 
        return similarity
    
    
    def forward_image_image(self, gen_image, real_image): #(bs,C,1023,1024)

        gen_image = self.preprocess(gen_image, size=224, mode='bicubic')
        real_image = self.preprocess(real_image, size=224, mode='bicubic')

        gen_image_embedding = self.model.encode_image(gen_image)
        real_image_embedding = self.model.encode_image(real_image)
        similarity = 1. - F.cosine_similarity(gen_image_embedding,real_image_embedding,dim=1,eps=1e-8) 
        return similarity


class TAPS3D_loss(Loss):
    def __init__(
            self, device, G, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
            gamma_mask=10, load=['', ''], vis_count=0):
        super().__init__()
        self.device = device
        self.G = G
        # self.D = D
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.gamma_mask = gamma_mask
        self.load = load
        self.clip_loss = []
        for path in load:
            if path != '':
                self.clip_loss.append(CLIPLoss(path, device))
        self.vis_count = vis_count

    def run_G(
            self, z, c, camera, update_emas=False, return_shape=False,
    ):
        # Step 1: Map the sampled z code to w-space
        # FIXME : mapping network 1st layer = [z :512dim] + [clip :512dim]
        ws = self.G.mapping(z, c, update_emas=update_emas)
        geo_z = torch.randn_like(z)
        ws_geo = self.G.mapping_geo(
            geo_z, c,
            update_emas=update_emas)

        # Step 2: Apply style mixing to the latent code
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
                
                cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws_geo.shape[1]))
                ws_geo[:, cutoff:] = self.G.mapping_geo(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

        # Step 3: Generate rendered image of 3D generated shapes.
        if return_shape:
            img, sdf, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, render_return_value = self.G.synthesis(
                ws,
                return_shape=return_shape,
                ws_geo=ws_geo,
                camera=camera
            )
            return img, sdf, ws, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, ws_geo, sdf_reg_loss, render_return_value
        else:
            img, syn_camera, mask_pyramid, sdf_reg_loss, render_return_value = self.G.synthesis(
                ws, return_shape=return_shape,
                ws_geo=ws_geo,
                camera=camera
                )
        return img, ws, syn_camera, mask_pyramid, render_return_value


    # TODO !!!!!!!!!!
    # code 넘 지저분하다... 일단은 그냥 짜고
    # dataset_TAPS3D return 값들부터 다듬어야할 듯
    def accumulate_gradients(
            self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, img_for_CLIP=None):
        # FIXME : img_for_CLIP -> ?
        
        assert phase in ['Gmain', 'Greg', 'Gboth'] #,'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        # if self.r1_gamma == 0:
        #     phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # First generate the rendered image of generated 3D shapes
                gen_img, gen_sdf, _gen_ws, gen_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _gen_ws_geo, \
                sdf_reg_loss, render_return_value = self.run_G(
                    gen_z, gen_c, real_c, return_shape=True
                )

                real_img_rgb = real_img[:, :3]
                real_img_alpha = real_img[:, -1]
                gen_img_rgb = gen_img[:,:3]
                gen_img_alpha = gen_img[:, -1]
                gen_img_rgb, img_for_CLIP = aug.augmentation(gen_img_rgb, gen_img_alpha, real_img_rgb, real_img_alpha, self.vis_count)
                if self.vis_count >= 0:
                    self.vis_count += 1
                loss_CLIP = None
                loss_img = None
                for clip in self.clip_loss:
                    loss_img_ = clip.forward_image_image(gen_img_rgb, img_for_CLIP)
                    loss_CLIP_ = clip.forward_image_text(gen_img_rgb, gen_c)
                    # print("DEBUG : ", loss_img_, loss_CLIP_)
                    if loss_img == None:
                        loss_img = loss_img_
                    else:
                        loss_img += loss_img_

                    if loss_CLIP == None:
                        loss_CLIP = loss_CLIP_
                    else:
                        loss_CLIP += loss_CLIP_

                loss_Gmain = loss_CLIP + loss_img
                
                training_stats.report('Loss/G/IMG_reg', loss_img)
                training_stats.report('Loss/G/CLIP', loss_CLIP)
                training_stats.report('Loss/G/Loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

            # --------- MINSU
            return loss_Gmain.mean()  
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
# TODO : background aug 수정
import os

def get_checkerboard(images, n=8):
    '''
    function for generating checkerboard background for low level regualrization loss
    images : images tensor of shape (B, C, H, W)
    n  : integer for number of checkboard patterns 
    return torch.tensor of shape (B, 3, H, W)
    '''
    B, _, H, W = images.shape
    colors = torch.rand(2, B, 3, 1, 1, 1, 1, dtype=images.dtype, device=images.device)
    h = H // n
    w = W // n
    bg = torch.ones(B, 3, n, h, n, w, dtype=images.dtype, device=images.device) * colors[0]
    bg[:, :, ::2, :, 1::2] = colors[1]
    bg[:, :, 1::2, :, ::2] = colors[1]
    bg = bg.view(B, 3, H, W)
    return bg

def get_checkerboard_np(image, n=8):
    # numpy
    H, W, _ = image.shape
    colors = np.random.randint(0, 250, size=2)
    h = H // n
    w = H // n
    bg = np.ones((3, n, h, n, w)) * colors[0]
    bg[:, ::2, :, 1::2] = colors[1]
    bg[:, 1::2, :, ::2] = colors[1]
    return bg.reshape(image.shape)


def get_Gaussiannoise(images, mean, std):
    '''
    function for generating gaussian noise background for low level regualrization loss
    images : images tensor of shape (B, C, H, W)
    mean : mean  
    std : standard deviation
    output : B, C, H, W
    '''
    B, _, H, W = images.shape
    noise = torch.randn(B, 3, H, W,dtype=images.dtype, device=images.device) * std + mean 
    noise = torch.clamp(noise,0,1)
    return noise

##############################implementing########################
import torch
import numpy as np
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

def rand_fft_image_np(image, sd=None, decay_power=1):
    sd = 0.01 if sd is None else sd
    H, W, _ = image.shape
    freq = rfft2d_freqs(H, W)
    fh, fw = freq.shape
    spectrum_var = sd * torch.randn((2, _, fh, fw), dtype=torch.float32)
    spectrum = torch.complex(spectrum_var[0], spectrum_var[1])
    spectrum_scale = 1.0 / np.maximum(freq, 1.0 / max(H, W)) ** decay_power
    spectrum_scale *= np.sqrt(W * H)
    scaled_spectrum = spectrum * spectrum_scale * 255
    img = np.fft.irfft2(scaled_spectrum.detach().cpu().numpy())
    return img.transpose(1, 2, 0)

#rand_fft_image((4,224,224,3),sd=1)[1].shape = (3,224,224)
def rand_fft_image(image, sd=None, decay_power=1):
    '''
    shape :  b, h, w, ch 
    sd : 0.08 ~ 0.12
    output shape : b, ch, h, w
    '''
    shape = image.shape
    b, ch, h, w = shape
    sd = 0.01 if sd is None else sd

    imgs = []
    for _ in range(b):
        freqs = rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        spectrum_var = sd * torch.randn((2, ch, fh, fw), dtype=torch.float32)
        spectrum = torch.complex(spectrum_var[0], spectrum_var[1])
        spertum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay_power
        # Scale the spectrum by the square-root of the number of pixels
        # to get a unitary transformation. This allows to use similar
        # learning rates to pixel-wise optimisation.
        spertum_scale *= np.sqrt(w * h)
        scaled_spectrum = spectrum * spertum_scale
        img = torch.tensor(np.fft.irfft2(scaled_spectrum))
        #img = torch.fft.irfft(scaled_spectrum, 2)
        # in case of odd input dimension we cut off the additional pixel
        # we get from irfft2d length computation
        # img = img[:h, :w, :ch]
        # img = img.permute(2, 0, 1)
        imgs.append(img)
    return torch.stack(imgs) / 4.0
############################################################################

def torch_vis_2d(x, renormalize=True, name=''):
    '''
    function for visualizing 2d tensor 
    '''
    # x: [3, H, W] or [1, H, W] or [H, W]
    x = x[0]
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
    # print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        # x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)
        # >> 원본 코드가 어땠는지는 모르겠지만 우리랑 안 맞음
        x += 1
        x *= 127.5

    x = Image.fromarray(np.uint8(x)).convert('RGB')
    x.save(f'{name}.png')


def augmentation_np(img1, img2):
    # img = (B, C, H, W)
    prob = np.random.rand(1)
    if prob < 0.5:
        aug = get_checkerboard
    else:
        aug = rand_fft_image
    return aug(img1), aug(img2)
    
def augmentation(img1, mask1, img2, mask2, num):
    # mask1 = mask1.repeat(1, 3, 1, 1)
    # mask2 = mask2.repeat(1, 3, 1, 1)

    prob = np.random.rand(1)
    if prob < 0.5:
        aug = get_checkerboard
    else:
        aug = rand_fft_image
    aug_bg = aug(img1).to(img1.device)

    mask1 = mask1.unsqueeze(1)
    mask2 = mask2.unsqueeze(1)
    
    img1 = img1 * mask1 + aug_bg * (1.0 - mask1)
    img2 = img2 * mask2 + aug_bg * (1.0 - mask2)

    if num == -1:
        pass
    else:
        if num < 10:
            while os.path.exists(f'./visualize/ori_img1_{num}.png'):
                num += 1
            torch_vis_2d(img1, name=f'./visualize/ori_img1_{num}')
            torch_vis_2d(img2, name=f'./visualize/ori_img2_{num}')    
            torch_vis_2d(img1, name=f'./visualize/img1_{num}')
            torch_vis_2d(img2, name=f'./visualize/img2_{num}')
    return img1, img2


if __name__ == "__main__":
    
    dummy = np.zeros((224, 224, 3))

    checkboard = get_checkerboard_np(dummy)
    print(checkboard.dtype, checkboard.shape)
    import cv2
    cv2.imwrite("checkerboard.png", checkboard)

    dummy = np.random.randint(low=0, high=255, size=(224, 224, 3))
    fourier = rand_fft_image_np(dummy)
    print(fourier.dtype, fourier.shape)
    cv2.imwrite("fourier.png", fourier)


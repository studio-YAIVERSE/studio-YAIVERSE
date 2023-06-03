# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import glob
import numpy as np
import zipfile
import torch
import dnnlib
import cv2
from PIL import Image
try:
    import pyspng
except ImportError:
    pyspng = None
import random
import torchvision
import torchvision.transforms as T
# ----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            name,  # Name of the dataset.
            raw_shape,  # Shape of the raw image data (NCHW).
            max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
            use_labels=True,  # Enable conditioning labels? False = label dimension is zero.
            xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
            random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # We don't Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._w[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

class ImageFolderDataset(Dataset):
    def __init__(
            self,
            path,  # Path to directory or zip.
            camera_path,  # Path to camera
            resolution=None,  # Ensure specific resolution, None = highest available.
            data_camera_mode='TAPS3D',
            add_camera_cond=False,
            split='all',
            label='',
            debug = True,
            **super_kwargs  # Additional arguments for the Dataset base class.
    ):
        ##debug!##
        self.debug = debug
        ###########
        self.data_camera_mode = data_camera_mode
        self._path = path
        self._zipfile = None
        self.root = path
        self.mask_list = None
        self.add_camera_cond = add_camera_cond
        root = self._path
        self.camera_root = camera_path
        if data_camera_mode == 'shapenet_car' or data_camera_mode == 'shapenet_chair' \
                or data_camera_mode == 'renderpeople' or data_camera_mode == 'shapenet_motorbike' \
                or data_camera_mode == 'ts_house' \
                or data_camera_mode == 'ts_animal'\
                or data_camera_mode == 'TAPS3D' \
                or data_camera_mode == 'objaverse_multi':
            print('==> use shapenet dataset')
            if not os.path.exists(root):
                print(root)
                print('==> ERROR!!!! THIS SHOULD ONLY HAPPEN WHEN USING INFERENCE')
                n_img = 1234
                self._raw_shape = (n_img, 3, resolution, resolution)
                self.img_size = resolution
                self._type = 'dir'
                self._all_fnames = [None for i in range(n_img)]
                self._image_fnames = self._all_fnames
                name = os.path.splitext(os.path.basename(path))[0]
                print(
                    '==> use image path: %s, num images: %d' % (
                        self.root, len(self._all_fnames)))
                super().__init__(name=name, raw_shape=self._raw_shape, **super_kwargs)
                return
            print("DEBUG : 192ln : ", root)
            folder_list = sorted(os.listdir(root))
            if data_camera_mode == 'TAPS3D':
                split_name = './3dgan_data_split/TAPS3D_temp/%s.txt' % (split)
                if split == 'all':
                    split_name = './3dgan_data_split/TAPS3D_temp.txt'
                valid_folder_list = []
                with open(split_name, 'r') as f:
                    all_line = f.readlines()
                    for l in all_line:
                        valid_folder_list.append(l.strip())
                valid_folder_list = set(valid_folder_list)
                useful_folder_list = set(folder_list).intersection(valid_folder_list)
                folder_list = sorted(list(useful_folder_list))
            if data_camera_mode == 'shapenet_chair' or data_camera_mode == 'shapenet_car':
                if data_camera_mode == 'shapenet_car':
                    split_name = './3dgan_data_split/shapenet_car/%s.txt' % (split)
                    if split == 'all':
                        split_name = './3dgan_data_split/shapenet_car.txt'
                elif data_camera_mode == 'shapenet_chair':
                    split_name = './3dgan_data_split/shapenet_chair/%s.txt' % (split)
                    if split == 'all':
                        split_name = './3dgan_data_split/shapenet_chair.txt'
                valid_folder_list = []
                #######debug####
                if self.debug:
                    split_name = '.' + split_name
                #####
                with open(split_name, 'r') as f:
                    all_line = f.readlines()
                    for l in all_line:
                        valid_folder_list.append(l.strip())
                valid_folder_list = set(valid_folder_list)
                useful_folder_list = set(folder_list).intersection(valid_folder_list)
                folder_list = sorted(list(useful_folder_list))

            if data_camera_mode == 'ts_animal':
                split_name = './3dgan_data_split/ts_animals/%s.txt' % (split)
                print('==> use ts animal split %s' % (split))
                if split != 'all':
                    valid_folder_list = []
                    with open(split_name, 'r') as f:
                        all_line = f.readlines()
                        for l in all_line:
                            valid_folder_list.append(l.strip())
                    valid_folder_list = set(valid_folder_list)
                    useful_folder_list = set(folder_list).intersection(valid_folder_list)
                    folder_list = sorted(list(useful_folder_list))
            elif data_camera_mode == 'shapenet_motorbike':
                split_name = './3dgan_data_split/shapenet_motorbike/%s.txt' % (split)
                print('==> use ts shapenet motorbike split %s' % (split))
                if split != 'all':
                    valid_folder_list = []
                    with open(split_name, 'r') as f:
                        all_line = f.readlines()
                        for l in all_line:
                            valid_folder_list.append(l.strip())
                    valid_folder_list = set(valid_folder_list)
                    useful_folder_list = set(folder_list).intersection(valid_folder_list)
                    folder_list = sorted(list(useful_folder_list))

            elif data_camera_mode == 'objaverse_multi': #objaverse
                split_name = './3dgan_data_split/objaverse_multi/%s.txt' % (split) #objaverse
                print('==> use objaverse_multi split %s' % (split))
                if split != 'all':
                    valid_folder_list = []
                    with open(split_name, 'r') as f:
                        all_line = f.readlines()
                        for l in all_line:
                            valid_folder_list.append(l.strip())
                    valid_folder_list = set(valid_folder_list)
                    folder_list = sorted(list(valid_folder_list)) 
                    print("DEBUG : SPLIT?")

            print('==> use shapenet folder number %s' % (len(folder_list)))
            folder_list = [os.path.join(root, f) for f in folder_list]
            all_img_list = []
            all_mask_list = []

            for folder in folder_list:
                rgb_list = sorted(os.listdir(folder))
                rgb_list = [n for n in rgb_list if n.endswith('.png') or n.endswith('.jpg')]
                rgb_file_name_list = [os.path.join(folder, n) for n in rgb_list]
                all_img_list.extend(rgb_file_name_list)
                all_mask_list.extend(rgb_list)

            self.img_list = all_img_list
            self.mask_list = all_mask_list

        else:
            raise NotImplementedError
        self.img_size = resolution
        self._type = 'dir'
        self._all_fnames = self.img_list
        self._image_fnames = self._all_fnames
        name = os.path.splitext(os.path.basename(self._path))[0]
        print(
            '==> use image path: %s, num images: %d' % (self.root, len(self._all_fnames)))

        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def __getitem__(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' \
                or self.data_camera_mode == 'renderpeople' \
                or self.data_camera_mode == 'shapenet_motorbike' or self.data_camera_mode == 'ts_house' or self.data_camera_mode == 'ts_animal' or self.data_camera_mode == 'TAPS3D' \
                or self.data_camera_mode == 'objaverse_multi' :
            ori_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            img = ori_img[:, :, :3][..., ::-1]
            mask = ori_img[:, :, 3:4]
            condinfo = np.zeros(2)
            fname_list = fname.split('/')
            img_idx = int(fname_list[-1].split('.')[0])
            obj_idx = fname_list[-2]
            syn_idx = fname_list[-3]
            caption_embedding_path = fname.replace('img','caption_embedding')
            caption_embedding_path = os.path.join(caption_embedding_path[:-8],'caption.npy')
            caption_feature = np.load(caption_embedding_path)
            # print(caption_feature)

            if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' \
                    or self.data_camera_mode == 'renderpeople' or self.data_camera_mode == 'shapenet_motorbike' \
                    or self.data_camera_mode == 'ts_house' or self.data_camera_mode == 'ts_animal'or self.data_camera_mode == 'TAPS3D' \
                        or self.data_camera_mode == 'objaverse_multi':
                # print(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy'))
                if not os.path.exists(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy')):
                    print('==> not found camera root')
                else:
                    rotation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'rotation.npy'))
                    elevation_camera = np.load(os.path.join(self.camera_root, syn_idx, obj_idx, 'elevation.npy'))
                    condinfo[0] = rotation_camera[img_idx] / 180 * np.pi
                    condinfo[1] = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
        else:
            raise NotImplementedError

        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        clip_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        if not mask is None:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            clip_mask = cv2.resize(mask, (224 ,224), interpolation=cv2.INTER_NEAREST)  ########
        else:
            mask = np.ones(1)
        
        img = resize_img.transpose(2, 0, 1) #(c,h,w)

        clip_img = clip_img.transpose(2, 0, 1)
        
        background = np.zeros_like(img)
        clip_img_background = np.zeros_like(clip_img)
    
        img = img * (mask > 0).astype(np.float) + background * (1 - (mask > 0).astype(np.float))
        clip_img = clip_img * (clip_mask > 0).astype(np.float) + clip_img_background * (1 - (clip_mask > 0).astype(np.float))

        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(-1,1,1)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(-1,1,1)
        rand_ind = random.randint(0,19)

        
        return np.ascontiguousarray(((clip_img/225.)-mean)/std), condinfo, np.ascontiguousarray(mask), np.ascontiguousarray(clip_mask), caption_feature[rand_ind], np.ascontiguousarray(img)
        
        # TODO : FIXME return 더 깔끔하게 할 수 있었을텐데 ... 맘이 급했나보구만 ㅎㅅㅎ
        # 1. 어차피 return 하고 나서 중간단에서 F.interpolate 있기 때문에 np를 여기서 막 만들어서 줄 필요가 없다
        # 2. return 하는 값들의 순서가 의미적으로 정렬되지 않았다. 
        # return np.ascontiguousarray(((clip_img/225.)-mean)/std), condinfo, np.ascontiguousarray(mask), fname, caption_feature[rand_ind], np.ascontiguousarray(img)
                # img for making caption following CLIP preprocess / condiinfo / mask / filename / cpationfeature / original img data for GET3D
    
    def _load_raw_image(self, raw_idx):
        if raw_idx >= len(self._image_fnames) or not os.path.exists(self._image_fnames[raw_idx]):
            # print("--------",raw_idx)
            resize_img = np.zeros((3, self.img_size, self.img_size))
            return resize_img

        img = cv2.imread(self._image_fnames[raw_idx])[..., ::-1]
        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) / 255.0

        resize_img = resize_img.transpose(2, 0, 1)
        return resize_img

    # ----------- MINSU -------------- #
    def _load_raw_labels(self):
        return None

    def get_label(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        caption_embedding_path = fname.replace('img','caption_embedding')
        caption_embedding_path = os.path.join(caption_embedding_path[:-8],'caption.npy')
        caption_feature = np.load(caption_embedding_path)
        return caption_feature[0]


class DebugDataset(Dataset):
    def __init__(
            self,
            path,                    # Path to directory or zip.
            camera_path,             # Path to camera
            resolution = None,       # Ensure specific resolution, None = highest available.
            data_camera_mode = None, # what type of dataset are we going to make?
            gen_caption = False,     # set to true if trying to generate caption(cannot call caption_embedding/*/caption_*.npy)
                                     # of gen_caption = True -> then version=GET3D!
            add_camera_cond = False,
            split='all',             # are we using this dataset as training?validation? or just all?
            clip_patch = 16,         # what clip patch size are we going to use?
            version = 'GET3D',       # are we going to train TAPS3D or GET3D?
            debug = False,           # im debugging
            **super_kwargs           # Additional arguments for the Dataset base class.
    ):
        ###debug!!!###
        self.debug = debug
        self.data_camera_mode = data_camera_mode
        assert data_camera_mode in ['shapenet_car', 'shapenet_chair','shapenet_motorbike','objaverse_fruit','objaverse_shoe'] , "invalid data_camera_mode input"
        self._path = path
        self._zipfile = None
        self.root = path
        self.mask_list = None
        self.add_camera_cond = add_camera_cond
        root = self._path
        self.camera_root = camera_path
        self.gen_caption = gen_caption
        if gen_caption:
            assert version == 'GET3D' , "to generate caption, use version = GET3D"

        # additional class variables for TAPS3D
        self.version = version
        assert version == 'GET3D' or version == 'TAPS3D', f"version of dataset should be 'GET3D' or 'TAPS3D' but got {version}"
        if version == 'TAPS3D':
            assert self.gen_caption == False, "TAPS3D dataset cannot be used for gerating captions"
        self.clip_patch = clip_patch
        assert clip_patch == 16 or clip_patch == 32 , f"clip_patch must be 16 or 32 but got {clip_patch}"
        self.rand_ind = 0   # random index for indexing one of 20 captions

        # if root path does not exist
        if not os.path.exists(root):
            print('==> ERROR!!!! THIS SHOULD ONLY HAPPEN WHEN USING INFERENCE')
            n_img = 1234
            self._raw_shape = (n_img, 3, resolution, resolution)
            self.img_size = resolution
            self._type = 'dir'
            self._all_fnames = [None for i in range(n_img)]
            self._image_fnames = self._all_fnames
            name = os.path.splitext(os.path.basename(path))[0]
            print(
                '==> use image path: %s, num images: %d' % (
                    self.root, len(self._all_fnames)))
            super().__init__(name=name, raw_shape=self._raw_shape, **super_kwargs)
            return

        if data_camera_mode == 'shapenet_car' or data_camera_mode == 'shapenet_chair' \
            or data_camera_mode =='shapenet_motorbike':
            print('==> use shapenet dataset')

            ## 기존 GET3D는 .../shapenet_car_result/img/02958343 까지 받음
            ## 근데 이젠 single class, multiclass를 한번에 처리하고 싶기 때문에 .../img 까지만 받음
            folder_list = []
            object_class_list = os.listdir(root)
            for object_class in object_class_list:
                folder_list.extend(os.listdir(os.path.join(root, object_class)))

            folder_list = sorted(folder_list)
            ##
            # folder_list = sorted(os.listdir(root))

            if data_camera_mode == 'shapenet_car':
                split_name = './3dgan_data_split/shapenet_car/%s.txt' % (split)
                if split == 'all':
                    split_name = './3dgan_data_split/shapenet_car.txt'
            elif data_camera_mode == 'shapenet_chair':
                split_name = './3dgan_data_split/shapenet_chair/%s.txt' % (split)
                if split == 'all':
                    split_name = './3dgan_data_split/shapenet_chair.txt'
            #######주의!!!debugging을 위함###
            if self.debug:
                split_name = '.'+split_name
                ########################
            valid_folder_list = []
            with open(split_name, 'r') as f:
                all_line = f.readlines()
                for l in all_line:
                    valid_folder_list.append(l.strip())
            valid_folder_list = set(valid_folder_list)
            useful_folder_list = set(folder_list).intersection(valid_folder_list)
            folder_list = sorted(list(useful_folder_list))

            all_img_list = []
            all_mask_list = []

            print('==> use shapenet folder number %s' % (len(folder_list)))
            for rgb_file in glob.glob(os.path.join(root,'*/*/*.png')):
                if rgb_file.split('/')[-2] in folder_list:
                    all_img_list.append(rgb_file)

            self.img_list = all_img_list
            self.mask_list = all_mask_list

        elif data_camera_mode == 'objaverse_fruit' or data_camera_mode == 'objaverse_shoe':
            print('==> use objaverse dataset')

            folder_list = []
            object_class_list = os.listdir(root)
            for object_class in object_class_list:
                folder_list.extend(os.listdir(os.path.join(root, object_class)))
            folder_list = sorted(folder_list)

            if data_camera_mode == 'objaverse_fruit':
                split_name = './3dgan_data_split/objaverse_fruit/%s.txt' % (split)
                if split == 'all':
                    split_name = './3dgan_data_split/objaverse_fruit.txt'
            elif data_camera_mode == 'objaverse_shoe':
                split_name = './3dgan_data_split/objaverse_shoe/%s.txt' % (split)
                if split == 'all':
                    split_name = './3dgan_data_split/objaverse_shoe.txt'

            #######주의!!!debugging을 위함###
            if self.debug:
                split_name = '.'+split_name
                ########################
            valid_folder_list = []
            with open(split_name, 'r') as f:
                all_line = f.readlines()
                for l in all_line:
                    valid_folder_list.append(l.strip())
            valid_folder_list = set(valid_folder_list)

            useful_folder_list = set(folder_list).intersection(valid_folder_list)
            folder_list = sorted(list(useful_folder_list))
            all_img_list = []
            all_mask_list = []

            print('==> use objaverse folder number %s' % (len(folder_list)))
            for rgb_file in glob.glob(os.path.join(root,'*/*/*/*.png')):
                if rgb_file.split('/')[-3] in folder_list:
                    all_img_list.append(rgb_file)

            self.img_list = all_img_list
            self.mask_list = all_mask_list

        else:
            raise NotImplementedError
        

        self.img_size = resolution
        self._type = 'dir'
        self._all_fnames = self.img_list
        self._image_fnames = self._all_fnames
        name = os.path.splitext(os.path.basename(self._path))[0]
        print(
            '==> use image path: %s, num images: %d' % (self.root, len(self._all_fnames)))
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)


    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    

    def __getitem__(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]

        ori_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        img = ori_img[:, :, :3][..., ::-1]
        mask = ori_img[:, :, 3:4]
        condinfo = np.zeros(2)
        fname_list = fname.split('/')
        img_idx = int(fname_list[-1].split('.')[0])

        if self.version == 'TAPS3D':
            caption_embedding_path = fname.replace('img','caption_embedding')
            if "model" in caption_embedding_path:
                caption_embedding_path = caption_embedding_path.replace("/model","")
            caption_embedding_path = caption_embedding_path.replace(caption_embedding_path.split('/')[-1],'')
            caption_embedding_path = os.path.join(caption_embedding_path,f'caption_{self.clip_patch}.npy')
            caption_feature = np.load(caption_embedding_path)

        # the rotation and the elevation has to change with the data_camera_mode 
        if self.data_camera_mode == 'shapenet_car' or self.data_camera_mode == 'shapenet_chair' or self.data_camera_mode == 'shapenet_motorbike':
            
            syn_idx = fname_list[-2]  
            obj_idx = fname_list[-3]

            if not os.path.exists(os.path.join(self.camera_root, obj_idx, syn_idx,'rotation.npy')):
                print(fname)
                print(os.path.join(self.camera_root, obj_idx, syn_idx,'rotation.npy'))
                print('==> not found camera root')

            else:
                rotation_camera = np.load(os.path.join(self.camera_root, obj_idx, syn_idx, 'rotation.npy'))
                elevation_camera = np.load(os.path.join(self.camera_root, obj_idx, syn_idx, 'elevation.npy'))
                condinfo[0] = rotation_camera[img_idx] / 180 * np.pi
                # MINSU & JUNGBIN
                condinfo[0] = (rotation_camera[img_idx] + 90) / 180 * np.pi
                condinfo[1] = (90 - elevation_camera[img_idx]) / 180.0 * np.pi
                condinfo[0] = -1 * condinfo[0]

        elif self.data_camera_mode == 'objaverse_fruit' or self.data_camera_mode == 'objaverse_shoe':

            syn_idx = fname_list[-3]  
            obj_idx = fname_list[-4]

            if not os.path.exists(os.path.join(self.camera_root, obj_idx, syn_idx,'rotation.npy')):
                print(fname)
                print(os.path.join(self.camera_root, obj_idx, syn_idx,'rotation.npy'))
                print('==> not found camera root')

            else:
                rotation_camera = np.load(os.path.join(self.camera_root, obj_idx, syn_idx, 'rotation.npy'))
                elevation_camera = np.load(os.path.join(self.camera_root, obj_idx, syn_idx, 'elevation.npy'))
                # FIXME : objaverse dataset : ambiguous direction alignment
                condinfo[0] = rotation_camera[img_idx] / 180 * np.pi
                condinfo[1] = (90 - elevation_camera[img_idx]) / 180.0 * np.pi

        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if not mask is None:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.ones(1)
        img = resize_img.transpose(2, 0, 1) #(c,h,w)
        background = np.zeros_like(img)   
        img = img * (mask > 0).astype(np.float32) + background * (1 - (mask > 0).astype(np.float32))

        #----jungbin----#
        # made rand_ind to a self value so that we can get caption through get_caption function
        # rand_ind = random.randint(0,19)

        if self.version == 'TAPS3D': # return values of TAPS3D
            return  np.ascontiguousarray(img), condinfo, np.ascontiguousarray(mask), caption_feature[self.rand_ind]
                # img for making caption following CLIP preprocess / condiinfo / mask / filename / cpationfeature / original img data for GET3D
        else: # return values of GET3D
            if self.gen_caption:
                mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(-1,1,1)
                std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(-1,1,1)
                return np.ascontiguousarray((img/255.-mean)/std), condinfo, np.ascontiguousarray(mask), fname
            return  np.ascontiguousarray(img), condinfo, np.ascontiguousarray(mask)
        

    def _load_raw_image(self, raw_idx):
        if raw_idx >= len(self._image_fnames) or not os.path.exists(self._image_fnames[raw_idx]):
            # print("--------",raw_idx)
            resize_img = np.zeros((3, self.img_size, self.img_size))
            return resize_img

        img = cv2.imread(self._image_fnames[raw_idx])[..., ::-1]
        resize_img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) / 255.0

        resize_img = resize_img.transpose(2, 0, 1)
        return resize_img

    # ----------- MINSU -------------- #
    def _load_raw_labels(self):
        return None
    
    def get_label(self, idx): # get caption as feature which are used as labels
        if self.version == 'TAPS3D' and self.gen_caption == False:
            fname = self._image_fnames[self._raw_idx[idx]]
            caption_embedding_path = fname.replace('img','caption_embedding') 
            if "model" in caption_embedding_path:
                caption_embedding_path = caption_embedding_path.replace("/model","")
            caption_embedding_path = caption_embedding_path.replace(caption_embedding_path.split('/')[-1],'')
            caption_embedding_path = os.path.join(caption_embedding_path,f'caption_{self.clip_patch}.npy')
            caption_feature = np.load(caption_embedding_path)
            return caption_feature[self.rand_ind]
        else:
            return super().get_label(idx)
    
        
    def get_caption(self, idx): # get caption as string
        if self.version == 'TAPS3D' and self.gen_caption == False:
            fname = self._image_fnames[self._raw_idx[idx]]
            caption_embedding_path = fname.replace('img','caption_embedding') 
            if "model" in caption_embedding_path:
                caption_embedding_path = caption_embedding_path.replace("/model","")
            caption_embedding_path = caption_embedding_path.replace(caption_embedding_path.split('/')[-1],'')
            caption_text_path = os.path.join(caption_embedding_path,f'caption_{self.clip_patch}.txt')

            caption_list = []
            with open(caption_text_path, 'r') as f:
                all_line = f.readlines()
            for l in all_line:
                caption_list.append(l)
            
            return caption_list[self.rand_ind]
        else:
            return None
# class Data():
#     def __init__(
#                   version='get3d' # TAPS3D
#                  )
#         self.version = version
#     def __getitem__(self , idx):
#         if self.version = 'get3d':
#             sefl.
            
#     def getitem_get3d(self , idx)

#     def getitem_taps3d(self, idx)


if __name__ == '__main__':
      
    device = "cuda"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--camera_path')
    parser.add_argument('--data_camera_mode')
    parser.add_argument('--split')
    args = parser.parse_args()

    path = args.path
    camera_path = args.camera_path
    data_camera_mode = args.data_camera_mode
    split = args.split
    print('DebugDataset')
    dataset = DebugDataset(path = path, 
                           camera_path= camera_path,
                           resolution = 1024,
                           data_camera_mode = data_camera_mode,
                           gen_caption = False,
                           debug = True,
                           split = split,
                           clip_patch = 16,
                           version = 'TAPS3D',
                           )
    # print('ImageFolderDataset')
    # dataset1 = ImageFolderDataset(path = os.path.join(path,'02958343'), 
    #                        camera_path= camera_path,
    #                        resolution = 1024,
    #                        data_camera_mode = data_camera_mode,
    #                     #    gen_caption = False,
    #                        split = split,
    #                     #    clip_patch = 16,
    #                     #    version = 'GET3D'
    #                        )
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4) 
    print(next(iter(dataloader))[2][0])
    print("dataset[0][0].shape : ", dataset[0][0].shape)
    print("dataset[0][1].shape : ", dataset[0][1].shape)
    print("dataset[0][2].shape : ", dataset[0][2].shape)
    print("dataset[0][3].shape : ", dataset[0][3].shape)
    print("dataset.get_caption : ", dataset.get_caption(0))
    print("dataset.get_label.shape: ", dataset.get_label(0).shape)
    


'''
self,
name,  # Name of the dataset.
raw_shape,  # Shape of the raw image data (NCHW).
max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
use_labels=True,  # Enable conditioning labels? False = label dimension is zero.
xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
random_seed=0, 
path,  # Path to directory or zip.
camera_path,  # Path to camera
resolution=None,  # Ensure specific resolution, None = highest available.
data_camera_mode='apple', #if 'all' use apple, pineapple etc
gen_caption = False, #set to true if trying to generate caption(cannot call embedding)
add_camera_cond=False,
split='all',
label = 16, #32
**super_kwargs 




    training_set_kwargs = {
    "class_name": "",
    "path": "",
    "use_labels": false,
    "max_size": null,
    "xflip": false,
    "resolution": 1024,
    "data_camera_mode": "a_red_apple",
    "add_camera_cond": true,
    "label": 16,
    "camera_path": "./tmp",
    "split": "test",
    "random_seed": 0
    }
'''
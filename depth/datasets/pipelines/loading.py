import json
import mmcv
import numpy as np
import os.path as osp
from PIL import Image
from ..builder import PIPELINES

import cv2
import math
import json
import rasterio
from rasterio.plot import show
from rasterio.fill import fillnodata
from rasterio.features import rasterize

@PIPELINES.register_module()
class LoadKITTICamIntrinsic(object):
    """Load KITTI intrinsic
    """
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        # raw input
        if 'input' in  results['img_prefix']:
            date = results['filename'].split('/')[-5]
            results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]
        # benchmark test
        else:
            temp = results['filename'].replace('benchmark_test', 'benchmark_test_cam')
            cam_file = temp.replace('png', 'txt')
            results['cam_intrinsic'] = np.loadtxt(cam_file).reshape(3, 3).tolist()
        
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class DepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape

        results['depth_fields'].append('depth_gt')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class DisparityLoadAnnotations(object):
    """Load annotations for depth estimation.
    It's only for the cityscape dataset. TODO: more general.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        if results.get('camera_prefix', None) is not None:
            camera_filename = osp.join(results['camera_prefix'],
                                       results['cam_info']['cam_info'])
        else:
            camera_filename = results['cam_info']['cam_info']

        with open(camera_filename) as f:
            camera = json.load(f)
        baseline = camera['extrinsic']['baseline']
        focal_length = camera['intrinsic']['fx']

        disparity = (np.asarray(Image.open(filename), dtype=np.float32) -
                     1.) / results['depth_scale']
        NaN = disparity <= 0

        disparity[NaN] = 1
        depth_map = baseline * focal_length / disparity
        depth_map[NaN] = 0

        results['depth_gt'] = depth_map
        results['depth_ori_shape'] = depth_map.shape

        results['depth_fields'].append('depth_gt')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadSATIMGFromFile(object):
    def __init__(self,
                 to_float32 = True,
                 color_type = 'color',
                 file_client_args = dict(backend='disk'),
                 imdecode_backend = 'rasterio'):
        
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

    def channel_preprocess(self, channel):
        CLAHE = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
        
        channel_min = np.min(channel)
        channel_max = np.max(channel)
        
        channel = (channel-channel_min) / (channel_max-channel_min)
        channel *= 255
        channel = CLAHE.apply(channel.astype(np.uint8))

        return channel

    def __call__(self, results):
        filename = results['img_info']['filename']
        
        img = rasterio.open(filename)
        img = img.read()

        # shape of img : 3xHxW
        img = np.transpose(img, (1,2,0)).astype(np.float16)
        img = img[:,:,[4,2,1]]

        r_c = self.channel_preprocess(img[:,:,0])
        g_c = self.channel_preprocess(img[:,:,1])
        b_c = self.channel_preprocess(img[:,:,2])

        img = np.stack((r_c, g_c, b_c), axis=2).astype(np.float16)

        # shape of img : HxWx3
        results['filename']     = filename
        results['ori_filename'] = filename.split('/')[-1]
        results['img']          = img
        results['img_shape']    = img.shape
        results['ori_shape']    = img.shape
        results['pad_shape']    = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean     = np.zeros(num_channels, dtype = np.float16),
                                       std      = np.ones(num_channels, dtype = np.float16),
                                       to_rgb   = False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadANDProcessNDSM(object):
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='rasterio'):
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        rgb = rasterio.open(results['img_info']['filename'])
        dsm = rasterio.open(results['ann_info']['dsm'])
        dtm = rasterio.open(results['ann_info']['dtm'])

        nodata_m = dtm.read_masks(1)
        nodata_m = nodata_m.astype(np.bool_)

        rgb = rgb.read()
        dtm = dtm.read()
        
        rgb = np.sum(rgb, axis=0)
        eo_nomask = np.ones(rgb.shape)
        eo_nomask[rgb<=0] = False
        eo_nomask = eo_nomask.astype(np.bool_)

        nodata_m = np.logical_and(nodata_m, eo_nomask)

        dsm_mask = dsm.read_masks(1)
        dsm = dsm.read()
        if len(dsm.shape) == 3:
            dsm = dsm[0,:,:]
            dsm_mask = np.ones(dsm_mask.shape)
            dsm_mask[dsm <= 0] = False
            dsm_mask = dsm_mask.astype(np.bool_)
        else:
            pass

        nodata_m = np.logical_and(nodata_m, dsm_mask)
        
        dsm[dsm<0] = 0
        dtm[dtm<0] = 0

        ndsm = dsm-dtm

        results['depth_gt']             = ndsm
        results['depth_ori_shape']      = ndsm.shape
        results['val_mask']             = nodata_m
        results['depth_fields'].append('depth_gt')

        with open(results['ann_info']['meta']) as m:
            meta = json.load(m)
        m.close()

        az_angle = meta['NITF_CSEXRA_AZ_OF_OBLIQUITY']
        delta_x  = math.cos(math.radians(90 + az_angle))
        delta_y  = math.sin(math.radians(90 - az_angle))
        ps_angle = math.degrees(math.atan(delta_y / delta_x))

        results['pose_angle'] = ps_angle
        results['delta_xy'] = [delta_y, delta_x]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str = f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


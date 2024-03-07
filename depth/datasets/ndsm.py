from logging import raiseExceptions
from copy import deepcopy
import os
import csv
import math
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

from PIL import Image
import torch
import json
import glob

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger

from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize
##############################

import rasterio
from rasterio.plot import show
##############################

@DATASETS.register_module()
class nDSMTileDataset(Dataset):
    def __init__(self,
                 pipeline,
                 data_root='Toy_tile',
                 img_dir='RGB',
                 dsm_dir='DSM',
                 dtm_dir='DTM',
                 phase='Training',
                 AoIs=['JAX'],
                 test_mode=False,
                 pose_aligned=True,
                 min_depth = 0,
                 max_depth = 200,
                 depth_scale=1):

        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.path_template = os.path.join(data_root, '{}', '{}', 'single_tile', 'imgs')
        if pose_aligned is True:
            self.dsm_path_template = os.path.join(data_root, '{}', '{}', 'single_tile', 'imgsv2')
        else:
            self.dsm_path_template = os.path.join(data_root, '{}', '{}', 'single_tile', 'imgs')
        self.img_path = img_dir
        self.dsm_path = dsm_dir
        self.dtm_path = dtm_dir
        self.aois = AoIs
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.phase = phase
        self.img_infos = self.load_annotations(self.img_path)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, img_dir):
        img_infos = []

        tr_sample_namelist = 'trlist91.csv'
        te_sample_namelist = 'telist91.csv'
        for aoi in self.aois:
            if self.phase == 'train':
                flist_dir = os.path.join(self.data_root, aoi, img_dir, 'single_tile', tr_sample_namelist)
            else:
                flist_dir = os.path.join(self.data_root, aoi, img_dir, 'single_tile', te_sample_namelist)

            self.img_files = list(csv.reader(open(flist_dir), delimiter=','))[0]
            for img in self.img_files:
                img_info = dict()

                img_info['filename'] = os.path.join(self.path_template.format(aoi, self.img_path), img)
                dsm_name = os.path.join(self.dsm_path_template.format(aoi, self.dsm_path), img)
                dtm_name = os.path.join(self.path_template.format(aoi, self.dtm_path), img)
                meta_name = os.path.join(self.path_template.format(aoi, self.img_path), img).replace('/imgs/','/meta/').replace('.tif', '.json')

                img_info['ann'] = dict(dsm=dsm_name, dtm=dtm_name, meta=meta_name)
                img_info['name'] = img

                img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())


        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']
    
    def __getitem__(self, idx):
        return self.prepare_img(idx)

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        results['depth_fields'] = []
        results['depth_scale'] = 200
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """ Placeholder to format result to dataset specific output. """
        raise NotImplementedError

    def get_gt_dsm(self):
        dsms = []
        nodata_masks = []
        
        for img_info in self.img_infos:
            rgb = rasterio.open(img_info['filename'])
            dsm = rasterio.open(img_info['ann']['dsm'])
            dtm = rasterio.open(img_info['ann']['dtm'])
            
            nodata_mask = dtm.read_masks(1)
            nodata_mask = nodata_mask.astype(np.bool_)
            
            rgb = rgb.read()
            dtm = dtm.read()
            
            #### N/A Handling for NA value in the EO Image
            rgb = np.sum(rgb, axis=0)
            eo_nomask = np.ones(rgb.shape)
            eo_nomask[rgb <= 0] = False
            eo_nomask = eo_nomask.astype(np.bool_)
            
            nodata_m = np.logical_and(nodata_m, eo_nomask)
            
            final_mask = dsm.read_masks(1)
            dsm = dsm.read()
    
            if len(dsm.shape) == 3:#### Version for Pose Aligned DSM Dataset
                dsm = dsm[0,:,:]

                final_mask = np.ones((nodata_mask.shape))
                final_mask[dsm <= 0] = False
                final_mask = final_mask.astype(np.bool_)

                nodata_m = np.logical_and(nodata_m, final_mask)
            else:#### Version for Orthographic DSM Dataset
                nodata_m = np.logical_and(nodata_m, final_mask)

            #### nDSM Calculation
            # ndsm shape -> HxW
            dsm[dsm<0] = 0
            dtm[dtm<0] = 0

            ndsm = dsm-dtm
            
            dsms.append(ndsm)
            nodata_masks.append(nodata_m)

            with open(img_info['ann']['meta']) as m:
                meta = json.load(m)
            m.close()

            az_angle = meta['NITF_CSEXRA_AZ_OF_OBLIQUITY']
            delta_x  = math.cos(math.radians(90+az_angle))
            delta_y  = math.sin(math.radians(90-az_angle))
            ps_angle = math.degrees(math.atan(delta_y/delta_x))
        
        # dsms shape -> NxHxW
        # nodata masks shape -> NxHxW
        return dsms, nodata_masks, [delta_y, delta_x], ps_angle

    def pre_eval(self, preds, indices):
        def eval_mask(nodata_mask, dsm, dsm_mask):
            final_mask = dsm_mask

            if len(dsm.shape) == 3:
                dsm_tmp = deepcopy(dsm[0,:,:])

                final_mask = np.ones(nodata_mask.shape)
                final_mask[dsm_tmp <= 0] = False
                final_mask = final_mask.astype(np.bool_)
            else:
                pass

            nodata_mask = np.logical_and(nodata_mask, final_mask)
            nodata_mask = np.expand_dims(nodata_mask, axis=0)
            
            # nodata mask shape -> 1xHxW
            return nodata_mask

        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds   = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            img_info = self.img_infos[index]
            rgb      = rasterio.open(img_info['filename'])
            dsm      = rasterio.open(img_info['ann']['dsm'])
            dtm      = rasterio.open(img_info['ann']['dtm'])
            
            print('rgb data : ', rgb.read().shape)

            nodata_mask = dtm.read_masks(1)
            nodata_mask = nodata_mask.astype(np.bool_)

            rgb = rgb.read()
            rgb = np.sum(rgb, axis=0)
            eo_nomask = np.ones(rgb.shape)
            eo_nomask[rgb <= 0] = False
            eo_nomask = eo_nomask.astype(np.bool_)

            nodata_m = np.logical_and(nodata_mask, eo_nomask)
            
            dsm_ndmask = dsm.read_masks(1)
            dsm = dsm.read()
            dtm = dtm.read()

            valid_mask = eval_mask(nodata_m, dsm, dsm_ndmask)
            
            if len(dsm.shape) == 3:
                dsm = dsm[0,:,:]
            dsm[dsm<0] = 0
            dtm[dtm<0] = 0

            ndsm = dsm-dtm
            print('ndsm shape : ', ndsm.shape)
            print('pred shape : ', pred.shape)
            print('val mask shape : ', valid_mask.shape)
            eval = metrics(ndsm[valid_mask], pred[valid_mask], self.min_depth, self.max_depth)
            pre_eval_results.append(eval)

            pre_eval_preds.append(eval)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='rmse', logger=None, **kwargs):
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        eval_results = {}

        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            gt_dsms, val_masks = self.get_gt_dsm()
            # gt_dsms shape -> NxHxW
            # val masks shape -> NxHxW

            ret_metrics = eval_metrics(gt_dsms, results, val_masks, min_depth=self.min_depth, max_depth=self.max_depth)
        else:
            ret_metrics = pre_eval_to_metrics(results)

        ret_metric_names    = []
        ret_metric_values   = []

        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metric) // 9

        for i in range(num_table):
            names   = ret_metric_names[i*9: i*9 + 9]
            values  = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results






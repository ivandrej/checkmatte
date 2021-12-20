"""

python evaluate_experiment.py \
    --pred-dir ~/dev/data/composited_evaluation/VideoMatte5x3/out \
    --experiment-metadata experiment_metadata/VMxDVM_0013.json


A .csv file will be written in the /out directory
"""
import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pims
import torch
from PIL import Image
from tqdm import tqdm

import composite
from evaluation_metrics import MetricMAD, MetricMSE, MetricGRAD, MetricDTSSD


# Returns args object
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, required=True)
    parser.add_argument('--experiment-metadata', type=str, required=True)
    parser.add_argument('--resize', type=int, required=True, nargs=2)
    parser.add_argument('--num-workers', type=int, default=48)
    parser.add_argument('--metrics', type=str, nargs='+', default=[
        'pha_mad', 'pha_mse', 'pha_grad', 'pha_dtssd', 'fgr_mad', 'fgr_mse'])
    return parser.parse_args()



class Evaluator:
    def __init__(self, args):
        self.args = args
        self.init_metrics()
        self.evaluate()
        # self.write_excel()
        self.write_csv()

    def init_metrics(self):
        self.mad = MetricMAD()
        self.mse = MetricMSE()
        self.grad = MetricGRAD()
        self.dtssd = MetricDTSSD()

    def evaluate(self):
        tasks = []

        clips = composite.read_metadata(args.experiment_metadata)
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            for clip in clips:
                future = executor.submit(self.evaluate_worker_img_seq, clip.bgr_type, clip)
                tasks.append((clip.bgr_type, clip, future))

        self.results = [(dataset, clip.clipname, future.result()) for dataset, clip, future in tasks]

    # TODO: Update description of method
    """
        Output table has the format: clipname, mad, mse, ....
        output = {
         clip1: [pha_mad_mean, mse_mean, ..., fgr_mad_mean
         ]
        } 
    """
    def write_csv(self):
        output_dict = {}
        pha_mad_dict = {}
        max_num_frames = 0
        # Each row is one clip
        for row, (dataset, clipname, metrics) in enumerate(self.results):
            pha_mad_dict[clipname] = np.array(metrics['pha_mad']).astype(float)

            max_num_frames = max(max_num_frames, len(metrics['pha_mad']))
            metric_mean_over_frames = list(map(lambda metric_values: np.mean(metric_values), metrics.values()))
            output_dict[clipname] = metric_mean_over_frames
            output_dict[clipname] = metric_mean_over_frames + [dataset]

        columns = list(self.results[0][2].keys()) + ["bgr_type"]
        df = pd.DataFrame.from_dict(output_dict, orient="index", columns=columns)

        df.loc["dynamic_mean"] = df.loc[df.bgr_type == "dynamic"].mean()
        df.loc["semi_dynamic_mean"] = df.loc[df.bgr_type == "semi_dynamic"].mean()
        df.loc["static_mean"] = df.loc[df.bgr_type == "static"].mean()

        df.to_csv(os.path.join(self.args.pred_dir, "metrics.csv"), index_label="clipname")

        # A column for each frame
        pha_mad_columns = list(range(max_num_frames))
        df_pha_mad = pd.DataFrame.from_dict(pha_mad_dict, orient="index", columns=pha_mad_columns)
        df_pha_mad.to_csv(os.path.join(self.args.pred_dir, "pha_mad.csv"), index_label="clipname")

    def evaluate_worker(self, dataset, clip: composite.CompositedClipPaths):
        print("Clip: ", clip.clipname)
        true_fgr_frames = pims.PyAVVideoReader(clip.fgr_path)
        true_pha_frames = pims.PyAVVideoReader(clip.pha_path)

        pred_fgr_frames = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, clip.clipname, "fgr.mp4"))
        pred_pha_frames = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, clip.clipname, "pha.mp4"))

        assert (len(true_fgr_frames) == len(true_pha_frames))
        assert (len(pred_fgr_frames) == len(pred_pha_frames))

        metrics = {metric_name: [] for metric_name in self.args.metrics}

        pred_pha_tm1 = None
        true_pha_tm1 = None

        num_frames = min(len(true_fgr_frames), len(pred_fgr_frames))
        for t in tqdm(range(num_frames), desc=f'{dataset} {clip.clipname}'):
            pred_pha = video_frame_to_torch(pred_pha_frames[t], grayscale=True)
            true_pha = video_frame_to_torch(true_pha_frames[t], grayscale=True, resize=args.resize)

            # print(pred_pha.shape, true_pha.shape)
            assert (true_pha.shape == pred_pha.shape)
            if 'pha_mad' in self.args.metrics:
                metrics['pha_mad'].append(self.mad(pred_pha, true_pha))
            if 'pha_mse' in self.args.metrics:
                metrics['pha_mse'].append(self.mse(pred_pha, true_pha))
            if 'pha_grad' in self.args.metrics:
                metrics['pha_grad'].append(self.grad(pred_pha, true_pha))
            if 'pha_conn' in self.args.metrics:
                metrics['pha_conn'].append(self.conn(pred_pha, true_pha))
            if 'pha_dtssd' in self.args.metrics:
                if t == 0:
                    metrics['pha_dtssd'].append(0)
                else:
                    metrics['pha_dtssd'].append(self.dtssd(pred_pha, pred_pha_tm1, true_pha, true_pha_tm1))

            pred_pha_tm1 = pred_pha
            true_pha_tm1 = true_pha

            if 'fgr_mse' in self.args.metrics or 'fgr_mda' in self.args.metrics:
                pred_fgr = video_frame_to_torch(pred_fgr_frames[t])
                true_fgr = video_frame_to_torch(true_fgr_frames[t], resize=args.resize)
                true_msk = true_pha > 0

                if 'fgr_mse' in self.args.metrics:
                    metrics['fgr_mse'].append(self.mse(pred_fgr[true_msk], true_fgr[true_msk]))
                if 'fgr_mad' in self.args.metrics:
                    metrics['fgr_mad'].append(self.mad(pred_fgr[true_msk], true_fgr[true_msk]))

        return metrics

    def evaluate_worker_img_seq(self, dataset, clip: composite.CompositedClipPaths):
        print("Clip: ", clip.clipname)
        framenames = sorted(os.listdir(os.path.join(self.args.pred_dir, clip.clipname, "pha")))
        # remove extension from framename, e.g "00000.png" --> "00000"
        framenames = [name.split('.')[0] for name in framenames]

        metrics = {metric_name: [] for metric_name in self.args.metrics}

        pred_pha_tm1 = None
        true_pha_tm1 = None

        for t, framename in enumerate(tqdm(framenames, desc=f'{clip.clipname}', dynamic_ncols=True)):
            true_pha = torch.from_numpy(
                np.asarray(cv2.resize(cv2.imread(os.path.join(clip.pha_path, "0" + framename + ".jpg"),
                                                 cv2.IMREAD_GRAYSCALE), tuple(self.args.resize)))).float().div_(255)
            pred_pha = torch.from_numpy(
                np.asarray(cv2.imread(os.path.join(self.args.pred_dir, clip.clipname, 'pha', framename + ".png"),
                                      cv2.IMREAD_GRAYSCALE))).float().div_(255)

            assert (true_pha.shape == pred_pha.shape)
            if 'pha_mad' in self.args.metrics:
                metrics['pha_mad'].append(self.mad(pred_pha, true_pha))
            if 'pha_mse' in self.args.metrics:
                metrics['pha_mse'].append(self.mse(pred_pha, true_pha))
            if 'pha_grad' in self.args.metrics:
                metrics['pha_grad'].append(self.grad(pred_pha, true_pha))
            if 'pha_conn' in self.args.metrics:
                metrics['pha_conn'].append(self.conn(pred_pha, true_pha))
            if 'pha_dtssd' in self.args.metrics:
                if t == 0:
                    metrics['pha_dtssd'].append(0)
                else:
                    metrics['pha_dtssd'].append(self.dtssd(pred_pha, pred_pha_tm1, true_pha, true_pha_tm1))

            pred_pha_tm1 = pred_pha
            true_pha_tm1 = true_pha

            if 'fgr_mse' in self.args.metrics or 'fgr_mda' in self.args.metrics:
                true_fgr = torch.from_numpy(
                    np.asarray(cv2.resize(cv2.imread(os.path.join(clip.pha_path, "0" + framename + ".jpg"),
                                                     cv2.IMREAD_COLOR), tuple(self.args.resize)))).float().div_(255)
                pred_fgr = torch.from_numpy(
                    np.asarray(cv2.imread(os.path.join(self.args.pred_dir, clip.clipname, 'fgr', framename + ".png"),
                                          cv2.IMREAD_COLOR))).float().div_(255)
                true_msk = true_pha > 0

                if 'fgr_mse' in self.args.metrics:
                    metrics['fgr_mse'].append(self.mse(pred_fgr[true_msk], true_fgr[true_msk]))
                if 'fgr_mad' in self.args.metrics:
                    metrics['fgr_mad'].append(self.mad(pred_fgr[true_msk], true_fgr[true_msk]))

        return metrics


def video_frame_to_torch(video_frame, resize=None, grayscale=False):
    img_frame = Image.fromarray(video_frame)
    if grayscale:
        img_frame = img_frame.convert("L")

    if resize is not None:
        img_frame = img_frame.resize(resize, Image.BILINEAR)
    return torch.from_numpy(np.asarray(img_frame)).float().div_(255)


if __name__ == '__main__':
    args = parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        Evaluator(args)

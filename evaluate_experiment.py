"""
HR (High-Resolution) evaluation. We found using numpy is very slow for high resolution, so we moved it to PyTorch using CUDA.

Note, the script only does evaluation. You will need to first inference yourself and save the results to disk
Expected directory format for both prediction and ground-truth is:

I THINK THIS FILE STRUCTURE IS WRONG, 
    videomatte_1920x1080
        ├── videomatte_motion
          ├── pha
            ├── 0000
              ├── 0000.png
          ├── fgr
            ├── 0000
              ├── 0000.png
        ├── videomatte_static
          ├── pha
            ├── 0000
              ├── 0000.png
          ├── fgr
            ├── 0000
              ├── 0000.png

Prediction must have the exact file structure and file name as the ground-truth,
meaning that if the ground-truth is png/jpg, prediction should be png/jpg.

Example usage:

python evaluate.py \
    --pred-dir pred/videomatte_1920x1080 \
    --true-dir true/videomatte_1920x1080
    
An excel sheet with evaluation results will be written to "pred/videomatte_1920x1080/videomatte_1920x1080.xlsx"
"""
import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pims
import torch
import xlsxwriter
from PIL import Image
from tqdm import tqdm

from composite import fgr_path_of_com, pha_path_of_com
from evaluation_metrics import MetricMAD, MetricMSE, MetricGRAD, MetricDTSSD


# Returns args object
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, required=True)
    # parser.add_argument('--true-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=48)
    parser.add_argument('--metrics', type=str, nargs='+', default=[
        'pha_mad', 'pha_mse', 'pha_grad', 'pha_dtssd', 'fgr_mse'])
    return parser.parse_args()


# TODO: Take this from argument
res = (512, 288)


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
        position = 0

        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            # for dataset in sorted(os.listdir(self.args.pred_dir)):
            dataset = "dataset"
            print("Fgr paths: ", fgr_path_of_com)
            for clip in sorted(os.listdir(os.path.join(self.args.pred_dir))):
                if os.path.isdir(os.path.join(self.args.pred_dir, clip)):
                    future = executor.submit(self.evaluate_worker, dataset, clip, position)
                    tasks.append((dataset, clip, future))
                    position += 1

        self.results = [(dataset, clip, future.result()) for dataset, clip, future in tasks]

    """
        Output table has the format: clipname, mad, mse, ....
        output = {
         clip1: [mean mad over frame, mean mse over frames]
        } 
    """

    def write_csv(self):
        # Each row is one clip
        output_dict = {}
        for row, (dataset, clip, metrics) in enumerate(self.results):
            print(metrics)
            metric_mean_over_frames = list(map(lambda metric_values: np.mean(metric_values), metrics.values()))
            output_dict[clip] = metric_mean_over_frames

        df = pd.DataFrame.from_dict(output_dict, orient="index", columns=self.results[0][2].keys())
        df.to_csv(os.path.join(self.args.pred_dir, "metrics.csv"), index_label="clipname")

    def evaluate_worker(self, dataset, clip, position):
        print("Clip: ", clip)
        true_fgr_frames = pims.PyAVVideoReader(fgr_path_of_com[clip])
        true_pha_frames = pims.PyAVVideoReader(pha_path_of_com[clip])

        pred_fgr_frames = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, clip, "fgr.mp4"))
        pred_pha_frames = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, clip, "pha.mp4"))

        assert (len(true_fgr_frames) == len(true_pha_frames))
        assert (len(pred_fgr_frames) == len(pred_pha_frames))

        metrics = {metric_name: [] for metric_name in self.args.metrics}

        pred_pha_tm1 = None
        true_pha_tm1 = None

        num_frames = min(len(true_fgr_frames), len(pred_fgr_frames))
        for t in tqdm(range(num_frames), desc=f'{dataset} {clip}'):
            true_pha = video_frame_to_numpy(true_pha_frames[t], grayscale=True)
            pred_pha = video_frame_to_numpy(pred_pha_frames[t], grayscale=True)

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

            if 'fgr_mse' in self.args.metrics:
                true_fgr = video_frame_to_numpy(true_fgr_frames[t])
                pred_fgr = video_frame_to_numpy(pred_fgr_frames[t])

                true_msk = true_pha > 0
                metrics['fgr_mse'].append(self.mse(pred_fgr[true_msk], true_fgr[true_msk]))

        return metrics


def video_frame_to_numpy(video_frame, grayscale=False):
    img_frame = Image.fromarray(video_frame)
    if grayscale:
        img_frame = img_frame.convert("L")

    resized_frame = img_frame.resize(res, Image.BILINEAR)
    return torch.from_numpy(np.asarray(resized_frame)).float().div_(255)


if __name__ == '__main__':
    args = parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("deprecated", DeprecationWarning)
        Evaluator(args)

"""

python evaluate_experiment.py \
    --pred-dir ~/dev/data/composited_evaluation/VideoMatte5x3/out


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
import xlsxwriter
from PIL import Image
from tqdm import tqdm

from evaluation_metrics import MetricMAD, MetricMSE, MetricGRAD, MetricDTSSD


# Returns args object
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--num-workers', type=int, default=48)
    parser.add_argument('--metrics', type=str, nargs='+', default=[
        'pha_mad', 'pha_mse', 'pha_grad', 'pha_dtssd', 'fgr_mad', 'fgr_mse'])
    return parser.parse_args()



class Evaluator:
    def __init__(self, args):
        self.args = args
        self.init_metrics()
        self.evaluate()
        self.write_excel()
        # self.write_csv()

    def init_metrics(self):
        self.mad = MetricMAD()
        self.mse = MetricMSE()
        self.grad = MetricGRAD()
        self.dtssd = MetricDTSSD()

    def evaluate(self):
        tasks = []
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            for clip_name in os.listdir(os.path.join(self.args.experiment_dir, "input")):
                # pred_pha_path = os.path.join(self.args.experiment_dir, "out", clip_folder, "pha")
                # pred_fgr_path = os.path.join(self.args.experiment_dir, "out", clip_folder, "fgr")
                #
                # true_pha_path = os.path.join(self.args.experiment_dir, "input", clip_folder, "pha")
                # true_fgr_path = os.path.join(self.args.experiment_dir, "input", clip_folder, "fgr")

                future = executor.submit(self.evaluate_worker, "dataset", clip_name)
                tasks.append(("dataset", clip_name, future))

        self.results = [(dataset, clip_name, future.result()) for dataset, clip_name, future in tasks]

    def write_excel(self):
        workbook = xlsxwriter.Workbook(os.path.join(self.args.experiment_dir, "out.xlsx"))
        print("Writing Excel: ", workbook.filename)
        summarysheet = workbook.add_worksheet('summary')
        metricsheets = [workbook.add_worksheet(metric) for metric in self.results[0][2].keys()]

        for i, metric in enumerate(self.results[0][2].keys()):
            summarysheet.write(i, 0, metric)
            summarysheet.write(i, 1, f'={metric}!B2')

        for row, (dataset, clip, metrics) in enumerate(self.results):
            for metricsheet, metric in zip(metricsheets, metrics.values()):
                # Write the header
                if row == 0:
                    metricsheet.write(1, 0, 'Average')
                    metricsheet.write(1, 1, f'=AVERAGE(C2:ZZ2)')
                    for col in range(len(metric)):
                        metricsheet.write(0, col + 2, col)
                        colname = xlsxwriter.utility.xl_col_to_name(col + 2)
                        metricsheet.write(1, col + 2, f'=AVERAGE({colname}3:{colname}9999)')

                metricsheet.write(row + 2, 0, dataset)
                metricsheet.write(row + 2, 1, clip)
                metricsheet.write_row(row + 2, 2, metric)

        workbook.close()

    def evaluate_worker(self, dataset, clip_name):
        # true_fgr_frames = pims.PyAVVideoReader(clip.fgr_path)
        # true_pha_frames = pims.PyAVVideoReader(clip.pha_path)
        #
        # pred_fgr_frames = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, clip.clipname, "fgr.mp4"))
        # pred_pha_frames = pims.PyAVVideoReader(os.path.join(self.args.pred_dir, clip.clipname, "pha.mp4"))
        #
        # assert (len(true_fgr_frames) == len(true_pha_frames))
        # assert (len(pred_fgr_frames) == len(pred_pha_frames))
        print(clip_name)
        framenames = sorted(os.listdir(os.path.join(self.args.experiment_dir, "input", clip_name, "pha")))
        metrics = {metric_name: [] for metric_name in self.args.metrics}

        pred_pha_tm1 = None
        true_pha_tm1 = None

        # num_frames = min(len(true_fgr_frames), len(pred_fgr_frames))
        for i, framename in enumerate(tqdm(framenames, desc=f'{clip_name}', dynamic_ncols=True)):
            true_pha = torch.from_numpy(np.asarray(cv2.imread(os.path.join(self.args.experiment_dir, "input", clip_name, 'pha', framename),
                                  cv2.IMREAD_GRAYSCALE))).float().div_(255)
            pred_pha = torch.from_numpy(np.asarray(cv2.imread(os.path.join(self.args.experiment_dir, "out", clip_name, 'pha', framename),
                                  cv2.IMREAD_GRAYSCALE))).float().div_(255)

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
                if i == 0:
                    metrics['pha_dtssd'].append(0)
                else:
                    metrics['pha_dtssd'].append(self.dtssd(pred_pha, pred_pha_tm1, true_pha, true_pha_tm1))

            pred_pha_tm1 = pred_pha
            true_pha_tm1 = true_pha

            if 'fgr_mse' in self.args.metrics or 'fgr_mda' in self.args.metrics:
                true_fgr = torch.from_numpy(
                    np.asarray(cv2.imread(os.path.join(self.args.experiment_dir, "input", clip_name, 'fgr', framename),
                                          cv2.IMREAD_GRAYSCALE))).float().div_(255)
                pred_fgr = torch.from_numpy(
                    np.asarray(cv2.imread(os.path.join(self.args.experiment_dir, "out", clip_name, 'fgr', framename),
                                          cv2.IMREAD_GRAYSCALE))).float().div_(255)

            true_msk = true_pha > 0

            if 'fgr_mse' in self.args.metrics:
                metrics['fgr_mse'].append(self.mse(pred_fgr[true_msk], true_fgr[true_msk]))
            if 'fgr_mad' in self.args.metrics:
                metrics['fgr_mad'].append(self.mad(pred_fgr[true_msk], true_fgr[true_msk]))

        return metrics


def video_frame_to_numpy(video_frame, resize=None, grayscale=False):
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

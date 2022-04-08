import argparse
import os

import pandas as pd

import evaluate_experiment
import perform_experiment
from composite import read_metadata


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-metadata', type=str, required=True)
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--load-model', type=str, required=True)
    parser.add_argument('--model-type', type=str, choices=['addition', 'concat', 'f3'], default='addition')
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random-bgr', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    input_dir = args.input_dir

    clips = read_metadata(args.experiment_metadata)
    # if args.random_bgr:
    #     clips = replace_bgr_with_random_bgr(clips)
    temporal_offsets = [10, 30, 50, 80]
    rotations = [(0, 0), (5, 10), (25, 30)]

    for temporal_offset in temporal_offsets:
        for rotation in rotations:
            out_dir = os.path.join(args.experiment_dir, f"offset{temporal_offset}_rotation{[rotation[0],rotation[1]]}")
            print(f"Saving results in {out_dir}")
            print("Performing inference...")
            perform_experiment.inference(out_dir, args.model_type, args.load_model,
                                         input_dir, clips, args.resize, output_type='png_sequence',
                                         bgr_offset=temporal_offset, bgr_rotation=rotation)

            print("Performing evaluation...")
            evaluate_experiment.Evaluator(out_dir, args.experiment_metadata, args.num_workers, args.resize,
                                  metrics=['pha_mad', 'pha_bgr_mad', 'pha_fgr_mad'])

    summary = {'misalignment': [], 'pha_mad': [], 'pha_bgr_mad': [], 'pha_fgr_mad': []}
    for misalignment_type in os.listdir(args.experiment_dir):
        df = pd.read_csv(os.path.join(args.experiment_dir, misalignment_type, 'metrics.csv')).set_index("clipname")
        summary['misalignment'].append(misalignment_type)
        # print(df)
        summary['pha_mad'].append(df.loc['mean', 'pha_mad'])
        summary['pha_bgr_mad'].append(df.loc['mean', 'pha_bgr_mad'])
        summary['pha_fgr_mad'].append(df.loc['mean', 'pha_fgr_mad'])

    pd.DataFrame.from_dict(summary).to_csv(os.path.join(args.experiment_dir, 'summary.csv'))
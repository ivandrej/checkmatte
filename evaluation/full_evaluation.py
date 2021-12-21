"""
    python full_evaluation.py --experiment-metadata experiment_metadata/VMxDVM_0013.json
    --experiment-dir /media/andivanov/DATA/experiments/training_iteration_1/VMxDVM_0013_specialized_2epoch_3VM
    --load-model /media/andivanov/DATA/training/specialized_iteration_1/2_epochs_3_vm_clips/checkpoint/stage1/epoch-1.pth
    --resize 512 288
    --num-frames 600

"""
import argparse
import os

import composite
import evaluate_experiment
import perform_experiment


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-metadata', type=str, required=True)
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--load-model', type=str, required=True)
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=48)
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    input_dir = os.path.join(args.experiment_dir, "input")
    out_dir = os.path.join(args.experiment_dir, "out")
    print("Composing foregrounds onto backgrounds...")
    composite.composite_fgrs_to_bgrs(input_dir, args.experiment_metadata, args)

    print("Performing inference...")
    perform_experiment.inference(args.experiment_dir, args.load_model)

    print("Performing evaluation...")
    evaluate_experiment.Evaluator(out_dir, args.experiment_metadata, args.num_workers, args.resize)

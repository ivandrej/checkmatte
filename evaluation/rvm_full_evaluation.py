"""
    python full_evaluation.py --experiment-metadata experiment_metadata/VMxDVM_0013.json
    --experiment-dir /media/andivanov/DATA/experiments/training_iteration_1/VMxDVM_0013_specialized_2epoch_3VM
    --load-model /media/andivanov/DATA/training/specialized_iteration_1/2_epochs_3_vm_clips/checkpoint/stage1/epoch-1.pth
    --resize 512 288
    --num-frames 600

"""
import argparse
import os

import evaluate_experiment
import rvm_perform_experiment


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-metadata', type=str, required=True)
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--resize', type=int, required=True, nargs=2)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--skip-compose', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    input_dir = args.input_dir
    out_dir = os.path.join(args.experiment_dir, "out")

    print("Performing inference...")
    rvm_perform_experiment.inference(args.experiment_dir, input_dir, args.resize, load_model=args.model)

    print("Performing evaluation...")
    evaluate_experiment.Evaluator(out_dir, args.experiment_metadata, args.num_workers, args.resize,
                                  metrics=['pha_mad', 'pha_bgr_mad', 'pha_fgr_mad'])

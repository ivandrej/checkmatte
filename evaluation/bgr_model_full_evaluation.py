"""
    python full_evaluation.py --experiment-metadata experiment_metadata/VMxDVM_0013.json
    --experiment-dir /media/andivanov/DATA/experiments/training_iteration_1/VMxDVM_0013_specialized_2epoch_3VM
    --load-model /media/andivanov/DATA/training/specialized_iteration_1/2_epochs_3_vm_clips/checkpoint/stage1/epoch-1.pth
    --resize 512 288
    --num-frames 600

"""
import argparse
import os

from composite import read_metadata
import evaluate_experiment
import perform_experiment


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-metadata', type=str, required=True)
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--load-model', type=str, required=True)
    parser.add_argument('--model-type', type=str, choices=['addition', 'concat', 'f3'], default='addition')
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--temporal-offset', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--skip-compose', action="store_true")
    parser.add_argument('--random-bgr', action="store_true")
    return parser.parse_args()

def replace_bgr_with_random_bgr(clips):
    leonhardstrasse_bgr = '/media/andivanov/DATA/dynamic_backgrounds_captured/img_seq/leonhardstrasse_building'
    for clip in clips:
        clip.bgr_path = leonhardstrasse_bgr

    return clips

if __name__ == "__main__":
    args = read_args()
    input_dir = args.input_dir
    out_dir = os.path.join(args.experiment_dir)

    clips = read_metadata(args.experiment_metadata)
    if args.random_bgr:
        clips = replace_bgr_with_random_bgr(clips)

    print("Performing inference...")
    perform_experiment.inference(args.experiment_dir, args.model_type, args.load_model, input_dir, clips, args.resize,
                                 output_type='png_sequence', bgr_offset=args.temporal_offset)

    print("Performing evaluation...")
    evaluate_experiment.Evaluator(out_dir, args.experiment_metadata, args.num_workers, args.resize,
                                  metrics=['pha_mad', 'pha_bgr_mad', 'pha_fgr_mad'])

import time
import argparse
import numpy as np
import os
import snakes2

class Args:
    def __init__(self,
                 contour_path='./contour', batch_id=0, n_images=200000,
                 window_size=[256, 256], padding=22, antialias_scale=4,
                 LABEL=1, seed_distance=27, marker_radius=3,
                 contour_length=15, distractor_length=5, num_distractor_snakes=6, snake_contrast_list=[1.], use_single_paddles=True,
                 max_target_contour_retrial=4, max_distractor_contour_retrial=4, max_paddle_retrial=2,
                 continuity=1.4, paddle_length=5, paddle_thickness=1.5, paddle_margin_list=[4], paddle_contrast_list=[1.],
                 pause_display=False, save_images=True, save_metadata=True):

        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images

        self.window_size = window_size
        self.padding = padding
        self.antialias_scale = antialias_scale

        self.LABEL = LABEL
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        self.distractor_length = distractor_length
        self.num_distractor_snakes = num_distractor_snakes
        self.snake_contrast_list = snake_contrast_list
        self.use_single_paddles = use_single_paddles

        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial

        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list
        self.paddle_contrast_list = paddle_contrast_list

        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Snake generation parameters')

    parser.add_argument('--num_machines', type=int, default=1, help='Number of machines (default: 1)')
    parser.add_argument('--current_id', type=int, default=1, help='Current batch ID (default: 1)')
    parser.add_argument('--total_images', type=int, default=100000, help='Total number of images (default: 100,000)')
    parser.add_argument('--dataset_root', type=str, default='./data/',
                        help='Dataset root path (default: ./data/')

    args_parsed = parser.parse_args()

    t = time.time()
    args = Args()

    # Set batch_id and number of images
    num_machines = args_parsed.num_machines
    current_id = args_parsed.current_id
    total_images = args_parsed.total_images
    dataset_root = args_parsed.dataset_root

    args.batch_id = current_id
    args.n_images = total_images / num_machines

    # Configure other parameters
    args.antialias_scale = 4
    args.paddle_margin_list = [3]
    args.window_size = [1024, 1024]
    args.marker_radius = 5
    args.contour_length = 64
    args.paddle_thickness = 2
    args.continuity = 1.6
    args.distractor_length = args.contour_length
    args.num_distractor_snakes = 5
    args.snake_contrast_list = [1.0]
    args.use_single_paddles = False
    args.segmentation_task = False
    args.segmentation_task_double_circle = False

    ################################# DS: BASELINE
    dataset_subpath = 'curv_baseline'
    args.contour_path = os.path.join(dataset_root, dataset_subpath)
    snakes2.from_wrapper(args)

    elapsed = time.time() - t
    print('n_totl_imgs (per condition):', str(total_images))
    print('ELAPSED TIME OVER ALL CONDITIONS:', str(elapsed))

if __name__ == '__main__':
    main()

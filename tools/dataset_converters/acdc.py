import argparse
import glob
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image

# ACDC dataset color palette (RGB values for each class)
ACDC_palette = {
    0: [0, 0, 0],  # Background (unrecognized classes)
    1: [171, 171, 171],  # LV
    2: [114, 114, 114],  # MYO
    3: [57, 57, 57]  # RV
}

# Inverse palette mapping (RGB tuple -> class index)
ACDC_invert_palette = {tuple(v): k for k, v in ACDC_palette.items()}


def ACDC_convert_from_color(arr_3d, palette=ACDC_invert_palette):
    """Convert RGB annotation images to single-channel class index maps.

    Args:
        arr_3d (np.ndarray): Input RGB annotation image (H,W,3)
        palette (dict): Mapping from RGB tuples to class indices

    Returns:
        np.ndarray: Single-channel label map (H,W) with class indices
    """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for rgb, class_id in palette.items():
        # Create boolean mask for pixels matching current RGB
        mask = np.all(arr_3d == np.array(rgb).reshape(1, 1, 3), axis=2)
        arr_2d[mask] = class_id

    return arr_2d


def process_annotations(src_dir, out_dir):
    """Process ACDC annotation dataset and convert to mmsegmentation format.

    Args:
        src_dir (str): Path to source annotations directory
        out_dir (str): Path to output directory
    """
    # Create output directories if they don't exist
    mkdir_or_exist(osp.join(out_dir, 'train'))
    mkdir_or_exist(osp.join(out_dir, 'val'))

    # Process both training and validation sets
    for mode in ['training', 'validation']:
        # Get all PNG annotation files for current mode
        src_path_list = glob.glob(osp.join(src_dir, mode, '*.png'))
        prog_bar = ProgressBar(len(src_path_list))

        for img_path in src_path_list:
            # Read RGB annotation image (channel order: RGB)
            label = mmcv.imread(img_path, channel_order='rgb')

            # Convert to single-channel class indices
            label_idx = ACDC_convert_from_color(label)

            # Prepare output path (maintain original filename)
            basename = osp.basename(img_path)
            save_path = osp.join(out_dir, 'train' if mode == 'training' else 'val', basename)

            # Save as single-channel PNG
            Image.fromarray(label_idx).save(save_path)

            # Update progress bar
            prog_bar.update()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert ACDC annotations to mmsegmentation format')
    parser.add_argument(
        '--src-dir',
        default='data/acdc/ann_dir',
        help='Path to ACDC annotations directory')
    parser.add_argument(
        '--out-dir',
        default='data/acdc/ann_dir',
        help='Output directory for processed data')
    return parser.parse_args()


def main():
    """Main function for annotation conversion."""
    args = parse_args()
    print(f'Processing ACDC annotations from {args.src_dir}...')

    # Process all annotations
    process_annotations(args.src_dir, args.out_dir)

    print('Conversion complete! Output saved to:', args.out_dir)


if __name__ == '__main__':
    main()
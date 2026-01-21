import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from pathlib import Path

import torch.nn.functional as F
from droid import Droid
from droid_async import DroidAsync


def show_image(image):
    # image: [3,H,W]
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(datapath, maskdir=None, mask_ext=".png", mask_suffix=""):
    """TUM RGB image generator with optional masks.

    Returns a list of tuples:
        (t, image[None], intrinsics, dynmask)

    where dynmask is either:
        - torch.uint8 [H,W] with 1=dynamic, 0=static
        - or None if no mask for that frame
    """

    # Fixed TUM freiburg intrinsics/distortion (as in original DROID script)
    #fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
    K_l = np.array([fx, 0.0, cx,
                    0.0, fy, cy,
                    0.0, 0.0, 1.0]).reshape(3, 3)
    d_l = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])

    # read every 2nd rgb frame (as in original)
    images_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::2]

    data_list = []
    for t, imfile in enumerate(images_list):
        image = cv2.imread(imfile)
        if image is None:
            raise RuntimeError(f"Failed to read image: {imfile}")

        # load mask if available
        dynmask = None
        if maskdir is not None:
            stem = os.path.splitext(os.path.basename(imfile))[0]
            mpath = os.path.join(maskdir, stem + mask_suffix + mask_ext)
            if os.path.isfile(mpath):
                m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    dynmask = m

        # undistort image
        ht0, wd0, _ = image.shape
        image = cv2.undistort(image, K_l, d_l)

        # undistort mask using same K and d if present
        if dynmask is not None:
            dynmask = cv2.undistort(dynmask, K_l, d_l)

        # resize to 320+32 x 240+16 as original
        target_w = 320 + 32
        target_h = 240 + 16

        image = cv2.resize(image, (target_w, target_h))
        image_t = torch.from_numpy(image).permute(2, 0, 1)  # [3,H,W]

        if dynmask is not None:
            dynmask = cv2.resize(dynmask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # compute intrinsics scaling
        intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        intrinsics[0] *= image_t.shape[2] / 640.0
        intrinsics[1] *= image_t.shape[1] / 480.0
        intrinsics[2] *= image_t.shape[2] / 640.0
        intrinsics[3] *= image_t.shape[1] / 480.0

        # crop image to remove distortion boundary (same as original)
        intrinsics[2] -= 16
        intrinsics[3] -= 8
        image_t = image_t[:, 8:-8, 16:-16]  # [3,Hc,Wc]

        if dynmask is not None:
            dynmask = dynmask[8:-8, 16:-16]
            dynmask = torch.from_numpy((dynmask > 0).astype(np.uint8))  # [Hc,Wc], 1=dynamic

        data_list.append((t, image_t[None], intrinsics, dynmask))

    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=1.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.0)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    parser.add_argument("--motion_damping", type=float, default=0.5)

    # NEW: mask options
    parser.add_argument("--maskdir", type=str, default=None,
                        help="directory with dynamic masks aligned to rgb filenames.")
    parser.add_argument("--mask_ext", type=str, default=".png",
                        help="mask extension (default .png)")
    parser.add_argument("--mask_suffix", type=str, default="",
                        help="optional suffix before extension, e.g. '_mask'")

    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = DroidAsync(args) if args.asynchronous else Droid(args)
    scene = Path(args.datapath).name

    # load images (+optional masks) once, reuse for tracking and terminate
    images = image_stream(args.datapath, maskdir=args.maskdir,
                          mask_ext=args.mask_ext, mask_suffix=args.mask_suffix)

    for (t, image, intrinsics, dynmask) in tqdm(images, desc=scene):
        if not args.disable_vis:
            show_image(image[0])
        droid.track(t, image, intrinsics=intrinsics, dynmask=dynmask)

    # terminate() expects an iterable of the same tuples;
    # your PoseTrajectoryFiller should handle 3- or 4-tuples gracefully.
    traj_est = droid.terminate(images)

    ### run evaluation ###

    print("#" * 20 + " Results...")

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, 'rgb')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::2]
    tstamps = [float(os.path.basename(x)[:-4]) for x in images_list]

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:],
        timestamps=np.array(tstamps)
    )

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(
        traj_ref, traj_est, est_name='traj',
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True   # monocular -> allow scale correction
    )

    print(result)

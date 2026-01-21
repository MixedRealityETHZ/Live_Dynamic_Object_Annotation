import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse

import torch.nn.functional as F
from droid import Droid
from droid_async import DroidAsync


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def _load_mask(maskdir, rgb_path, mask_ext=".png", mask_suffix=""):
    """Load mask matching ETH3D rgb filename stem. Returns numpy uint8 or None."""
    if maskdir is None:
        return None
    stem = os.path.splitext(os.path.basename(rgb_path))[0]
    mpath = os.path.join(maskdir, stem + mask_suffix + mask_ext)
    if not os.path.isfile(mpath):
        return None
    m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    return m


def image_stream(datapath, use_depth=False, stride=1, maskdir=None, mask_ext=".png", mask_suffix=""):
    """ETH3D image generator with optional depth + optional dynamic mask.

    Expected ETH3D layout:
      datapath/
        calibration.txt
        rgb/*.png
        depth/*.png   (only if use_depth=True)

    Yields:
      if use_depth: (tstamp, image[None], depth, intrinsics, dynmask)
      else:         (tstamp, image[None], intrinsics, dynmask)

    dynmask is torch.uint8 [H,W] with 1=dynamic, 0=static, or None.
    """
    fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, "calibration.txt")).tolist()
    image_list = sorted(glob.glob(os.path.join(datapath, "rgb", "*.png")))[::stride]

    if len(image_list) == 0:
        raise RuntimeError(f"No RGB images found in {os.path.join(datapath, 'rgb')}")

    if use_depth:
        depth_list = sorted(glob.glob(os.path.join(datapath, "depth", "*.png")))[::stride]
        if len(depth_list) == 0:
            raise RuntimeError(f"--depth set but no depth images found in {os.path.join(datapath, 'depth')}")
        if len(depth_list) != len(image_list):
            raise RuntimeError(f"RGB/Depth count mismatch: {len(image_list)} rgb vs {len(depth_list)} depth")
        pairs = zip(image_list, depth_list)
    else:
        pairs = [(im, None) for im in image_list]

    for rgb_path, depth_path in pairs:
        # ETH3D timestamps are the rgb filename stems
        tstamp = float(os.path.splitext(os.path.basename(rgb_path))[0])

        image = cv2.imread(rgb_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {rgb_path}")

        dynmask_np = _load_mask(maskdir, rgb_path, mask_ext=mask_ext, mask_suffix=mask_suffix)

        depth = None
        if use_depth and depth_path is not None:
            d = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if d is None:
                raise RuntimeError(f"Failed to read depth: {depth_path}")
            depth = d / 5000.0  # ETH3D depth scale

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        # resize/crop image
        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image_t = torch.as_tensor(image).permute(2, 0, 1)

        # resize/crop depth if present (nearest is fine for depth)
        depth_t = None
        if depth is not None:
            depth_t = torch.as_tensor(depth)
            depth_t = F.interpolate(depth_t[None, None], (h1, w1), mode="nearest")[0, 0]
            depth_t = depth_t[: h1 - h1 % 8, : w1 - w1 % 8]

        # resize/crop dynmask if present (nearest to keep binary)
        dynmask_t = None
        if dynmask_np is not None:
            dynmask_np = cv2.resize(dynmask_np, (w1, h1), interpolation=cv2.INTER_NEAREST)
            dynmask_np = dynmask_np[: h1 - h1 % 8, : w1 - w1 % 8]
            dynmask_t = torch.from_numpy((dynmask_np > 0).astype(np.uint8))  # [H,W], 1=dynamic

        # intrinsics scale
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if use_depth:
            yield tstamp, image_t[None], depth_t, intrinsics, dynmask_t
        else:
            yield tstamp, image_t[None], intrinsics, dynmask_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--depth", action="store_true", help="use depth for tracking (expects datapath/depth/*.png)")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--motion_damping", type=float, default=0.5)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    # NEW: dynamic mask options
    parser.add_argument("--maskdir", type=str, default=None,
                        help="directory containing dynamic masks aligned to rgb filename stems (timestamp).")
    parser.add_argument("--mask_ext", type=str, default=".png",
                        help="mask extension (default: .png)")
    parser.add_argument("--mask_suffix", type=str, default="",
                        help="optional suffix before extension (e.g. '_mask')")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    print(f"Running evaluation on {args.datapath}")
    print(args)

    stride = 1
    droid = None

    # --- tracking ---
    if args.depth:
        track_stream = image_stream(
            args.datapath, use_depth=True, stride=stride,
            maskdir=args.maskdir, mask_ext=args.mask_ext, mask_suffix=args.mask_suffix
        )
        for (tstamp, image, depth, intrinsics, dynmask) in tqdm(track_stream):
            if not args.disable_vis:
                show_image(image[0])

            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = DroidAsync(args) if args.asynchronous else Droid(args)

            # dynmask can be None; your Droid/MotionFilter/DepthVideo should handle it
            droid.track(tstamp, image, depth, intrinsics=intrinsics, dynmask=dynmask)
    else:
        track_stream = image_stream(
            args.datapath, use_depth=False, stride=stride,
            maskdir=args.maskdir, mask_ext=args.mask_ext, mask_suffix=args.mask_suffix
        )
        for (tstamp, image, intrinsics, dynmask) in tqdm(track_stream):
            if not args.disable_vis:
                show_image(image[0])

            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = DroidAsync(args) if args.asynchronous else Droid(args)

            droid.track(tstamp, image, intrinsics=intrinsics, dynmask=dynmask)

    if droid is None:
        raise RuntimeError("No frames were read. Check datapath/rgb and (if --depth) datapath/depth.")

    # --- terminate / pose filling ---
    traj_est = droid.terminate(
        image_stream(
            args.datapath, use_depth=False, stride=stride,
            maskdir=args.maskdir, mask_ext=args.mask_ext, mask_suffix=args.mask_suffix
        )
    )

    # --- evaluation (ATE / APE translation) ---
    print("#" * 20 + " Results...")

    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    # timestamps from rgb filenames (ETH3D convention)
    rgb_files = sorted(glob.glob(os.path.join(args.datapath, "rgb", "*.png")))[::stride]
    tstamps = np.array([float(os.path.splitext(os.path.basename(x))[0]) for x in rgb_files], dtype=np.float64)

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:],
        timestamps=tstamps
    )

    gt_file = os.path.join(args.datapath, "groundtruth.txt")
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(
        traj_ref, traj_est, est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=False
    )

    print(result)

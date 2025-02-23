import tinycudann as tcnn
import copy
import gc
import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
from PIL import Image
import functools
import math
from typing import Sequence, Callable
from multiprocessing.pool import ThreadPool

##############################################################################
# 1) Specify your device (GPU or CPU).
##############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

scene_type = "synthetic"
object_name = "chair"
scene_dir = "datasets/nerf_synthetic/"+object_name

with open('config_hash.json') as config_file:
	config = json.load(config_file)
with open('config_mlp.json') as config_file:
    config_mlp = json.load(config_file)

weights_dir = "weights"
samples_dir = "samples"
if not os.path.exists(weights_dir):
  os.makedirs(weights_dir)
if not os.path.exists(samples_dir):
  os.makedirs(samples_dir)
##############################################################################
# 2) Utility for writing float images via OpenCV (on CPU).
##############################################################################
def write_floatpoint_image(name, img_tensor):
    """
    img_tensor: Float in [0,1], shape [H, W, 3], on any device.
    """
    img_cpu = img_tensor.detach().cpu().numpy()  # Move to CPU, convert to NumPy
    img_cpu = (img_cpu * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(name, img_cpu[:, :, ::-1])  # BGR -> RGB flip with [::-1]

#%% --------------------------------------------------------------------------------
# ## Load the dataset
#%%
# """ Load dataset """

if scene_type=="synthetic":
  white_bkgd = True



##############################################################################
# 5) Data-loading function -> returns Torch Tensors on the chosen device
##############################################################################
if scene_type == "synthetic":

    def load_blender(data_dir, split):
        with open(os.path.join(data_dir, f"transforms_{split}.json"), "r") as fp:
            meta = json.load(fp)

        cams = []
        paths = []
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            # We'll store camera matrices in a list
            cam_mat = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            cams.append(cam_mat)

            fname = os.path.join(data_dir, frame["file_path"] + ".png")
            paths.append(fname)

        def image_read_fn(fname):
            with open(fname, "rb") as imgin:
                img_np = np.array(Image.open(imgin), dtype=np.float32) / 255.0
            return torch.from_numpy(img_np)  # shape [H, W, 4] if RGBA

        # Parallel load images
        with ThreadPool() as pool:
            images = pool.map(image_read_fn, paths)
            pool.close()
            pool.join()

        # Stack => shape [N, H, W, 4]
        images = torch.stack(images, dim=0)

        # Convert RGBA -> RGB, optionally with white background
        if white_bkgd:
            # images[..., :3] * alpha + (1 - alpha)
            rgb = images[..., :3] * images[..., 3:] + (1.0 - images[..., 3:])
        else:
            # images[..., :3] * alpha
            rgb = images[..., :3] * images[..., 3:]
        images = rgb  # shape [N, H, W, 3], on device

        # Compute camera intrinsics
        N, H, W = images.shape[0], images.shape[1], images.shape[2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * W / math.tan(0.5 * camera_angle_x)

        hwf = torch.tensor([H, W, focal], dtype=torch.float32, device=device)
        poses = torch.stack(cams, dim=0).to(device)  # shape [N, 4, 4]

        return {"images": images, "c2w": poses, "hwf": hwf}

    # Actually load the data
    data = {
        "train": load_blender(scene_dir, "train"),
        "test": load_blender(scene_dir, "test"),
    }

    # Just a quick shape check
    for s in ["train", "test"]:
        print(s)
        for k in data[s]:
            print(f"  {k}: {data[s][k].shape}")

    images, poses, hwf = data["train"]["images"], data["train"]["c2w"], data["train"]["hwf"]
    # Save one sample image -> shape [H, W, 3]
    write_floatpoint_image(samples_dir + "/training_image_sample.png", images[0])

    # Quick camera position scatter
    for i in range(3):
        plt.figure()
        x_vals = poses[:, i, 3].detach().cpu().numpy()
        y_vals = poses[:, (i + 1) % 3, 3].detach().cpu().numpy()
        plt.scatter(x_vals, y_vals)
        plt.axis("equal")
        plt.savefig(samples_dir + f"/training_camera{i}.png")
        plt.close()


def matmul(a, b):
    return torch.matmul(a, b)

def normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def sinusoidal_encoding(position, minimum_frequency_power,
                        maximum_frequency_power, include_identity=False):
    """
    position: shape [..., D], on device
    """
    freqs = 2.0 ** torch.arange(minimum_frequency_power,
                                maximum_frequency_power,
                                dtype=position.dtype,
                                device=position.device)
    # angle => [..., num_freqs, D]
    angle = position.unsqueeze(-2) * freqs.unsqueeze(-1)
    # sin_cos => [..., 2, num_freqs, D]
    sin_cos = torch.stack([torch.sin(angle), torch.sin(angle + 0.5 * math.pi)], dim=-3)
    # Flatten
    sin_cos = sin_cos.view(*position.shape[:-1], -1)

    if include_identity:
        sin_cos = torch.cat([position, sin_cos], dim=-1)
    return sin_cos


##############################################################################
# 7) Ray generation (Torch)
##############################################################################
def generate_rays(pixel_coords, pix2cam, cam2world):
    """
    pixel_coords: shape [..., 2], on device
    pix2cam: shape [3, 3], on device
    cam2world: shape [..., 3, 4], on device
    """
    homog = torch.ones_like(pixel_coords[..., :1])  # same shape as pixel_coords[..., :1]
    pixel_dirs = torch.cat([pixel_coords + 0.5, homog], dim=-1)[..., None]  # => [..., 3, 1]

    # [..., 3, 1]
    cam_dirs = matmul(pix2cam.unsqueeze(0), pixel_dirs)
    # => [..., 3]
    ray_dirs = matmul(cam2world[..., :3, :3], cam_dirs)[..., 0]
    # broadcast origin
    ray_origins = cam2world[..., :3, 3].expand_as(ray_dirs)
    return ray_origins, ray_dirs

def pix2cam_matrix(height, width, focal, device):
    return torch.tensor([
        [1.0 / focal,       0.0,          -0.5 * width / focal],
        [0.0,         -1.0 / focal,  0.5 * height / focal],
        [0.0,               0.0,                    -1.0],
    ], dtype=torch.float32, device=device)

def camera_ray_batch(cam2world, hwf):
    """
    cam2world: shape [..., 4, 4], on device
    hwf: [H, W, focal], on device
    """
    H, W, f = hwf[0].item(), hwf[1].item(), hwf[2].item()
    pix2cam = pix2cam_matrix(H, W, f, cam2world.device)

    # create meshgrid on device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=cam2world.device, dtype=torch.float32),
        torch.arange(W, device=cam2world.device, dtype=torch.float32),
        indexing="ij"
    )
    pixel_coords = torch.stack([xx, yy], dim=-1)  # shape [H, W, 2]
    return generate_rays(pixel_coords, pix2cam, cam2world)

def random_ray_batch(batch_size, data):
    images_cpu = data["images"]   # shape [N, H, W, 3] on CPU
    c2w = data["c2w"].to(device)  # shape [N, 4, 4] on GPU
    hwf = data["hwf"].to(device)

    N, H, W, _ = images_cpu.shape

    # Pick random indices
    cam_ind = torch.randint(low=0, high=N, size=(batch_size,), device=device)
    y_ind   = torch.randint(low=0, high=H, size=(batch_size,), device=device)
    x_ind   = torch.randint(low=0, high=W, size=(batch_size,), device=device)

    # Grab pixel values on CPU, then send those values to GPU
    # (We index the CPU tensor via numpy-like advanced indexing.)
    # Careful: we need .cpu() indexing or .numpy() indexing if 'cam_ind' is on GPU.
    # One approach is to do the indexing with plain Python or move 'cam_ind' to CPU:
    cam_ind_cpu = cam_ind.cpu()
    y_ind_cpu   = y_ind.cpu()
    x_ind_cpu   = x_ind.cpu()

    pixels_cpu  = images_cpu[cam_ind_cpu, y_ind_cpu, x_ind_cpu]  # still on CPU
    pixels      = pixels_cpu.to(device)                          # move to GPU

    # The next part: generating the rays
    # We'll keep pix2cam on GPU
    # etc...
    pix2cam = pix2cam_matrix(hwf[0].item(), hwf[1].item(), hwf[2].item(), device)

    chosen_c2w = c2w[cam_ind, :3, :4]  # [batch_size, 3, 4] on GPU

    # pixel_coords => also needs to be on GPU
    # but 'x_ind' and 'y_ind' are already on GPU (assuming you did torch.randint(..., device=device)).
    pixel_coords = torch.stack([x_ind, y_ind], dim=-1).float()  # [batch_size, 2]
    # Now generate rays on GPU
    rays = generate_rays(pixel_coords, pix2cam, chosen_c2w)

    return rays, pixels

# Learning rate helpers.
##############################################################################
# 8) Learning-rate schedule (Torch)
##############################################################################
def log_lerp(t, v0, v1):
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f"Interpolants {v0} and {v1} must be positive.")
    # Work as torch Tensors
    t_ = torch.tensor(t, dtype=torch.float32, device=device)
    v0_ = torch.tensor(v0, dtype=torch.float32, device=device)
    v1_ = torch.tensor(v1, dtype=torch.float32, device=device)
    t_clamped = torch.clamp(t_, 0.0, 1.0)
    lv0 = torch.log(v0_)
    lv1 = torch.log(v1_)
    return torch.exp(t_clamped * (lv1 - lv0) + lv0)

def lr_fn(step, max_steps, lr0, lr1, lr_delay_steps=20000, lr_delay_mult=0.1):
    step_t = torch.tensor(step, dtype=torch.float32, device=device)
    max_steps_t = torch.tensor(max_steps, dtype=torch.float32, device=device)
    if lr_delay_steps > 0:
        ratio = torch.clamp(step_t / lr_delay_steps, 0.0, 1.0)
        delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * torch.sin(0.5 * math.pi * ratio)
    else:
        delay_rate = torch.tensor(1.0, device=device)

    base_lr = log_lerp(step_t / max_steps_t, lr0, lr1)
    return delay_rate * base_lr

##############################################################################
# 9) Scene Setup / Grid Tensors on device
##############################################################################
if scene_type == "synthetic":
    scene_grid_scale = 1.2
    if "hotdog" in scene_dir or "mic" in scene_dir or "ship" in scene_dir:
        scene_grid_scale = 1.5

    grid_min = torch.tensor([-1, -1, -1], dtype=torch.float32, device=device) * scene_grid_scale
    grid_max = torch.tensor([ 1,  1,  1], dtype=torch.float32, device=device) * scene_grid_scale
    point_grid_size = 64

    def get_taper_coord(p):
        return p

    def inverse_taper_coord(p):
        return p

grid_dtype = torch.float16

# plane parameter grid (on device)
point_grid = torch.zeros(
    (point_grid_size, point_grid_size, point_grid_size, 3),
    dtype=grid_dtype,
    device=device
)
acc_grid = torch.zeros(
    (point_grid_size, point_grid_size, point_grid_size),
    dtype=grid_dtype,
    device=device
)
point_grid_diff_lr_scale = 16.0 / point_grid_size

##############################################################################
# 10) Additional helpers on device
##############################################################################
def get_acc_grid_masks(taper_positions, acc_grid):
    """
    taper_positions: [..., 3] on device
    acc_grid: [X, Y, Z] on device
    """
    grid_positions = (taper_positions - grid_min) * (point_grid_size / (grid_max - grid_min))

    grid_masks = (
        (grid_positions[..., 0] >= 1) & (grid_positions[..., 0] < point_grid_size - 1) &
        (grid_positions[..., 1] >= 1) & (grid_positions[..., 1] < point_grid_size - 1) &
        (grid_positions[..., 2] >= 1) & (grid_positions[..., 2] < point_grid_size - 1)
    )

    # zero out OOB positions
    grid_positions_in = grid_positions * grid_masks.unsqueeze(-1)
    grid_indices = grid_positions_in.long()

    ix = grid_indices[..., 0]
    iy = grid_indices[..., 1]
    iz = grid_indices[..., 2]

    # gather from acc_grid
    acc_vals = acc_grid[ix, iy, iz]  # shape [...]
    return acc_vals * grid_masks
##############################################################################
# 11) Example function gridcell_from_rays on device
##############################################################################
if scene_type == "synthetic":

    def gridcell_from_rays(rays, acc_grid, keep_num, threshold):
        """
        rays: (ray_origins, ray_directions), each [..., 3] on device
        acc_grid: [X, Y, Z] on device
        keep_num: int
        threshold: float
        """
        ray_origins, ray_directions = rays
        small_step = 1e-5
        epsilon = 1e-5

        ox = ray_origins[..., 0:1]
        oy = ray_origins[..., 1:2]
        oz = ray_origins[..., 2:3]

        dx = ray_directions[..., 0:1]
        dy = ray_directions[..., 1:2]
        dz = ray_directions[..., 2:3]

        dxm = (dx.abs() < epsilon).float()
        dym = (dy.abs() < epsilon).float()
        dzm = (dz.abs() < epsilon).float()

        dx = dx + dxm
        dy = dy + dym
        dz = dz + dzm

        # layers => shape [point_grid_size+1], broadcast to [..., point_grid_size+1]
        layers = torch.arange(point_grid_size + 1, dtype=ray_origins.dtype, device=ray_origins.device)
        layers = layers / point_grid_size
        batch_shape = ray_origins.shape[:-1]
        for _ in batch_shape:
            layers = layers.unsqueeze(0)
        layers = layers.expand(*batch_shape, point_grid_size + 1)

        gx_min, gy_min, gz_min = grid_min[0].item(), grid_min[1].item(), grid_min[2].item()
        gx_max, gy_max, gz_max = grid_max[0].item(), grid_max[1].item(), grid_max[2].item()

        tx = ((layers * (gx_max - gx_min) + gx_min) - ox) / dx
        ty = ((layers * (gy_max - gy_min) + gy_min) - oy) / dy
        tz = ((layers * (gz_max - gz_min) + gz_min) - oz) / dz

        tx = tx * (1 - dxm) + 1000 * dxm
        ty = ty * (1 - dym) + 1000 * dym
        tz = tz * (1 - dzm) + 1000 * dzm

        txyz = torch.stack([tx, ty, tz], dim=-1)

        # txyz <= 0 => set to large
        neg_mask = (txyz <= 0).float()
        txyz = txyz * (1 - neg_mask) + 1000 * neg_mask

        # + small_step
        txyz = txyz + small_step
        # print("ray_origins shape:", ray_origins.shape)
        # print("txyz shape:", txyz.shape)
        # print("ray_origins.unsqueeze(-2) shape:", ray_origins.unsqueeze(-2).shape)
        # print("txyz.unsqueeze(-1) shape:", txyz.unsqueeze(-1).shape)
        # print("ray.unsqueeze(-2 )", ray_directions.unsqueeze(-2).shape)
        # world_positions => [..., point_grid_size+1, 3]
        world_positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * txyz
        acc_grid_masks = get_acc_grid_masks(world_positions, acc_grid)

        # remove empty cells
        below_thresh_mask = (acc_grid_masks < threshold).float().unsqueeze(-1)
        txyz = txyz * (1 - below_thresh_mask) + 1000 * below_thresh_mask

        # sort along last axis
        txyz_sorted, _ = torch.sort(txyz, dim=-1)
        # keep first 'keep_num'
        txyz_clipped = txyz_sorted[..., :keep_num]
        # print("TXYZ CLIPPED ", txyz_clipped.shape)
        # Recompute positions
        world_positions_final = (
            ray_origins.unsqueeze(-2) +
            ray_directions.unsqueeze(-2) * txyz_clipped
        )

        # Convert to grid coords
        grid_positions = (world_positions_final - grid_min) * (point_grid_size / (grid_max - grid_min))
        grid_masks = (
            (grid_positions[..., 0] >= 1) & (grid_positions[..., 0] < point_grid_size - 1) &
            (grid_positions[..., 1] >= 1) & (grid_positions[..., 1] < point_grid_size - 1) &
            (grid_positions[..., 2] >= 1) & (grid_positions[..., 2] < point_grid_size - 1)
        )

        # If out-of-bounds => clamp or set safe positions
        grid_positions_safe = grid_positions * grid_masks.unsqueeze(-1) + (~grid_masks).unsqueeze(-1)
        grid_indices = grid_positions_safe.long()

        return grid_indices, grid_masks



#get barycentric coordinates
#P = (p1-p3) * a + (p2-p3) * b + p3
#P = d * t + o
##############################################################################
# 12) Barycentric (Torch)
##############################################################################
def get_barycentric(p1, p2, p3, O, d):
    """
    p1, p2, p3, O, d => shape [..., 3], device
    """
    epsilon = 1e-10
    r1 = p1 - p3
    r2 = p2 - p3

    Ox, Oy, Oz = O[..., 0], O[..., 1], O[..., 2]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    r1x, r1y, r1z = r1[..., 0], r1[..., 1], r1[..., 2]
    r2x, r2y, r2z = r2[..., 0], r2[..., 1], r2[..., 2]
    p3x, p3y, p3z = p3[..., 0], p3[..., 1], p3[..., 2]

    denominator = (
          - dx * r1y * r2z
          + dx * r1z * r2y
          + dy * r1x * r2z
          - dy * r1z * r2x
          - dz * r1x * r2y
          + dz * r1y * r2x
    )
    denom_mask = (denominator.abs() < epsilon)
    denominator = denominator + denom_mask.float()

    a_num = (
          (Ox - p3x) * dy * r2z
        + (p3x - Ox) * dz * r2y
        + (p3y - Oy) * dx * r2z
        + (Oy - p3y) * dz * r2x
        + (Oz - p3z) * dx * r2y
        + (p3z - Oz) * dy * r2x
    )
    b_num = (
          (p3x - Ox) * dy * r1z
        + (Ox - p3x) * dz * r1y
        + (Oy - p3y) * dx * r1z
        + (p3y - Oy) * dz * r1x
        + (p3z - Oz) * dx * r1y
        + (Oz - p3z) * dy * r1x
    )

    a = a_num / denominator
    b = b_num / denominator
    c = 1.0 - (a + b)

    mask = (a >= 0) & (b >= 0) & (c >= 0) & (~denom_mask)
    return a, b, c, mask

##############################################################################
# 13) Cell size variables on device
##############################################################################
cell_size_x = (grid_max[0] - grid_min[0]) / point_grid_size
cell_size_y = (grid_max[1] - grid_min[1]) / point_grid_size
cell_size_z = (grid_max[2] - grid_min[2]) / point_grid_size

half_cell_size_x = cell_size_x / 2
half_cell_size_y = cell_size_y / 2
half_cell_size_z = cell_size_z / 2

neg_half_cell_size_x = -half_cell_size_x
neg_half_cell_size_y = -half_cell_size_y
neg_half_cell_size_z = -half_cell_size_z

def get_inside_cell_mask(P, ooxyz):
    """
    P, ooxyz => shape [..., 3] on device
    """
    P_ = get_taper_coord(P) - ooxyz
    return (
        (P_[..., 0] >= neg_half_cell_size_x) &
        (P_[..., 0] <  half_cell_size_x)     &
        (P_[..., 1] >= neg_half_cell_size_y) &
        (P_[..., 1] <  half_cell_size_y)     &
        (P_[..., 2] >= neg_half_cell_size_z) &
        (P_[..., 2] <  half_cell_size_z)
    )

def compute_undc_intersection(
    point_grid,               # shape [..., X, Y, Z, 3], on device
    cell_xyz,                 # shape [..., 3], on device (int indices)
    masks,                    # shape [...], boolean or float, on device
    rays,                     # Tuple(Tensor, Tensor) -> (ray_origins, ray_directions), each [..., 3]
    keep_num                 # int
):
    """
    PyTorch version of 'compute_undc_intersection', fully on a chosen device.

    Returns:
        taper_positions: intersection points in "taper" coords, shape [..., keep_num, 3]
        world_masks:     boolean mask for valid intersections, shape [..., keep_num]
        ooo_times_mask:  shape [..., 3], the product of ooo_ and masks[..., None]
        world_tx:        sorted intersection distances, shape [..., keep_num], .detach()'ed
    """
    ray_origins, ray_directions = rays
    device = ray_origins.device
    dtype = ray_origins.dtype

    # Convert cell_xyz to float if needed for arithmetic.
    # We'll keep them as int for advanced indexing. (That's okay.)

    # 1) Base offset (ooxyz) for each cell
    #    cell_xyz: [..., 3], so cell_xyz.float() -> shape [..., 3] on device
    ooxyz = (cell_xyz.float() + 0.5) * ((grid_max - grid_min) / point_grid_size) + grid_min
    # shape: [..., 3], on device

    cell_x = cell_xyz[..., 0].long()
    cell_y = cell_xyz[..., 1].long()
    cell_z = cell_xyz[..., 2].long()

    # Offsets for neighboring cells (±1)
    cell_x1 = cell_x + 1
    cell_y1 = cell_y + 1
    cell_z1 = cell_z + 1
    cell_x0 = cell_x - 1
    cell_y0 = cell_y - 1
    cell_z0 = cell_z - 1

    # 2) Gather corners from point_grid
    # Ensure 'point_grid' is shape [X, Y, Z, 3], or similar,
    # and we do advanced indexing with (cell_x, cell_y, cell_z).
    # Then add the offset ooxyz, call inverse_taper_coord(...).

    # origin
    ooo_ = point_grid[cell_x, cell_y, cell_z] * point_grid_diff_lr_scale  # [..., 3]
    ooo = inverse_taper_coord(ooo_ + ooxyz)  # [..., 3]

    # We'll define a quick helper to create offset Tensors on the correct device/dtype
    def offset_tensor(x, y, z):
        return torch.tensor([x, y, z], dtype=dtype, device=device)

    obb = inverse_taper_coord(
        point_grid[cell_x, cell_y0, cell_z0] * point_grid_diff_lr_scale
        + offset_tensor(0, -cell_size_y, -cell_size_z)
        + ooxyz
    )
    obd = inverse_taper_coord(
        point_grid[cell_x, cell_y0, cell_z1] * point_grid_diff_lr_scale
        + offset_tensor(0, -cell_size_y,  cell_size_z)
        + ooxyz
    )
    odb = inverse_taper_coord(
        point_grid[cell_x, cell_y1, cell_z0] * point_grid_diff_lr_scale
        + offset_tensor(0,  cell_size_y, -cell_size_z)
        + ooxyz
    )
    odd = inverse_taper_coord(
        point_grid[cell_x, cell_y1, cell_z1] * point_grid_diff_lr_scale
        + offset_tensor(0,  cell_size_y,  cell_size_z)
        + ooxyz
    )
    obo = inverse_taper_coord(
        point_grid[cell_x, cell_y0, cell_z] * point_grid_diff_lr_scale
        + offset_tensor(0, -cell_size_y, 0)
        + ooxyz
    )
    oob = inverse_taper_coord(
        point_grid[cell_x, cell_y, cell_z0] * point_grid_diff_lr_scale
        + offset_tensor(0, 0, -cell_size_z)
        + ooxyz
    )
    odo = inverse_taper_coord(
        point_grid[cell_x, cell_y1, cell_z] * point_grid_diff_lr_scale
        + offset_tensor(0, cell_size_y, 0)
        + ooxyz
    )
    ood = inverse_taper_coord(
        point_grid[cell_x, cell_y, cell_z1] * point_grid_diff_lr_scale
        + offset_tensor(0, 0, cell_size_z)
        + ooxyz
    )

    bob = inverse_taper_coord(
        point_grid[cell_x0, cell_y, cell_z0] * point_grid_diff_lr_scale
        + offset_tensor(-cell_size_x, 0, -cell_size_z)
        + ooxyz
    )
    bod = inverse_taper_coord(
        point_grid[cell_x0, cell_y, cell_z1] * point_grid_diff_lr_scale
        + offset_tensor(-cell_size_x, 0,  cell_size_z)
        + ooxyz
    )
    dob = inverse_taper_coord(
        point_grid[cell_x1, cell_y, cell_z0] * point_grid_diff_lr_scale
        + offset_tensor( cell_size_x, 0, -cell_size_z)
        + ooxyz
    )
    dod = inverse_taper_coord(
        point_grid[cell_x1, cell_y, cell_z1] * point_grid_diff_lr_scale
        + offset_tensor( cell_size_x, 0,  cell_size_z)
        + ooxyz
    )
    boo = inverse_taper_coord(
        point_grid[cell_x0, cell_y, cell_z] * point_grid_diff_lr_scale
        + offset_tensor(-cell_size_x, 0, 0)
        + ooxyz
    )
    doo = inverse_taper_coord(
        point_grid[cell_x1, cell_y, cell_z] * point_grid_diff_lr_scale
        + offset_tensor( cell_size_x, 0, 0)
        + ooxyz
    )

    bbo = inverse_taper_coord(
        point_grid[cell_x0, cell_y0, cell_z] * point_grid_diff_lr_scale
        + offset_tensor(-cell_size_x, -cell_size_y, 0)
        + ooxyz
    )
    bdo = inverse_taper_coord(
        point_grid[cell_x0, cell_y1, cell_z] * point_grid_diff_lr_scale
        + offset_tensor(-cell_size_x,  cell_size_y, 0)
        + ooxyz
    )
    dbo = inverse_taper_coord(
        point_grid[cell_x1, cell_y0, cell_z] * point_grid_diff_lr_scale
        + offset_tensor( cell_size_x, -cell_size_y, 0)
        + ooxyz
    )
    ddo = inverse_taper_coord(
        point_grid[cell_x1, cell_y1, cell_z] * point_grid_diff_lr_scale
        + offset_tensor( cell_size_x,  cell_size_y, 0)
        + ooxyz
    )

    # Ray origin & direction => unsqueeze for broadcasting
    o = ray_origins.unsqueeze(-2)     # [..., 1, 3]
    d = ray_directions.unsqueeze(-2)  # [..., 1, 3]

    # Helper for triangle intersection
    def tri_intersect_and_mask(p1, p2, p3):
        a, b, c, mask_ = get_barycentric(p1, p2, p3, o, d)
        P_ = p1 * a.unsqueeze(-1) + p2 * b.unsqueeze(-1) + p3 * c.unsqueeze(-1)
        # print("mask_", mask_.shape)
        # print("masks.unsqueeze(-1 )", masks.unsqueeze(-1).shape)
        # print("get inside cell mask ", get_inside_cell_mask(P_, ooxyz).shape)
        inside_mask = get_inside_cell_mask(P_, ooxyz) & mask_ & masks
        return P_, inside_mask

    # ------ X faces ------
    P_x_1, P_x_1m = tri_intersect_and_mask(obb, obo, ooo)
    P_x_2, P_x_2m = tri_intersect_and_mask(obb, oob, ooo)
    P_x_3, P_x_3m = tri_intersect_and_mask(odd, odo, ooo)
    P_x_4, P_x_4m = tri_intersect_and_mask(odd, ood, ooo)
    P_x_5, P_x_5m = tri_intersect_and_mask(oob, odo, ooo)
    P_x_6, P_x_6m = tri_intersect_and_mask(oob, odo, odb)
    P_x_7, P_x_7m = tri_intersect_and_mask(obo, ood, ooo)
    P_x_8, P_x_8m = tri_intersect_and_mask(obo, ood, obd)

    # ------ Y faces ------
    P_y_1, P_y_1m = tri_intersect_and_mask(bob, boo, ooo)
    P_y_2, P_y_2m = tri_intersect_and_mask(bob, oob, ooo)
    P_y_3, P_y_3m = tri_intersect_and_mask(dod, doo, ooo)
    P_y_4, P_y_4m = tri_intersect_and_mask(dod, ood, ooo)
    P_y_5, P_y_5m = tri_intersect_and_mask(oob, doo, ooo)
    P_y_6, P_y_6m = tri_intersect_and_mask(oob, doo, dob)
    P_y_7, P_y_7m = tri_intersect_and_mask(boo, ood, ooo)
    P_y_8, P_y_8m = tri_intersect_and_mask(boo, ood, bod)

    # ------ Z faces ------
    P_z_1, P_z_1m = tri_intersect_and_mask(bbo, boo, ooo)
    P_z_2, P_z_2m = tri_intersect_and_mask(bbo, obo, ooo)
    P_z_3, P_z_3m = tri_intersect_and_mask(ddo, doo, ooo)
    P_z_4, P_z_4m = tri_intersect_and_mask(ddo, odo, ooo)
    P_z_5, P_z_5m = tri_intersect_and_mask(obo, doo, ooo)
    P_z_6, P_z_6m = tri_intersect_and_mask(obo, doo, dbo)
    P_z_7, P_z_7m = tri_intersect_and_mask(boo, odo, ooo)
    P_z_8, P_z_8m = tri_intersect_and_mask(boo, odo, bdo)

    # Concatenate masks => shape [..., 24]
    world_masks = torch.cat([
        P_x_1m, P_x_2m, P_x_3m, P_x_4m,
        P_x_5m, P_x_6m, P_x_7m, P_x_8m,
        P_y_1m, P_y_2m, P_y_3m, P_y_4m,
        P_y_5m, P_y_6m, P_y_7m, P_y_8m,
        P_z_1m, P_z_2m, P_z_3m, P_z_4m,
        P_z_5m, P_z_6m, P_z_7m, P_z_8m,
    ], dim=-1)

    # Concatenate positions => shape [..., 24, 3]
    world_positions = torch.cat([
        P_x_1, P_x_2, P_x_3, P_x_4,
        P_x_5, P_x_6, P_x_7, P_x_8,
        P_y_1, P_y_2, P_y_3, P_y_4,
        P_y_5, P_y_6, P_y_7, P_y_8,
        P_z_1, P_z_2, P_z_3, P_z_4,
        P_z_5, P_z_6, P_z_7, P_z_8,
    ], dim=-2)

    # Distance along ray => dot(world_positions, d), shape [..., 24]
    # d => [..., 1, 3], broadcast
    world_tx = torch.sum(world_positions * d, dim=-1)

    # Set invalid intersections to large_val
    large_val = torch.tensor(1000.0, dtype=dtype, device=device)
    world_tx = world_tx * world_masks + large_val * (~world_masks)

    # Sort => shape [..., 24], gather top keep_num
    ind = torch.argsort(world_tx, dim=-1)
    ind = ind[..., :keep_num]

    # Gather distances
    world_tx = torch.gather(world_tx, dim=-1, index=ind)

    # Gather masks
    world_masks = torch.gather(world_masks, dim=-1, index=ind)

    # Gather positions => shape [..., keep_num, 3]
    ind_expanded = ind.unsqueeze(-1).expand(*ind.shape, 3)
    world_positions = torch.gather(world_positions, dim=-2, index=ind_expanded)

    # Convert to "taper" coords & zero out invalid
    taper_positions = get_taper_coord(world_positions)
    taper_positions = taper_positions * world_masks.unsqueeze(-1)

    # ooo_times_mask => shape [..., 3], multiply by masks[..., None]
    # "masks" was shape [...], so unsqueeze => [..., 1]
    ooo_times_mask = ooo_ * masks.unsqueeze(-1)

    # Return .detach()'ed distances
    return taper_positions, world_masks, ooo_times_mask, world_tx.detach()

##############################################################################
# 1) compute_t_forwardfacing
#
#   - We interpret grid_max as a Torch tensor of shape [3], or a single float
#     that can broadcast with `world_masks[..., None]`.
#   - We remove jax.lax.stop_gradient(...) and use .detach().
##############################################################################

def compute_t_forwardfacing(taper_positions, world_masks, grid_max):
    """
    Use 'world_tx' in taper coord to avoid exponentially larger intervals.

    taper_positions: (Tensor) shape [..., N, 3], on device
    world_masks:     (Tensor) shape [..., N], boolean or float in {0,1}
    grid_max:        (Tensor or float) shape [3] or scalar, on device

    Returns:
        world_tx: (Tensor) shape [..., N] (detached from gradient)
    """
    device = taper_positions.device
    dtype = taper_positions.dtype

    # Expand world_masks to shape [..., N, 1] so that it can broadcast with 'grid_max'.
    # (1 - world_masks[..., None]) => picks 'grid_max' where mask=0, or 0 where mask=1
    offset = (1.0 - world_masks.unsqueeze(-1)) * grid_max

    # Subtract the "base" position taper_positions[..., 0:1, :] from each point
    # plus the offset for invalid points
    diff = taper_positions + offset - taper_positions[..., :1, :]

    # L2 distance along last axis => shape [..., N]
    world_tx = diff.square().sum(dim=-1).sqrt()

    return world_tx.detach()


##############################################################################
# 2) sort_and_compute_t_real360
#
#   - Similar logic, but we also sort the resulting distances and reorder
#     (taper_positions, world_masks) accordingly.
#   - Replaces np.argsort with torch.argsort and np.take_along_axis with torch.gather.
##############################################################################

def sort_and_compute_t_real360(taper_positions, world_masks):
    """
    Similar to `compute_t_forwardfacing` but with a fixed offset = 2.0,
    and then sorts the results.

    taper_positions: (Tensor) shape [..., N, 3], on device
    world_masks:     (Tensor) shape [..., N], boolean or float
    Returns:
        taper_positions: re-sorted positions
        world_masks:     re-sorted mask
        world_tx:        sorted distances (detached)
    """
    device = taper_positions.device
    dtype = taper_positions.dtype

    # offset = (1 - mask) * 2.0
    offset = (1.0 - world_masks.unsqueeze(-1)) * 2.0
    diff = taper_positions + offset - taper_positions[..., :1, :]

    # shape [..., N]
    world_tx = diff.square().sum(dim=-1).sqrt()

    # argsort => shape [..., N]
    ind = torch.argsort(world_tx, dim=-1)

    # gather the distances
    # shape => [..., N]
    world_tx_sorted = torch.gather(world_tx, dim=-1, index=ind)

    # gather the masks => shape [..., N]
    world_masks_sorted = torch.gather(world_masks, dim=-1, index=ind)

    # gather the positions => shape [..., N, 3]
    # expand the index => [..., N, 1]
    ind_expanded = ind.unsqueeze(-1).expand(*ind.shape, 3)
    taper_positions_sorted = torch.gather(taper_positions, dim=-2, index=ind_expanded)

    return taper_positions_sorted, world_masks_sorted, world_tx_sorted.detach()


##############################################################################
# 3) compute_box_intersection
#
#   - Replaces NumPy array ops with Torch ops.
#   - Expects global or passed-in variables for:
#       point_grid_size, scene_grid_zcc, etc.
#     or you can add them as function parameters.
##############################################################################

def compute_box_intersection(rays, point_grid_size, scene_grid_zcc):
    """
    Compute box intersections (no top & bottom).

    rays:            (ray_origins, ray_directions), each shape [..., 3], on device
    point_grid_size: (int) e.g. 128
    scene_grid_zcc:  (float) exponent scaling factor
    Returns:
        taper_positions: (Tensor) shape [..., K, 3]
        world_masks:     (Tensor) shape [..., K], boolean
    """
    ray_origins, ray_directions = rays
    device = ray_origins.device
    dtype = ray_origins.dtype

    epsilon = 1e-10

    ox = ray_origins[..., 0:1]
    oy = ray_origins[..., 1:2]
    oz = ray_origins[..., 2:3]

    dx = ray_directions[..., 0:1]
    dy = ray_directions[..., 1:2]
    dz = ray_directions[..., 2:3]

    # Avoid zero division by shifting dx, dy if they're small
    dx_mask = (dx.abs() < epsilon)
    dy_mask = (dy.abs() < epsilon)

    dx = dx + dx_mask.to(dtype)
    dy = dy + dy_mask.to(dtype)

    # shape => [ (point_grid_size//2) + 1 ]
    half_size = point_grid_size // 2
    layers_ = (torch.arange(half_size + 1, device=device, dtype=dtype) / half_size)
    # layers = (exp(layers_ * scene_grid_zcc) + (scene_grid_zcc-1)) / scene_grid_zcc
    layers = (torch.exp(layers_ * scene_grid_zcc) + (scene_grid_zcc - 1)) / scene_grid_zcc

    # Broadcast 'layers' to match rays
    # rays => shape [..., 3], so we want layers => [..., (half_size+1)]
    # This means we expand it across the batch dims of ray_origins
    # e.g. if ray_origins is shape [B, 3], we want shape [B, (half_size+1)]
    batch_shape = ray_origins.shape[:-1]
    for _ in batch_shape:
        layers = layers.unsqueeze(0)
    layers = layers.expand(*batch_shape, half_size + 1)

    # compute tx_p, tx_n, ty_p, ty_n
    tx_p = (layers - ox) / dx
    tx_n = (-layers - ox) / dx
    ty_p = (layers - oy) / dy
    ty_n = (-layers - oy) / dy

    # tx => pick whichever is > => torch.where
    tx = torch.where(tx_p > tx_n, tx_p, tx_n)
    ty = torch.where(ty_p > ty_n, ty_p, ty_n)

    # pick which to use by comparing |tx_py| vs. |ty_px|
    tx_py = oy + dy * tx  # shape [..., half_size+1]
    ty_px = ox + dx * ty

    # shape [..., half_size+1]
    # pick "tx" if abs(tx_py) < abs(ty_px), else "ty"
    t = torch.where(tx_py.abs() < ty_px.abs(), tx, ty)

    # shape [..., half_size+1]
    t_pz = oz + dz * t
    # world_masks => check |t_pz| < layers
    world_masks = (t_pz.abs() < layers)

    # shape => [..., half_size+1, 3]
    world_positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t.unsqueeze(-1)

    # taper_scales = (layers_+1)/layers * world_masks
    # But note that "layers_" is shape [half_size+1] (not broadcasted).
    # We need to replicate the same shape as "layers".
    # We'll replicate the same approach for broadcasting.

    # Rebuild layers_ with the same shape as 'layers'
    # We'll reconstruct layers_ the same way:
    layers_base = (torch.arange(half_size + 1, device=device, dtype=dtype) / half_size)
    for _ in batch_shape:
        layers_base = layers_base.unsqueeze(0)
    layers_base = layers_base.expand(*batch_shape, half_size + 1)

    # shape => [..., half_size+1]
    taper_scales = ((layers_base + 1.0) / layers) * world_masks

    # shape => [..., half_size+1, 3]
    taper_positions = world_positions * taper_scales.unsqueeze(-1)

    return taper_positions, world_masks



#%% --------------------------------------------------------------------------------
# ## MLP setup
#%%
num_bottleneck_features = 8

class RadianceField(nn.Module):

    def __init__(self, output_dim, config):
        super().__init__()
        self.model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=output_dim, 
                                                   encoding_config=config["encoding"], network_config=config["network"]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["optimizer"]["learning_rate"])
    def forward(self, positions):
        out = self.model(positions)
        return out
    
class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.model = tcnn.Network(input_dim, output_dim, config["network"]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["optimizer"]["learning_rate"])

    def forward(self, x):
        out = self.model(x)
        return out

density_model = RadianceField(1, config)
feature_model = RadianceField(num_bottleneck_features, config)
color_model = MLP(input_dim = 11, output_dim=3, config=config_mlp)
sigmoid = nn.Sigmoid()
# point_grid = None
# acc_grid = None

import torch

def compute_volumetric_rendering_weights_with_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Given alpha (shape [..., N]), compute the volume rendering weights using a
    transparency formulation. This function is fully in Torch and preserves
    device placement.

    Args:
        alpha: (Tensor) The per-sample opacity, shape [..., N].

    Returns:
        weights: (Tensor) The computed rendering weights, same shape [..., N].
    """
    # 1. density_exp = 1 - alpha
    density_exp = 1.0 - alpha
    
    # 2. density_exp_shifted: prepend a column of 1's (for transmittance at the first sample)
    density_exp_shifted = torch.cat([
        torch.ones_like(density_exp[..., :1]),  # shape [..., 1]
        density_exp[..., :-1]                   # shape [..., N-1]
    ], dim=-1)  # => shape [..., N]

    # 3. trans = cumprod of density_exp_shifted along the last dimension
    trans = torch.cumprod(density_exp_shifted, dim=-1)

    # 4. weights = alpha * trans
    weights = alpha * trans

    return weights

def render_rays(rays, point_grid, acc_grid, keep_num, threshold):

    grid_indices, grid_masks = gridcell_from_rays(rays, acc_grid, keep_num, threshold)

    pts, grid_masks, points, fake_t = compute_undc_intersection(point_grid, 
                                                            grid_indices, grid_masks, rays, keep_num)
    # print("rendering rays!")
    B, R, S, C = pts.shape
    pts_flat = pts.view(-1, C)
    # print("points shape ", pts_flat.shape)
    mlp_alpha_flat = density_model(pts_flat)
    mlp_alpha_flat = sigmoid(mlp_alpha_flat[..., 0] - 8)
    mlp_alpha = mlp_alpha_flat.view(B, R, S)
    # print("grid_masks shape ", grid_masks.shape)
    # print(" mlp alpha ", mlp_alpha.shape)
    mlp_alpha = mlp_alpha * grid_masks

    weights = compute_volumetric_rendering_weights_with_alpha(mlp_alpha)
    acc = torch.sum(weights, axis=-1)

    mlp_alpha_b = mlp_alpha + (torch.clip((mlp_alpha > 0.5).float(), 
                                          0.00001,0.99999) - mlp_alpha).detach()

    weights_b = compute_volumetric_rendering_weights_with_alpha(mlp_alpha_b)
    acc_b = torch.sum(weights_b, axis=-1)

    dirs = normalize(rays[1])
    dirs = dirs.unsqueeze(2).expand(-1, -1, pts.shape[2], -1)
    # print("dirs shape", dirs.shape)
    # print("rays 1 ", rays[1].shape)

    mlp_features_flat = sigmoid(feature_model(pts_flat))
    mlp_features = mlp_features_flat.view(B, R, S, -1)
    features_dir_enc = torch.concatenate([mlp_features, dirs], axis=-1)
    # print("Features dic ecn ", features_dir_enc.shape)
    features_dir_enc_flat = features_dir_enc.view(-1, 11)
    colors_flat = sigmoid(color_model(features_dir_enc_flat))

    colors = colors_flat.view(B, R, S, -1)
    # print("colors shape ", colors.shape)

    rgb = torch.sum(weights[..., None] * colors, axis=-2)
    rgb_b = torch.sum(weights_b[..., None] * colors, axis=-2)

    if white_bkgd:
        rgb=rgb + (1. - acc[..., None])
        rgb_b = rgb_b + (1. - acc_b[..., None])
    
    acc_grid_masks = get_acc_grid_masks(pts, acc_grid)
    acc_grid_masks = acc_grid_masks * grid_masks

    return rgb, acc, rgb_b, acc_b, mlp_alpha, weights, points, fake_t, acc_grid_masks


#%% --------------------------------------------------------------------------------
# ## Set up pmap'd rendering for test time evaluation.
#%%
test_batch_size = 2
test_keep_num = point_grid_size*3//16
print("test_keep_num ", test_keep_num)
test_threshold = 0.1
test_wbgcolor = 0.0

def render_test_p(rays, point_grid, acc_grid):
    return render_rays(rays, point_grid, acc_grid, test_keep_num, test_threshold)

def render_test(rays, point_grid, acc_grid):
    sh = rays[0].shape
    rays = [x.reshape((1, -1) + sh[1:]) for x in rays]
    out = render_test_p(rays, point_grid, acc_grid)
    # out = [np.reshape(np.array(x.detach().cpu()),sh[:-1]+(-1,)) for x in out]
    return out

def render_loop(rays, point_grid, acc_grid, chunk):
    sh = list(rays[0].shape[:-1])
    rays = [x.reshape([-1,3]) for x in rays]
    l = rays[0].shape[0]
    print("rays shape", rays[0].shape)
    n = 1
    p = ((l-1) // n + 1) * n - l
    rays = [F.pad(x, (0, 0, 0, p)) for x in rays]
    outs = [render_test([x[i:i+chunk] for x in rays], point_grid, acc_grid)
            for i in range(0, rays[0].shape[0], chunk)]
    outs = [torch.reshape(
        torch.concatenate([z[i] for z in outs])[:l], sh+ [-1]) for i in range(4)]
    
    return outs

def chunked_train_step(
    traindata,
    lr,
    wdistortion,
    batch_size=2048,      # total rays per train step
    chunk_size=256,       # how many rays to render at once
):
    """
    1) Sample 'batch_size' rays via random_ray_batch.
    2) Split them into chunks of 'chunk_size' each.
    3) For each chunk, call 'render_rays' & compute your losses,
       then do partial backward (gradient accumulation).
    4) At the end, do a single optimizer step.
    """
    # =======================
    #  Gather a batch of rays
    # =======================
    # e.g. batch_size=2048 -> 2k rays in total
    rays, pixels = random_ray_batch(batch_size, traindata)  # your existing function

    # rays[0] => shape [batch_size, 3] (origins)
    # rays[1] => shape [batch_size, 3] (dirs)
    # pixels => shape [batch_size, 3]

    # =========================
    #  Adjust the learning rate
    # =========================
    for param_group in density_model.optimizer.param_groups:
        param_group["lr"] = lr
    for param_group in feature_model.optimizer.param_groups:
        param_group["lr"] = lr
    for param_group in color_model.optimizer.param_groups:
        param_group["lr"] = lr

    # ================================
    #  Zero the grads before we start
    # ================================
    density_model.optimizer.zero_grad()
    feature_model.optimizer.zero_grad()
    color_model.optimizer.zero_grad()

    # We'll track total losses over all chunks
    total_loss_acc = 0.0
    color_loss_acc = 0.0

    # We'll also do a single backward pass per chunk. 
    # Then after all chunks, we do "optimizer.step()".
    # That means we want to scale each chunk's loss 
    # so the overall gradient is consistent with a single big batch.
    # scale_factor = chunk_size / batch_size for each chunk.

    n_rays_total = rays[0].shape[0]  # should equal 'batch_size'
    num_chunks = (n_rays_total + chunk_size - 1) // chunk_size

    # ============================================
    #  Loop over all rays in chunked mini-batches
    # ============================================
    for chunk_i in range(num_chunks):
        start = chunk_i * chunk_size
        end   = min(start + chunk_size, n_rays_total)
        this_count = end - start  # how many rays in this chunk

        # Slice out the chunk for origins, dirs, and pixels
        sub_origins = rays[0][start:end]  # shape [this_count, 3]
        sub_dirs    = rays[1][start:end]
        sub_pixels  = pixels[start:end]   # shape [this_count, 3]

        # We pass them to 'render_rays' with shape [1, this_count, 3],
        # because your code expects [B, R, 3], so we do B=1:
        rays_chunk = (
            sub_origins.unsqueeze(0),  # => [1, this_count, 3]
            sub_dirs.unsqueeze(0),     # => [1, this_count, 3]
        )

        # ==============
        #  Forward pass
        # ==============
        with autocast():
            # This calls your existing "render_rays" function
            # which returns (rgb, acc, rgb_b, acc_b, mlp_alpha, weights, points, fake_t, acc_grid_masks)
            rgb_est, _, rgb_est_b, _, mlp_alpha, weights, points, fake_t, acc_grid_masks = render_rays(
                rays_chunk, point_grid, acc_grid, test_keep_num, test_threshold
            )

            # rgb_est => shape [1, this_count, 3]
            # We'll squeeze out the batch dim:
            rgb_est_squeezed = rgb_est[0]  # => [this_count, 3]

            # ===============
            #  Compute losses
            # ===============
            # 1) color loss
            loss_color_l2 = torch.mean((rgb_est_squeezed - sub_pixels)**2)

            # 2) coverage/acc loss
            loss_acc = torch.mean(torch.clamp(weights.detach() - acc_grid_masks, min=0))
            loss_acc += torch.mean(torch.abs(acc_grid)) * 1e-5
            loss_acc += compute_TV(acc_grid) * 1e-5

            # 3) distortion
            loss_distortion = torch.mean(lossfun_distortion(fake_t, weights)) * wdistortion

            # 4) point_loss
            point_loss = torch.abs(points)
            point_loss_out = point_loss * 1000.0
            point_loss_in  = point_loss * 0.01
            point_mask = point_loss < (grid_max - grid_min) / point_grid_size / 2
            point_loss = torch.mean(torch.where(point_mask, point_loss_in, point_loss_out))

            # Combine them
            total_loss_chunk = (loss_color_l2 + loss_distortion + loss_acc + point_loss)

        # ===========================
        #  Scale + backward per chunk
        # ===========================
        # We'll scale this chunk's loss so the final gradient matches 
        # a single large batch. That means multiplying by (this_count / batch_size).
        scale_factor = float(this_count) / float(n_rays_total)
        scaled_loss  = total_loss_chunk * scale_factor

        scaler.scale(scaled_loss).backward()

        # Keep track of the *unscaled* loss values for logging
        total_loss_acc  += total_loss_chunk.item() * this_count
        color_loss_acc  += loss_color_l2.item()      * this_count

        # After this chunk finishes, the big intermediate Tensors 
        # (pts, weights, acc_grid_masks, etc.) go out of scope 
        # and can be freed by PyTorch.

    # end of chunk loop

    # ==========================
    #  Single optimizer step now
    # ==========================
    scaler.step(density_model.optimizer)
    scaler.step(feature_model.optimizer)
    scaler.step(color_model.optimizer)
    scaler.update()

    # We can zero_grad here or in the next iteration
    density_model.optimizer.zero_grad()
    feature_model.optimizer.zero_grad()
    color_model.optimizer.zero_grad()

    # Compute final average losses for logging
    final_total_loss  = total_loss_acc  / n_rays_total
    final_color_loss  = color_loss_acc  / n_rays_total

    return final_total_loss, final_color_loss

if scene_type=="synthetic":
  selected_test_index = 97
  preview_image_height = 800

print("testing rays")
rays = camera_ray_batch(
    data['test']['c2w'][selected_test_index], data['test']['hwf'])
gt = data['test']['images'][selected_test_index]
out = render_loop(rays, point_grid, acc_grid, 
                   test_batch_size)
rgb = out[0]
acc = out[1]
rgb_b = out[2]
acc_b = out[3]
write_floatpoint_image(samples_dir+"/s1_"+str(0)+"_rgb.png",rgb)
write_floatpoint_image(samples_dir+"/s1_"+str(0)+"_rgb_binarized.png",rgb_b)
write_floatpoint_image(samples_dir+"/s1_"+str(0)+"_gt.png",gt)
write_floatpoint_image(samples_dir+"/s1_"+str(0)+"_acc.png",acc)
write_floatpoint_image(samples_dir+"/s1_"+str(0)+"_acc_binarized.png",acc_b)

#%% --------------------------------------------------------------------------------
# ## Training loop
#%%

##############################################################################
# Training loop with AMP
##############################################################################
def lossfun_distortion(x, w):
    dux = torch.abs(x[..., :, None] - x[..., None, :])
    losses_cross = torch.sum(w * torch.sum(w[..., None, :] * dux, axis=-1), axis=-1)
    losses_self = torch.sum(
        (w[..., 1:]**2 + w[..., :-1]**2)*(x[...,1:] - x[..., :-1]),
        axis=-1
    ) / 6
    return losses_cross + losses_self

def compute_TV(acc_grid):
    dx = acc_grid[:-1,:,:] - acc_grid[1:,:,:]
    dy = acc_grid[:,:-1,:] - acc_grid[:,1:,:]
    dz = acc_grid[:,:,:-1] - acc_grid[:,:,1:]
    TV = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)
    return TV

# -------------------- CHANGED: Create a GradScaler for AMP --------------------
scaler = GradScaler()

def train_step(traindata, lr, wdistortion):
    # Example: pick a batch
    rays, pixels = random_ray_batch(16, traindata)
    # print("size of rays ", rays.shape)
    # print("size of pixels ", pixels.shape)

    # We'll define the forward pass in an inner function
    def loss_fn():
        rgb_est, _, rgb_est_b, _, mlp_alpha, weights, points, fake_t, acc_grid_masks = render_rays(
            rays, point_grid, acc_grid, test_keep_num, test_threshold
        )

        loss_color_l2 = torch.mean((rgb_est - pixels)**2)
        loss_acc = torch.mean(torch.clamp(weights.detach() - acc_grid_masks, min=0))
        loss_acc += torch.mean(torch.abs(acc_grid)) * 1e-5
        loss_acc += compute_TV(acc_grid) * 1e-5

        loss_distortion = torch.mean(lossfun_distortion(fake_t, weights)) * wdistortion

        point_loss = torch.abs(points)
        point_loss_out = point_loss * 1000.0
        point_loss_in = point_loss * 0.01
        point_mask = point_loss < (grid_max - grid_min) / point_grid_size/2
        point_loss = torch.mean(
            torch.where(point_mask, point_loss_in, point_loss_out)
        )
        return loss_color_l2 + loss_distortion + loss_acc + point_loss, loss_color_l2

    # -------------------- CHANGED: use autocast + GradScaler --------------------
    with autocast():
        total_loss, color_loss_l2 = loss_fn()

    density_model.optimizer.zero_grad()
    feature_model.optimizer.zero_grad()
    color_model.optimizer.zero_grad()

    # Instead of total_loss.backward(), we do scaled backward:
    scaler.scale(total_loss).backward()

    # Step each optimizer
    scaler.step(density_model.optimizer)
    scaler.step(feature_model.optimizer)
    scaler.step(color_model.optimizer)
    scaler.update()

    return total_loss.item(), color_loss_l2.item()

##############################################################################
# Main training loop
##############################################################################
step_init = 0
training_iters = 200000
train_iters_cont = 300000

traindata = data['train']
print("training")

for i in tqdm(range(step_init, training_iters + 1)):
    lr = lr_fn(i, train_iters_cont, 1e-3, 1e-5)
    wdistortion = 0.0  # etc.

    # Adjust the learning rate
    # for param_group in color_model.optimizer.param_groups:
    #     param_group["lr"] = lr
    # for param_group in density_model.optimizer.param_groups:
    #     param_group["lr"] = lr
    # for param_group in feature_model.optimizer.param_groups:
    #     param_group["lr"] = lr

    # total_loss, color_loss = train_step(traindata, lr, wdistortion)

    total_loss, color_loss = chunked_train_step(
            traindata,
            lr,
            wdistortion,
            batch_size=16,   # total rays
            chunk_size=4,    # chunk at a time
        )


    print(i, total_loss, color_loss)
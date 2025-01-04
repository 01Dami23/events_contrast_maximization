import argparse
import time
import numpy as np
import scipy
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
import torch
from event_utils import *
from objectives import *
from warps import *

import os
import cv2
from tqdm import tqdm
from scipy.ndimage import uniform_filter


def filter_events(xs, ys, ts, ps, window_size=5, C=1.1):
    """
    Filter events using adaptive thresholding based on local event density.
    
    Args:
        xs, ys, ts, ps: Event coordinates, timestamps, and polarities
        window_size: Size of local window for computing mean event density
        C: Scaling factor for threshold calculation
    """
    # Create 2D histogram of event counts
    coords = np.column_stack((xs, ys))
    unique_coords, counts = np.unique(coords, axis=0, return_counts=True)
    
    # Create density map
    density_map = np.zeros((int(np.max(ys)) + 1, int(np.max(xs)) + 1))
    density_map[unique_coords[:, 1].astype(int), unique_coords[:, 0].astype(int)] = counts
    
    # Calculate local mean using moving average
    local_mean = uniform_filter(density_map, size=window_size)
    
    # Calculate adaptive threshold
    threshold_map = local_mean * C
    
    # Get indices of events to keep
    keep_indices = []
    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        if density_map[y, x] >= threshold_map[y, x]:
            keep_indices.append(i)
    
    # Filter arrays
    return xs[keep_indices], ys[keep_indices], ts[keep_indices], ps[keep_indices]


def optimize_contrast(xs, ys, ts, ps, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None,
        numeric_grads=False, blur_sigma=None, img_size=(180, 240)):
    """
    Optimize contrast for a set of events
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp_function (function) The function with which to warp the events
    objective (objective class object) The objective to optimize
    optimizer (function) The optimizer to use
    x0 (np array) The initial guess for optimization
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor
    blur_sigma (float) Size of the blurring kernel. Blurring the images of warped events can
        have a large impact on the convergence of the optimization.

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    args = (xs, ys, ts, ps, warp_function, img_size, blur_sigma)
    
    
    
    
    
    
    
    
    
    # BUG IN THE CODE REINITIALIZING EVERY TIME X0 TO 0,0 EVEN IF PASSED AS ARGUMENT
    #x0 = np.array([0,0])

    if x0 is None:
        #x0 = np.random.rand(2)
        x0 = np.array([0,0])







    if x0 is None:
        x0 = np.zeros(warp_function.dims)




    print("x0: ", x0)





    
    if numeric_grads:
        argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
    else:
        argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
    return argmax

def optimize(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events. Uses optimize_contrast() for the optimiziation, but allows
    blurring schedules for successive optimization iterations.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=blur, img_size=img_size)
    return argmax_an

def optimize_r2(xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
    """
    Optimize contrast for a set of events, finishing with SoE loss.
    Parameters:
    xs (numpy float array) The x components of the events
    ys (numpy float array) The y components of the events
    ts (numpy float array) The timestamps of the events. Timestamps should be ts-t[0] to avoid precision issues.
    ps (numpy float array) The polarities of the events
    warp (function) The function with which to warp the events
    obj (objective class object) The objective to optimize
    numeric_grads (bool) If true, use numeric derivatives, otherwise use analytic drivatives if available.
        Numeric grads tend to be more stable as they are a little less prone to noise and don't require as much
        tuning on the blurring parameter. However, they do make optimization slower.
    img_size (tuple) The size of the event camera sensor

    Returns:
        The max arguments for the warp parameters wrt the objective
    """
    soe_obj = soe_objective()
    numeric_grads = numeric_grads if obj.has_derivative else True
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None)
    argmax_an = optimize_contrast(xs, ys, ts, ps, warp, soe_obj, x0=argmax_an, numeric_grads=numeric_grads, blur_sigma=1.0)
    return argmax_an

if __name__ == "__main__":
    """
    Quick demo of various objectives.
    Args:
        path Path to h5 file with event data
        gt Ground truth optic flow for event slice
        img_size The size of the event camera sensor
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file_path", default=None)
    parser.add_argument("--rec_scene_path", default=None)
    parser.add_argument("--output_scene_path", default=None)
    #parser.add_argument("--gt", nargs='+', type=float, default=(0,0))
    #parser.add_argument("--img_size", nargs='+', type=float, default=(180,240))

    args = parser.parse_args()

    xs, ys, ts, ps = read_h5_event_components(args.h5_file_path)
    img_size = h5py.File(args.h5_file_path, 'r').attrs['sensor_resolution']

    # normalize timestamps according to MULTIFLOW TIMESTAMP VALUES
    ts = ts - ts[0]
    ts = ts/1000000.0

    print("sensor resolution: ", img_size)

    img_tstamps_path = os.path.join(args.rec_scene_path, 'timestamps.txt')
    img_names = []
    img_tstamps = []
    with open(img_tstamps_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.split()
            img_names.append(parts[0])
            img_tstamps.append(float(parts[1])/1000000.0)

            img_tstamps[i] = img_tstamps[i] - img_tstamps[0]

    obj = r1_objective()
    warp = linvel_warp()

    #for i in range(1, len(img_names)):
    for i in tqdm(range(1, len(img_names))):
        start_ts = img_tstamps[i-1]
        end_ts = img_tstamps[i]
        rec_img = cv2.imread(os.path.join(args.rec_scene_path, img_names[i]))

        # normalize images
        rec_img = rec_img/255.0

        img_xs = xs[(ts >= start_ts) & (ts < end_ts)]
        img_ys = ys[(ts >= start_ts) & (ts < end_ts)]
        img_ts = ts[(ts >= start_ts) & (ts < end_ts)]
        img_ps = ps[(ts >= start_ts) & (ts < end_ts)]

        # Use R2 optimization (r_sos + r_sosa + r_soe) of R1 objective is (r_sos + r_sosa)
        argmax = optimize_r2(img_xs, img_ys, img_ts, img_ps, warp, obj, numeric_grads=True, img_size=img_size)

        loss = obj.evaluate_function(argmax, img_xs, img_ys, img_ts, img_ps, warp, img_size=img_size)
        # print("{}:({})={}".format(obj.name, argmax, loss))

        print("parameters: ", argmax)

        # warp events
        img_xs_warped, img_ys_warped, _, _ = warp.warp(img_xs, img_ys, img_ts, img_ps, end_ts, argmax)
        img_xs_warped = img_xs_warped.astype(np.int64)
        img_ys_warped = img_ys_warped.astype(np.int64)

        # delete events out of bounds
        img_xs = img_xs_warped[(img_xs_warped >= 0) & (img_xs_warped < img_size[1]) & (img_ys_warped >= 0) & (img_ys_warped < img_size[0])]
        img_ys = img_ys_warped[(img_ys_warped >= 0) & (img_ys_warped < img_size[0]) & (img_xs_warped >= 0) & (img_xs_warped < img_size[1])]
        img_ts = img_ts[(img_xs_warped >= 0) & (img_xs_warped < img_size[1]) & (img_ys_warped >= 0) & (img_ys_warped < img_size[0])]
        img_ps = img_ps[(img_xs_warped >= 0) & (img_xs_warped < img_size[1]) & (img_ys_warped >= 0) & (img_ys_warped < img_size[0])]

        # filter events with threshold based on number of events in each pixel
        print("number of events before filtering: ", len(img_xs))
        img_xs, img_ys, img_ts, img_ps = filter_events(img_xs, img_ys, img_ts, img_ps)
        print("number of events after filtering: ", len(img_xs))

        # get voxel grid from events
        B = 1 # ONLY USE 1 BIN DIRECTLY OR USE 5 BINS AND THEN AVERAGE ON THEM?
        voxel_grid = events_to_voxel(img_xs, img_ys, img_ts, img_ps, B, sensor_size=img_size, temporal_bilinear=True)

        # normalize voxel grid
        voxel_grid = (voxel_grid - np.min(voxel_grid))/(np.max(voxel_grid) - np.min(voxel_grid))
        voxel_grid = voxel_grid.transpose(1, 2, 0)
        voxel_grid = voxel_grid.repeat(3, axis=2)

        # add voxel grid to normalized image
        rec_img = rec_img + voxel_grid

        # normalize image between 0 and 1
        rec_img = (rec_img - np.min(rec_img))/(np.max(rec_img) - np.min(rec_img))

        # save image
        cv2.imwrite(os.path.join(args.output_scene_path, img_names[i]), rec_img*255.0)

        # show events
        event_image = events_to_image(img_xs, img_ys, img_ps, img_size)
        event_image = (event_image - np.min(event_image))/(np.max(event_image) - np.min(event_image))
        cv2.imwrite(os.path.join(args.output_scene_path, img_names[i].replace('.png', '_events.png')), event_image*255.0)

import argparse
import os
import numpy as np
from scipy import optimize
import cv2
from typing import Dict, Tuple, List
from tqdm import tqdm
from event_utils import *
from objectives import *
from warps import *

class EventRepresentation:
    """
    Handles event stream voxelization
    """
    def __init__(self, num_bins: int = 5):
        """
        Initialize event representation
        Parameters:
            num_bins (int): Number of bins
        """
        self.num_bins = num_bins
        
    def create_voxel_grid(self, events: Dict[str, np.ndarray], img_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert events to voxel grid using time bilinear interpolation
        Parameters:
            events: Dictionary containing event data with keys 'x', 'y', 't', 'p'
            img_size: Tuple of (height, width) for the event camera sensor
        Returns:
            Voxel grid of shape (num_bins, height, width)
        """

        voxel_grid = np.zeros((self.num_bins, img_size[0], img_size[1]))
        
        t0 = events['t'].min()
        tref = events['t'].max()
        delta_t = tref - t0
        
        t_norm = (events['t'] - t0) / delta_t * (self.num_bins - 1)
        
        for i in range(len(events['x'])):
            x = int(round(events['x'][i]))
            y = int(round(events['y'][i]))
            
            if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
                t = t_norm[i]
                
                for j in range(self.num_bins):
                    weight = max(0, 1 - abs(j - t))
                    if weight > 0:
                        voxel_grid[j, y, x] += events['p'][i] * weight
        
        return voxel_grid


class EventDenoiser:
    def __init__(self, img_size=(180, 240)):
        """
        Initialize the event denoiser
        Parameters:
            img_size (tuple): The size of the event camera sensor (height, width)
        """
        self.img_size = img_size
        self.event_representation = EventRepresentation()
        
    def warp_events(self, events, theta, t_ref=None):
        """
        Warp events according to motion trajectory theta
        Parameters:
            events: dict containing x, y, t, p arrays
            theta: motion trajectory (vx, vy)
            t_ref: reference time (if None, uses last timestamp)
        Returns:
            warped events as dict with x, y, p arrays
            mask of valid events after warping
        """
        if t_ref is None:
            t_ref = events['t'][-1]
            
        dt = events['t'] - t_ref
        x_warped = events['x'] - dt * theta[0]
        y_warped = events['y'] - dt * theta[1]
        
        # Subtract 0.5 to avoid rounding to out of bounds pixels
        valid = (x_warped >= 0) & (x_warped < self.img_size[1]-0.5) & \
                (y_warped >= 0) & (y_warped < self.img_size[0]-0.5)
                
        return {
            'x': x_warped[valid],
            'y': y_warped[valid],
            'p': events['p'][valid]
        }, valid
    
    def create_event_image(self, warped_events):
        """
        Create image from warped events
        Parameters:
            warped_events: dict containing warped x, y, p arrays
        Returns:
            2D numpy array of event counts
        """
        img = np.zeros(self.img_size)
        x = np.round(warped_events['x']).astype(int)
        y = np.round(warped_events['y']).astype(int)
        
        for i in range(len(x)):
            img[y[i], x[i]] += warped_events['p'][i]
            
        return img
    
    def compute_variance(self, img):
        """
        Compute variance of event image
        """
        return np.var(img)
    
    def objective_function(self, theta, events):
        """
        Objective function to maximize contrast
        """
        warped, _ = self.warp_events(events, theta)
        img = self.create_event_image(warped)
        return -self.compute_variance(img)
    
    def optimize_contrast(self, xs, ys, ts, ps, warp_function, objective, optimizer=optimize.fmin_bfgs, x0=None,
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
        
        if x0 is None:
            #x0 = np.random.rand(2)
            x0 = np.array([0,0])

        if numeric_grads:
            argmax = optimizer(objective.evaluate_function, x0, args=args, epsilon=1, disp=False)
        else:
            argmax = optimizer(objective.evaluate_function, x0, fprime=objective.evaluate_gradient, args=args, disp=False)
        return argmax
    
    def optimize_r2(self, xs, ys, ts, ps, warp, obj, numeric_grads=True, img_size=(180, 240)):
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
        argmax_an = self.optimize_contrast(xs, ys, ts, ps, warp, obj, numeric_grads=numeric_grads, blur_sigma=None)
        argmax_an = self.optimize_contrast(xs, ys, ts, ps, warp, soe_obj, x0=argmax_an, numeric_grads=numeric_grads, blur_sigma=1.0)
        return argmax_an
    
    def denoise_events(self, events, threshold_factor=1.0, optim_opt=None):
        """
        Denoise events using contrast maximization
        Parameters:
            events: dict containing x, y, t, p arrays
            threshold_factor: factor for adaptive threshold
            optim_opt: optimization strategy to employ for contrast maximization
        Returns:
            denoised events as dict and voxel grid representation
        """

        theta_init = np.array([0.0, 0.0])
        #theta_init = np.random.randn(2)

        if optim_opt is None or optim_opt == 'Variance':
            result = optimize.minimize(
                self.objective_function,
                theta_init,
                args=(events,),
                method='Nelder-Mead'
            )

            print("Warping parameters: ", result.x)
            # Warp events with optimal motion
            warped, mask = self.warp_events(events, result.x)

        elif optim_opt == 'r2':
            obj = r1_objective()
            r2_warp = linvel_warp()
            result = self.optimize_r2(events['x'], events['y'], events['t'], events['p'], r2_warp, obj, numeric_grads=True, img_size=self.img_size)    

            print("Warping parameters: ", result)
            # Warp events with optimal motion
            warped, mask = self.warp_events(events, result)

        else:
            raise NotImplementedError(f"Optimization strategy {optim_opt} not implemented")
        
        # Filter out events that were delete when warping
        events = {
            'x': events['x'][mask],
            'y': events['y'][mask],
            't': events['t'][mask],
            'p': events['p'][mask]
        }
            
            
        img = self.create_event_image(warped)
        
        # Adaptive thresholding
        mean_events = np.mean(np.abs(img[img != 0]))
        threshold = mean_events * threshold_factor
        
        event_mask = np.abs(img) >= threshold
        valid = event_mask[np.round(warped['y']).astype(int), 
                         np.round(warped['x']).astype(int)]
        
        denoised_events = {
            'x': events['x'][valid],
            'y': events['y'][valid],
            't': events['t'][valid],
            'p': events['p'][valid]
        }
        
        # Create voxel grid representation of denoised events
        voxel_grid = self.event_representation.create_voxel_grid(denoised_events, self.img_size)
        
        return denoised_events, voxel_grid
    
    def restore_image(self, rec_image: np.ndarray, voxel_grid: np.ndarray) -> np.ndarray:
        """
        Add denoised events to reconstructed image to enhance edge information
        Parameters:
            rec_image: Normalized reconstruction image
            voxel_grid: Denoised event voxel grid
        Returns:
            Enhanced image with enhanced edge information
        """
        # Normalize image between 0 and 1 if necessary
        rec_image = (rec_image - np.min(rec_image)) / (np.max(rec_image) - np.min(rec_image))

        # Average voxel grid across bins dimension
        edge_information = np.mean(voxel_grid, axis=0)
        edge_information = edge_information / np.max(np.abs(edge_information))
        
        # Add edge information to reconstructed image
        enhanced_image = rec_image + edge_information
        enhanced_image = np.clip(enhanced_image, 0, 1)
        
        return enhanced_image
    

def get_scene_data(scene_path: str):
    """
    Read scene data from txt file
    Parameters:
        scene_path: Path to txt file containing scene data
    Returns:
        img_names: filenames of reconstructed images in the scene
        img_tstamps: timestamps of reconstructed images in the scene
    """

    file_path = os.path.join(scene_path, 'timestamps.txt')
    img_names = []
    img_tstamps = []

    with open(file_path, 'r') as f:

        for i, line in enumerate(f):

            parts = line.split()
            img_names.append(parts[0])
            img_tstamps.append(float(parts[1])/1000000.0)

            img_tstamps[i] = img_tstamps[i] - img_tstamps[0]
    
    return img_names, img_tstamps


def demo(args):
    """
    Demonstration of event denoising qith contrast maximization
    Parameters:
        args: Command line arguments
    Returns:
        denoised_events: Denoised events as dict
        voxel_grid: Voxel grid representation of denoised events
        enhanced_image: Enhanced image with edge information
    """
    # Create sample events
    xs, ys, ts, ps = read_h5_event_components(args.h5_file_path)

    img_size = h5py.File(args.h5_file_path, 'r').attrs['sensor_resolution']

    # Normalize timestamps according to MULTIFLOW TIMESTAMP VALUES
    ts = ts - ts[0]
    ts = ts/1000000.0

    # Process each image in the scene
    img_names, img_tstamps = get_scene_data(args.rec_scene_path)

    for i in tqdm(range(1, len(img_names))):
        img_path = os.path.join(args.rec_scene_path, img_names[i])
        rec_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Normalize image between 0 and 1 if necessary
        rec_image = (rec_image - np.min(rec_image)) / (np.max(rec_image) - np.min(rec_image))

        # Denoise events
        t0 = img_tstamps[i-1]
        tref = img_tstamps[i]
        events = {
            'x': xs[(ts >= t0) & (ts < tref)],
            'y': ys[(ts >= t0) & (ts < tref)],
            't': ts[(ts >= t0) & (ts < tref)],
            'p': ps[(ts >= t0) & (ts < tref)]
        }

        denoiser = EventDenoiser(img_size=img_size)
        
        if args.optim_opt is None:
            denoised_events, voxel_grid = denoiser.denoise_events(events, threshold_factor=1.0)
        elif args.optim_opt == 'Variance' or args.optim_opt == 'r2':
            denoised_events, voxel_grid = denoiser.denoise_events(events, threshold_factor=1.0, optim_opt=args.optim_opt)
        else:
            raise NotImplementedError(f"Optimization strategy {args.optim_opt} not implemented")
        
        # Enhance reconstructed image with event information
        enhanced_image = denoiser.restore_image(rec_image, voxel_grid)

        # Normalize enhanced image between 0 and 255
        enhanced_image = (enhanced_image * 255).astype(np.uint8)

        # Save enhanced image
        cv2.imwrite(os.path.join(args.output_scene_path, img_names[i]), enhanced_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file_path", default=None, help="h5 file must first e tranformed to e2vid h5 format")
    parser.add_argument("--rec_scene_path", default=None)
    parser.add_argument("--output_scene_path", default=None)
    parser.add_argument("--optim_opt", default=None, help="Optimization strategy for contrast maximization: \"Variance\" or \"r2\"")
    args = parser.parse_args()

    demo(args)

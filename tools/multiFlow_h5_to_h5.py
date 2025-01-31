# import os
# import glob
# import argparse
# import numpy as np
# import tables
# import h5py
# import cv2
# from event_packagers import *
# from tqdm import tqdm

# """Version Using CPU instead of GPU for numpy operations"""

# def extract_events_images(event_path, image_dir, output_path, event_packager=hdf5_packager):
#     """
#     Extract events from input H5 file and images from image directory to a new HDF5 file.
    
#     Args:
#     event_path (str): Path to input events.h5 file
#     image_dir (str): Directory containing image files
#     output_path (str): Path to save the output HDF5 file
#     event_packager (function): Event packaging function (default: hdf5_packager)
#     """
#     # Initialize event packager
#     ep = event_packager(output_path)

#     # Find image files
#     image_file_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
#     # Load events from H5 file
#     with h5py.File(event_path, 'r') as f:
#         events_x = f['x'][:]
#         events_y = f['y'][:]
#         events_t = f['t'][:]
#         events_p = f['p'][:]

#     # Prepare metadata tracking
#     event_num = len(events_x)
#     num_pos = np.sum(events_p == 1)
#     num_neg = np.sum(events_p == 0)
#     first_ts = events_t[0]
#     last_ts = events_t[-1]
#     t0 = first_ts
#     img_cnt = 0
#     max_buffer_size = 1000000

#     # Set data availability
#     ep.set_data_available(len(image_file_paths), 0)  # images, no flow

#     # Buffers for events
#     xs, ys, ts, ps = [], [], [], []

#     # Process events in chunks
#     for i in tqdm(range(0, event_num, max_buffer_size), desc="Processing events"):
#         chunk_end = min(i + max_buffer_size, event_num)
        
#         # Add chunk of events to buffers
#         xs.extend(events_x[i:chunk_end])
#         ys.extend(events_y[i:chunk_end])
#         ts.extend(events_t[i:chunk_end])
#         ps.extend(events_p[i:chunk_end])

#         # Package events if buffer is full
#         if len(xs) >= max_buffer_size or chunk_end == event_num:
#             ep.package_events(xs, ys, ts, ps)
#             del xs[:]
#             del ys[:]
#             del ts[:]
#             del ps[:]

#     # Process images
#     for image_file in tqdm(image_file_paths, desc="Processing images"):
#         # Load and package image
#         image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
#         sensor_size = image.shape
#         image = np.expand_dims(image, axis=-1)
#         image = np.repeat(image, 3, axis=-1)

#         # Calculate timestamp for image based on event timestamps
#         # This assumes images are ordered and correspond to event timestamps
#         img_timestamp = first_ts + (last_ts - first_ts) * (img_cnt / (len(image_file_paths) - 1))
        
#         ep.package_image(image, img_timestamp, img_cnt)
#         img_cnt += 1

#     print(f"Detected sensor size: {sensor_size}")

#     # Add metadata
#     ep.add_metadata(
#         num_pos, 
#         num_neg, 
#         last_ts - t0,  # total duration
#         t0,            # start time
#         last_ts,       # end time
#         img_cnt,       # image count
#         0,             # flow count
#         sensor_size
#     )

#     print(f"Extracted data to {output_path}")
#     print(f"Total events: {event_num}")
#     print(f"Total positive events: {num_pos}")
#     print(f"Total negative events: {num_neg}")


# def main():
#     """
#     Command-line interface for converting H5 events and PNG images to new HDF5 format
#     """
#     parser = argparse.ArgumentParser(description="Convert H5 events and PNG images to new HDF5 format")
#     parser.add_argument("path", help="Directory containing events.h5 and images subdirectory")
#     parser.add_argument("--output_dir", default="/tmp/extracted_data", 
#                         help="Path to output HDF5 file")
    
#     args = parser.parse_args()
    
#     # Construct paths
#     event_path = os.path.join(args.path, 'events', 'events.h5')
#     image_dir = os.path.join(args.path, 'images')
    
#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

#     # Construct output path
#     h5_file_path = os.path.join(args.output_dir, "{}.h5".format(os.path.basename(args.path)))

#     # If h5 file exists, delete it
#     if os.path.exists(h5_file_path):
#         os.remove(h5_file_path)
    
#     # Extract events and images
#     extract_events_images(
#         event_path,
#         image_dir, 
#         h5_file_path,
#     )

# if __name__ == "__main__":
#     main()


import os
import glob
import argparse
import numpy as np
import cupy as cp
import tables
import h5py
import cv2
import torch
from event_packagers import *
from tqdm import tqdm

def extract_events_images(event_path, image_dir, output_path, gpu_id=None, event_packager=hdf5_packager):
    """Extract events and images using GPU acceleration with CuPy"""
    # Set CUDA device
    if gpu_id is not None:
        cp.cuda.Device(gpu_id).use()
        print(f"Using GPU {gpu_id} with CuPy")
    
    # Initialize event packager
    ep = event_packager(output_path)
    
    # Find image files
    image_file_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    # Load events from H5 file and transfer to GPU
    with h5py.File(event_path, 'r') as f:
        # Transfer data to GPU directly
        events_x = cp.array(f['x'][:])
        events_y = cp.array(f['y'][:])
        events_t = cp.array(f['t'][:])
        events_p = cp.array(f['p'][:])

    # Prepare metadata tracking (on GPU)
    event_num = len(events_x)
    num_pos = int(cp.sum(events_p == 1).get())
    num_neg = int(cp.sum(events_p == 0).get())
    first_ts = float(events_t[0].get())
    last_ts = float(events_t[-1].get())
    t0 = first_ts
    img_cnt = 0
    max_buffer_size = 1000000

    # Set data availability
    ep.set_data_available(len(image_file_paths), 0)

    # Process events in chunks
    for i in tqdm(range(0, event_num, max_buffer_size), desc="Processing events"):
        chunk_end = min(i + max_buffer_size, event_num)
        
        # Get chunk of events (already on GPU)
        xs = events_x[i:chunk_end]
        ys = events_y[i:chunk_end]
        ts = events_t[i:chunk_end]
        ps = events_p[i:chunk_end]

        # Move to CPU for packaging
        ep.package_events(
            cp.asnumpy(xs),
            cp.asnumpy(ys),
            cp.asnumpy(ts),
            cp.asnumpy(ps)
        )

    # Process images
    for image_file in tqdm(image_file_paths, desc="Processing images"):
        # Load image
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        sensor_size = image.shape
        
        # Process image on GPU
        image_gpu = cp.array(image)
        image_gpu = cp.expand_dims(image_gpu, axis=-1)
        image_gpu = cp.repeat(image_gpu, 3, axis=-1)
        
        # Calculate timestamp
        img_timestamp = first_ts + (last_ts - first_ts) * (img_cnt / (len(image_file_paths) - 1))
        
        # Move back to CPU for packaging
        ep.package_image(cp.asnumpy(image_gpu), img_timestamp, img_cnt)
        img_cnt += 1

    print(f"Detected sensor size: {sensor_size}")

    # Add metadata
    ep.add_metadata(
        num_pos,
        num_neg,
        last_ts - t0,
        t0,
        last_ts,
        img_cnt,
        0,
        sensor_size
    )

    print(f"Extracted data to {output_path}")
    print(f"Total events: {event_num}")
    print(f"Total positive events: {num_pos}")
    print(f"Total negative events: {num_neg}")


def main():
    parser = argparse.ArgumentParser(description="Convert H5 events and PNG images to new HDF5 format")
    parser.add_argument("path", help="Directory containing events.h5 and images subdirectory")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", 
                        help="Path to output HDF5 file")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU index to use (e.g. --gpu 0)")

    args = parser.parse_args()

    # Construct paths
    event_path = os.path.join(args.path, 'events', 'events.h5')
    image_dir = os.path.join(args.path, 'images')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Construct output path
    h5_file_path = os.path.join(args.output_dir, "{}.h5".format(os.path.basename(args.path)))

    # If h5 file exists, delete it
    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)
    
    # Extract events and images with specified GPUs
    extract_events_images(
        event_path,
        image_dir, 
        h5_file_path,
        gpu_id=args.gpu
    )

if __name__ == "__main__":
    main()
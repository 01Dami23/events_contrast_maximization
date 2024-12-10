import os
import glob
import argparse
import numpy as np
import cv2
from event_packagers import *
from tqdm import tqdm

FRAME_WIDTH = 970
FRAME_HEIGHT = 625

# "convert_and_fix_event_pixels" inspired from https://github.com/ercanburak/EVREAL.git
def convert_and_fix_event_pixels(data, upper_limit, fix_overflows=True):
    data = data.astype(np.int32)
    overflow_indices = np.where(data > upper_limit*32)
    num_overflows = overflow_indices[0].shape[0]
    if fix_overflows and num_overflows > 0:
        data[overflow_indices] = data[overflow_indices] - 65536
    data = data / 32.0
    data = np.rint(data)
    data = data.astype(np.int16)
    data = np.clip(data, 0, upper_limit)
    return data


def extract_events_images(path, output_path, event_packager=hdf5_packager):
    """
    Extract events from .npz files and images from .png files to an HDF5 file.
    
    Args:
    path (str): Directory containing 'events' and 'images' subdirectories
    output_path (str): Path to save the output HDF5 file
    event_packager (function): Event packaging function (default: hdf5_packager)
    """
    # Initialize event packager
    ep = event_packager(output_path)

    # Find event and image files
    event_file_paths = sorted(glob.glob(os.path.join(path, 'events', '*.npz')))
    image_file_paths = sorted(glob.glob(os.path.join(path, 'images', '*.png')))

    assert len(image_file_paths) == len(event_file_paths) + 1, f"Number of images ({len(image_file_paths)}) should be one more than number of events ({len(event_file_paths)})"
    
    # skip the first image as events n are from image n to n+1 (so reconstructed image refers to image n+1)
    # image_file_paths = image_file_paths[1:]

    # Prepare metadata tracking
    event_num = 0
    num_pos, num_neg = 0, 0
    first_ts = -1
    last_ts = -1
    t0 = -1
    img_cnt = 0
    max_buffer_size = 1000000

    # Set data availability
    ep.set_data_available(len(image_file_paths), 0)  # images, no flow

    # Buffers for events
    xs, ys, ts, ps = [], [], [], []

    # Process each event and image pair
    for i, (event_file, image_file) in tqdm(enumerate(zip(event_file_paths, image_file_paths)), total=len(event_file_paths)):
        # Load events
        event_data = np.load(event_file)
        event_xs = convert_and_fix_event_pixels(event_data['x'], FRAME_WIDTH - 1)
        event_ys = convert_and_fix_event_pixels(event_data['y'], FRAME_HEIGHT - 1)
        event_ts = event_data['timestamp']
        event_ps = event_data['polarity']

        # Handle first timestamp
        if first_ts == -1:
            first_ts = event_ts[0]
            t0 = event_ts[0]

        # Process events
        for j in range(len(event_xs)):
            xs.append(event_xs[j])
            ys.append(event_ys[j])
            ts.append(event_ts[j])
            ps.append(1 if event_ps[j] else 0)
            
            if event_ps[j]:
                num_pos += 1
            else:
                num_neg += 1
            
            last_ts = event_ts[j]
            event_num += 1

            if len(xs) > max_buffer_size:
                ep.package_events(xs, ys, ts, ps)
                del xs[:]
                del ys[:]
                del ts[:]
                del ps[:]

        # Load and package image
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        sensor_size = image.shape
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)

        # img_cnt += 1
        ep.package_image(image, last_ts, img_cnt)
        img_cnt += 1

    # Package any remaining events
    if xs:
        ep.package_events(xs, ys, ts, ps)

    print(f"Detected sensor size: {sensor_size}")

    # Add metadata
    ep.add_metadata(
        num_pos, 
        num_neg, 
        last_ts - t0,  # total duration
        t0,            # start time
        last_ts,       # end time
        img_cnt,       # image count
        0,             # flow count
        sensor_size
    )

    print(f"Extracted data to {output_path}")
    print(f"Total events: {event_num}")
    print(f"Total positive events: {num_pos}")
    print(f"Total negative events: {num_neg}")


def main():
    """
    Command-line interface for converting npz events and png images to HDF5
    """
    parser = argparse.ArgumentParser(description="Convert NPZ events and PNG images to HDF5")
    parser.add_argument("path", help="Directory containing 'events' and 'images' subdirectories")
    parser.add_argument("--output_dir", default="/tmp/extracted_data", 
                        help="Path to output HDF5 file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # If h5 file exists, delete it
    h5_file_path = os.path.join(args.output_dir, "{}.h5".format(os.path.basename(args.path)))

    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)
    
    # Extract events and images
    extract_events_images(
        args.path, 
        h5_file_path, 
    )


if __name__ == "__main__":
    main()
















































#     start_timestamp_s = 0.0

# # Save events
# first_idx = np.zeros((1, 1), dtype=np.int64)
# image_event_indices = [first_idx]
# total_event_num = 0

# for event_file_path in event_file_paths:
#     event_file_data = np.load(event_file_path)
#     event_num = event_file_data['x'].shape[0]
#     total_event_num = total_event_num + event_num
#     event_idx = np.array(total_event_num)
#     event_idx = np.expand_dims(event_idx, axis=-1)
#     event_idx = np.expand_dims(event_idx, axis=0)
#     image_event_indices.append(event_idx)

# image_event_indices = np.concatenate(image_event_indices)

# x_data = np.zeros(shape=total_event_num, dtype=np.uint16)
# y_data = np.zeros(shape=total_event_num, dtype=np.uint16)
# t_data = np.zeros(shape=total_event_num, dtype=np.uint32)
# p_data = np.zeros(shape=total_event_num, dtype=np.uint8)

# for frame_idx, event_file_path in enumerate(event_file_paths):
#     start_event_idx = image_event_indices[frame_idx].item()
#     end_event_idx = image_event_indices[frame_idx+1].item()
#     event_file_data = np.load(event_file_path)
#     x_data[start_event_idx:end_event_idx] = convert_and_fix_event_pixels(event_file_data['x'], FRAME_WIDTH - 1)
#     y_data[start_event_idx:end_event_idx] = convert_and_fix_event_pixels(event_file_data['y'], FRAME_HEIGHT - 1)
#     t_data[start_event_idx:end_event_idx] = event_file_data['timestamp']
#     p_data[start_event_idx:end_event_idx] = event_file_data['polarity']

#     if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
#             sensor_size = [max(xs), max(ys)]
#             print("Sensor size inferred from events as {}".format(sensor_size))

#     ep.package_events(xs, ys, ts, ps)
#     del xs[:]
#     del ys[:]
#     del ts[:]
#     del ps[:]

# t_data = t_data.astype(np.float64)
# t_data = t_data / 1000000.0  # convert us to s
# t_data = t_data - start_timestamp_s  # zeroize timestamps



# del x_data
# del y_data
# del t_data
# del p_data

# # Save images
# images_list = []
# for image_file_path in image_file_paths:
#     img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
#     img = np.expand_dims(img, axis=-1)
#     img = np.repeat(img, 3, axis=-1)

#     ep.package_image(img, event_ts[0], img_cnt)
#     img_cnt += 1
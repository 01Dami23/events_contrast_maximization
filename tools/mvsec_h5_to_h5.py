# import os
# import argparse
# import h5py
# import numpy as np
# from tqdm import tqdm
# from PIL import Image

# class MVSECtoE2VIDConverter:
#     def __init__(self, input_path, output_path):
#         """Initialize converter with input and output paths"""
#         self.input_path = input_path
#         self.output_path = output_path
        
#     def load_mvsec_data(self):
#         """Load event data from MVSEC dataset format"""
#         print("Loading MVSEC data...")
#         with h5py.File(self.input_path, 'r') as data:
#             left_camera = data['davis']['left']
            
#             # Load events
#             events = np.array(left_camera['events'])
#             self.xs = events[:, 0].astype(np.int32)
#             self.ys = events[:, 1].astype(np.int32)
#             self.ts = events[:, 2].astype(np.float64)
#             self.ps = events[:, 3].astype(np.int32)

#             print("Polarities are between: ", np.unique(self.ps))
            
#             # Load images
#             self.images = np.array(left_camera['image_raw'])
#             self.image_timestamps = np.array(left_camera['image_raw_ts'])
            
#             self.sensor_size = self.images[0].shape
            
#     def save_e2vid_format(self):
#         """Save data in E2VID format"""
#         print("Converting to E2VID format...")
        
#         # Calculate metadata
#         num_pos = np.sum(self.ps == 1)
#         num_neg = np.sum(self.ps == 0)
#         first_timestamp = self.ts[0]
#         last_timestamp = self.ts[-1]
#         duration = last_timestamp - first_timestamp

#         self.basename = os.path.splitext(os.path.basename(self.input_path))[0]
#         print("basename is: ", self.basename)
#         file_path = os.path.join(self.output_path, self.basename + '.h5')
        
#         # Create output file
#         with h5py.File(file_path, 'w') as f:
#             # Create events datasets
#             f.create_dataset('events/xs', data=self.xs, dtype=np.int16)
#             f.create_dataset('events/ys', data=self.ys, dtype=np.int16)
#             f.create_dataset('events/ts', data=self.ts, dtype=np.float64)
#             f.create_dataset('events/ps', data=self.ps, dtype=np.int16)
            
#             # Create image datasets
#             images_group = f.create_group('images')
#             for i, (image, timestamp) in enumerate(tqdm(zip(self.images, self.image_timestamps), 
#                                                       total=len(self.images), 
#                                                       desc="Saving images")):
#                 # Convert grayscale to RGB
#                 rgb_image = np.stack([image] * 3, axis=-1)
#                 images_group.create_dataset(f'image{i:09d}', data=rgb_image)
#                 images_group.create_dataset(f'timestamp{i:09d}', data=timestamp)
            
#             # Add metadata
#             metadata = f.create_group('metadata')
#             metadata.create_dataset('num_events', data=num_pos + num_neg)
#             metadata.create_dataset('num_pos', data=num_pos)
#             metadata.create_dataset('num_neg', data=num_neg)
#             metadata.create_dataset('duration', data=duration)
#             metadata.create_dataset('t0', data=first_timestamp)
#             metadata.create_dataset('tk', data=last_timestamp)
#             metadata.create_dataset('num_imgs', data=len(self.images))
#             metadata.create_dataset('sensor_resolution', data=self.sensor_size)

# def main():
#     parser = argparse.ArgumentParser(description="Convert MVSEC dataset to E2VID format")
#     parser.add_argument("--input_path", help="Path to input MVSEC .h5 file")
#     parser.add_argument("--output_path", help="Path to output E2VID format .h5 file")
#     parser.add_argument("--start_ts", type=float, default=None, 
#                         help="Start timestamp offset from beginning of recording")
#     parser.add_argument("--end_ts", type=float, default=None,
#                         help="End timestamp offset from beginning of recording")
    
#     args = parser.parse_args()
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
#     # Initialize converter
#     converter = MVSECtoE2VIDConverter(args.input_path, args.output_path)
    
#     try:
#         # Load and process data
#         converter.load_mvsec_data()
        
#         # Apply timestamp filtering if specified
#         if args.start_ts is not None or args.end_ts is not None:
#             base_ts = converter.ts[0]
#             mask = np.ones_like(converter.ts, dtype=bool)
            
#             if args.start_ts is not None:
#                 start_ts = base_ts + args.start_ts
#                 mask &= (converter.ts >= start_ts)
            
#             if args.end_ts is not None:
#                 end_ts = base_ts + args.end_ts
#                 mask &= (converter.ts <= end_ts)
            
#             # Filter events
#             converter.xs = converter.xs[mask]
#             converter.ys = converter.ys[mask]
#             converter.ts = converter.ts[mask]
#             converter.ps = converter.ps[mask]
            
#             # Filter images
#             if args.start_ts is not None:
#                 start_idx = np.argmax(converter.image_timestamps >= start_ts)
#                 converter.images = converter.images[start_idx:]
#                 converter.image_timestamps = converter.image_timestamps[start_idx:]
            
#             if args.end_ts is not None:
#                 end_idx = np.argmax(converter.image_timestamps >= end_ts)
#                 converter.images = converter.images[:end_idx]
#                 converter.image_timestamps = converter.image_timestamps[:end_idx]
        
#         # Save converted data
#         converter.save_e2vid_format()
#         print(f"Successfully converted data to {args.output_path}")
        
#     except Exception as e:
#         print(f"Error during conversion: {e}")
#         raise

# if __name__ == "__main__":
#     main()




















import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

class MVSECtoE2VIDConverter:
    def __init__(self, input_path, output_path):
        """Initialize converter with input and output paths"""
        self.input_path = input_path
        self.output_path = output_path
        
    def load_mvsec_data(self):
        """Load event data from MVSEC dataset format"""
        print("Loading MVSEC data...")
        with h5py.File(self.input_path, 'r') as data:
            left_camera = data['davis']['left']
            
            # Load events
            events = np.array(left_camera['events'])
            self.xs = events[:, 0].astype(np.int16)
            self.ys = events[:, 1].astype(np.int16)
            self.ts = events[:, 2].astype(np.float64)
            self.ps = events[:, 3].astype(np.int16)
            
            # Load images
            self.images = np.array(left_camera['image_raw'])
            self.image_timestamps = np.array(left_camera['image_raw_ts'])
            
            self.sensor_size = self.images[0].shape
            
    def save_e2vid_format(self):
        """Save data in E2VID format using the HDF5 packager"""
        print("Converting to E2VID format...")

        self.basename = os.path.splitext(os.path.basename(self.input_path))[0]
        print("basename is: ", self.basename)
        file_path = os.path.join(self.output_path, self.basename + '.h5')
        
        # Initialize the HDF5 packager
        packager = hdf5_packager(file_path)
        
        # Set available data
        packager.set_data_available(len(self.images), 0)  # No flow data
        
        # Package events
        chunk_size = packager.max_buffer_size
        for i in tqdm(range(0, len(self.xs), chunk_size), desc="Packaging events"):
            end_idx = min(i + chunk_size, len(self.xs))
            packager.package_events(
                self.xs[i:end_idx],
                self.ys[i:end_idx],
                self.ts[i:end_idx],
                self.ps[i:end_idx]
            )
        
        # Package images
        for i, (image, timestamp) in enumerate(tqdm(zip(self.images, self.image_timestamps), 
                                                  total=len(self.images), 
                                                  desc="Packaging images")):
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            packager.package_image(image, timestamp, i)
        
        # Calculate metadata
        num_pos = np.sum(self.ps)
        num_neg = len(self.ps) - num_pos
        first_timestamp = self.ts[0]
        last_timestamp = self.ts[-1]
        
        # Add metadata
        packager.add_metadata(
            num_pos=num_pos,
            num_neg=num_neg,
            duration=last_timestamp - first_timestamp,
            t0=first_timestamp,
            tk=last_timestamp,
            num_imgs=len(self.images),
            num_flow=0,
            sensor_size=self.sensor_size
        )

class hdf5_packager:
    """
    This class packages data to hdf5 files
    """
    def __init__(self, output_path, max_buffer_size=1000000):
        print("CREATING FILE IN {}".format(output_path))
        self.events_file = h5py.File(output_path, 'w')
        self.max_buffer_size = max_buffer_size
        
        # Create event datasets
        self.event_xs = self.events_file.create_dataset("events/xs", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True)
        self.event_ys = self.events_file.create_dataset("events/ys", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True)
        self.event_ts = self.events_file.create_dataset("events/ts", (0,), dtype=np.dtype(np.float64), maxshape=(None,), chunks=True)
        self.event_ps = self.events_file.create_dataset("events/ps", (0,), dtype=np.dtype(np.bool_), maxshape=(None,), chunks=True)

    def append_to_dataset(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data):] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append_to_dataset(self.event_xs, xs)
        self.append_to_dataset(self.event_ys, ys)
        self.append_to_dataset(self.event_ts, ts)
        self.append_to_dataset(self.event_ps, ps)

    def package_image(self, image, timestamp, img_idx):
        image_dset = self.events_file.create_dataset(
            f"images/image{img_idx:09d}",
            data=image,
            dtype=np.dtype(np.uint8)
        )
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr"

    def add_event_indices(self):
        datatypes = ['images', 'flow']
        for datatype in datatypes:
            if datatype in self.events_file.keys():
                s = 0
                added = 0
                ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                for image in self.events_file[datatype]:
                    img_ts = self.events_file[datatype][image].attrs['timestamp']
                    event_idx = np.searchsorted(ts, img_ts)
                    if event_idx == len(ts):
                        added += len(ts)
                        s += self.max_buffer_size
                        ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                        event_idx = np.searchsorted(ts, img_ts)
                    event_idx = max(0, event_idx-1)
                    self.events_file[datatype][image].attrs['event_idx'] = event_idx + added

    def add_metadata(self, num_pos, num_neg, duration, t0, tk, num_imgs, num_flow, sensor_size):
        self.events_file.attrs['num_events'] = num_pos + num_neg
        self.events_file.attrs['num_pos'] = num_pos
        self.events_file.attrs['num_neg'] = num_neg
        self.events_file.attrs['duration'] = duration
        self.events_file.attrs['t0'] = t0
        self.events_file.attrs['tk'] = tk
        self.events_file.attrs['num_imgs'] = num_imgs
        self.events_file.attrs['num_flow'] = num_flow
        self.events_file.attrs['sensor_resolution'] = sensor_size
        self.add_event_indices()

    def set_data_available(self, num_images, num_flow):
        if num_images > 0:
            self.image_dset = self.events_file.create_group("images")
            self.image_dset.attrs['num_images'] = num_images
        if num_flow > 0:
            self.flow_dset = self.events_file.create_group("flow")
            self.flow_dset.attrs['num_images'] = num_flow

def main():
    parser = argparse.ArgumentParser(description="Convert MVSEC dataset to E2VID format")
    parser.add_argument("--input_path", help="Path to input MVSEC .h5 file")
    parser.add_argument("--output_path", help="Path to output E2VID format .h5 file")
    parser.add_argument("--start_ts", type=float, default=None, 
                        help="Start timestamp offset from beginning of recording")
    parser.add_argument("--end_ts", type=float, default=None,
                        help="End timestamp offset from beginning of recording")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Initialize converter
    converter = MVSECtoE2VIDConverter(args.input_path, args.output_path)
    
    try:
        # Load and process data
        converter.load_mvsec_data()
        
        # Apply timestamp filtering if specified
        if args.start_ts is not None or args.end_ts is not None:
            base_ts = converter.ts[0]
            mask = np.ones_like(converter.ts, dtype=bool)
            
            if args.start_ts is not None:
                start_ts = base_ts + args.start_ts
                mask &= (converter.ts >= start_ts)
            
            if args.end_ts is not None:
                end_ts = base_ts + args.end_ts
                mask &= (converter.ts <= end_ts)
            
            # Filter events
            converter.xs = converter.xs[mask]
            converter.ys = converter.ys[mask]
            converter.ts = converter.ts[mask]
            converter.ps = converter.ps[mask]
            
            # Filter images
            if args.start_ts is not None:
                start_idx = np.argmax(converter.image_timestamps >= start_ts)
                converter.images = converter.images[start_idx:]
                converter.image_timestamps = converter.image_timestamps[start_idx:]
            
            if args.end_ts is not None:
                end_idx = np.argmax(converter.image_timestamps >= end_ts)
                converter.images = converter.images[:end_idx]
                converter.image_timestamps = converter.image_timestamps[:end_idx]
        
        # Save converted data
        converter.save_e2vid_format()
        print(f"Successfully converted data to {args.output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

if __name__ == "__main__":
    main()
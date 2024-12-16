# hscimproc

This module contains a single class called `FrameGenerator` which is used to extract frames from MRAW (Motion RAW) files commonly found on high speed cameras, or from other standard image formats.

## Dependencies

- numpy
- opencv-python (optional, for video playback and MP4 output)

## Installation

Clone and `cd` this repo. Then, in the environment you would like to use this package:

`pip install ./`

## Class: FrameGenerator

### Usage

FrameGenerator can provide a generator of either 8-bit or 16-bit image arrays. It has been tested with 8, 12 and 16 bit images. The class has the following methods:

- `frame_generator_int16(self, mraw, start_frame=0, n_frames=None)`: This method is a generator that yields frames from an MRAW file. The parameters are:
  - `mraw`: The MRAW file to extract frames from.
  - `start_frame`: The frame to start extraction from (default 0).
  - `n_frames`: The number of frames to extract. Defaults to last frame.
  - `scale`: For 12 bit images, the data is bit-shifted to start at the most significant bit. Images will be dark without this. Defaults to True.

- `frame_generator_int8(self,mraw_start_frame=0, n_frames=None)`: Exactly the same as `frame_generator_int16` but returns 8-bit images.

- `standard_format_frame_generator(self, glob_pattern, start_frame=0, n_frames=None, sort_fn=None)`: This method is a generator that yields frames from a standard image format. The parameters are:
  - `glob_pattern`: A glob pattern to match the images you want to extract frames from.
  - `start_frame`: The frame to start extraction from (default 0).
  - `n_frames`: The number of frames to extract. Defaults to last frame.
  - `sort_fn`: A function to sort the images. If not provided, the code will sort by the last group of numbers found in the filename.

### Example 1: Create a generator for postprocessing with a .mraw file

```
from hscimproc import FrameGenerator
generator = FrameGenerator()

filename = 'camera_1.mraw'

frames = generator.raw_frame_generator_int8(filename,1000,12,0)

for frame in frames:
    # your postprocessing goes here

```

### Example 2: Create a generator using a standard image format

```
from hscimproc import FrameGenerator
generator = standard_format_frame_generator()

# specify a glob pattern to capture all images
filename = 'run_XXXX/CAMERA1_*.tif'

# If no sort function is passed, the code will default to sorting by the last group of numbers found in the filename. An example of setting your own (this would sort by everything after the last underscore):
sort_fn = lambda x: int(x.split('_')[-1].split('.')[0])

frames = generator.standard_frame_generator_int8(filename,start_frame=0,n_frames=None,sort_fn=sort_fn)

for frame in frames:
    # your postprocessing goes here

```

### Example 3: Video playback (requires package `opencv-python`)
```
# Import and instantiate the FrameGenerator
from hscimproc import FrameGenerator
generator = FrameGenerator()

# Default fps is 30. Setting fps=0 will allow you to step through images by pressing any key. 'q' or 'esc' will quit playback.

# Settings for playback
settings = {
        'fps':100,
        'filename':'camera_1.mraw'
        'start_frame':0,
        'n_frames':1000 
}

frames = generator.raw_frame_generator_int8(frames)
generator.play(frames,**settings)
```

### Example 4: Turn your frames into a cool MP4 to show all your friends

This is not guaranteed to work well yet as the OpenCV VideoWriter class is very sensitive to your ffmpeg setup and different codecs. Let [Gerard Armstrong](gerard.armstrong@unisq.edu.au) know if you would like to use this feature and it doesn't work for you.

```
from hscimproc import FrameGenerator
generator = FrameGenerator()

# instantiate either a standard format or raw frame generator
frames = generator.standard_frame_generator_int8('run_XXXX/CAMERA1_*.tif',1000) # or generator.raw_frame_generator_int8('camera_1.mraw')

generator.to_mp4(frames,'output_file.mp4',fps=100)

```

## Known Issues

1. You will probably face issues if you pass a generator through nested function calls. This appears to be an issue with the Python or glibc garbage collector. Making persistent references to the generator does not seem to help. This will most likely manifest as random segfaults when iterating through the generator (random meaning sometimes it will work and sometimes the generator has been GC'd). The only advice I can give for this is to use the generator in the same function as it is istantiated, and avoid passing it to other functions.

## Credits

The original memmap code was developed by J. Javh, J. Slavič and M. Boltežar. You can find it at https://github.com/ladisk/pyMRAW. This code adds:
- A generator for image frames to avoid large usage of RAM
- MP4 output functionality
- Standard image format frame generators as a convenience. This allows users to use either mraw videos or image files in postprocessing.

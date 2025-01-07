# hscimproc

This module contains several classes used to extract frames from MRAW (Motion RAW) files from high speed cameras, or from other standard image formats (TIFFs, PNGs etc).

- **FrameGenerator**: Base class
- **RawFrameGenerator**: Generate frames from a .mraw and .cihx file.
- **StandardFormatFrameGenerator**: For image format files like TIFFs, PNGs, JPGs or any format PIL can open.
- **FrameGeneratorCollection**: Container class for iterating over multiple FrameGenerator-like objects. Returns the image at current index from each set of frames, or `None` if no image.
- **AlignedFrameGenerator**: Child class of FrameGenerator. Looks at .cihx metadata and can align two sets of frames with one another.
- **AlignedStandardFormatFrameGenerator**: Align sets of frames based on defined FPS and offset.

## Dependencies

- numpy
- opencv-python (optional, for video playback and MP4 output)

## Installation

Clone and `cd` this repo. Then, in the environment you would like to use this package:

`pip install ./`

## Class: FrameGenerator

When instantiating any `FrameGenerator` there are various options you can pass.


- **hflip**: Flip images horizontally
- **vflip**: Flip images vertically
- **start_frame**: Frame to start at
- **n_frames**: How many frames to output
- **brighten**: Scale intensities to use the maximum bit depth

## Examples

### Example 1: Create a generator for postprocessing with a .mraw file

```
from hscimproc import RawFrameGenerator
filename = 'camera_1.mraw'
frame_generator = RawFrameGenerator(filename)

for frame in frame_generator:
    # your postprocessing goes here

```

### Example 2: Create a generator using a standard image format

```
from hscimproc import StandardFormatFrameGenerator

# specify a glob pattern to capture all images
glob_pattern = 'run_XXXX/CAMERA1_*.tif'

# If no sort function is passed, the code will default to sorting by the last group of numbers found in the filename. An example of setting your own (this would sort by everything after the last underscore):
sort_fn = lambda x: int(x.split('_')[-1].split('.')[0])

frame_generator = StandardFormatFrameGenerator(glob_pattern,sort_fn=sort_fn)

for frame in frame_generator:
    # your postprocessing goes here

```

### Example 3: Video playback (requires package `opencv-python`)
```
from hscimproc import RawFrameGenerator

file_path = '/path/to/mraw.mraw'

# Create any subclass of FrameGenerator.
generator = RawFrameGenerator(file_path)

# Default fps is 30. Setting fps=0 will allow you to step through images by pressing any key. 'q' or 'esc' will quit playback.

generator.play(fps=30)

```

### Example 4: Turn your frames into a cool MP4 to show all your friends

This is not guaranteed to work well yet as the OpenCV VideoWriter class is very sensitive to your ffmpeg setup and different codecs. Let [Gerard Armstrong](gerard.armstrong@unisq.edu.au) know if you would like to use this feature and it doesn't work for you.

```
from hscimproc import FrameGenerator

# instantiate either a standard format or raw frame generator
frames = StandardFormatFrameGenerator('run_XXXX/CAMERA1_*.tif')

frames.to_mp4(frames,'output_file.mp4',fps=100)
```

## Example 5: Changing output resolution and offset

You can also set the frame output to a custom resolution. This is useful in image tracking analysis where the cameras may be calibrated based on full frame resolution, but the data was captured with a downsize window and offset.

Unfortunately there are all sorts of different conventions for image view axes. I try to stick to OpenCV which is [left to right, top-down] as much as possible. However, camera center offsets are usually specified in graph xy coordinates (left to right, bottom-up), so that is used here. Once you have the frame generator you can set these properties like so:

```
frame_generator.set_output_resolution((1024,1024))
frame_generator.set_center_offset((100,100))
```

## Example 6: Aligning Frames

For an AlignedRawFrameGenerator the FPS and offset is automatically found from the metadata. For an AlignedStandardFormatFrameGenerator you must specify the `fps` and `start_offset` attributes. In either case, after you create the object you must bind its parent.

```
# Create the first frame generator
fg = RawFrameGenerator(...)

# Bind the parent
fg2 = AlignedRawFrameGenerator(...)
fg2.set_parent(fg)

```

You can then put these into a FrameCollection or iterate over each separately..

## Known Issues

1. You will probably face issues if you pass a generator through nested function calls. This appears to be an issue with the Python or glibc garbage collector. Making persistent references to the generator does not seem to help. This will most likely manifest as random segfaults when iterating through the generator (random meaning sometimes it will work and sometimes the generator has been GC'd). The only advice I can give for this is to use the generator in the same function as it is instantiated, and avoid passing it to other functions.

## Credits

The original memmap code was developed by J. Javh, J. Slavič and M. Boltežar. You can find it at https://github.com/ladisk/pyMRAW. This code adds:
- A generator for image frames to avoid large usage of RAM
- MP4 output functionality
- Standard image format frame generators as a convenience. This allows users to use either mraw videos or image files in postprocessing.

from tkinter import font
import numpy as np
import lxml.etree
import os
import glob
import re
from PIL import Image
from time import sleep
import hashlib
import cv2 as cv
from typing import Union

"""
Author: Gerard Armstrong
Date: 20 May 2024

Credit to J. Javh, J. Slavič and M. Boltežar for the original memmap code
https://github.com/ladisk/pyMRAW

This code makes several improvements:
- Added a generator for frames which prevents large files from being loaded into memory at once
- Added playback functionality
- Added functionality to convert a generator of frames to an mp4 file
- Added functionality to read standard image formats

"""

if not os.path.exists('/tmp/hscimproc_data'):
    os.mkdir('/tmp/hscimproc_data')


class FrameGenerator:

    def __init__(self, name='Unknown Frame Generator', overlay_frame_index=True):
        self.aligned_frame_generator = None
        self.brighten = False
        self.name = name
        self.overlay_frame_index = overlay_frame_index

    def hflip(self, im):
        return np.flip(im, 1)

    def vflip(self, im):
        return np.flip(im, 0)

    def play(self, fps=30, **kwargs) -> None:

        assert fps >= 0

        # Check if opencv is installed
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError('OpenCV is required for FrameGenerator playback')

        print('Press q or esc to stop playback')

        for frame in self:

            # if self.output_resolution is not None:
            #     frame = self.pad_to(frame,self.output_resolution,center_offset=self.center_offset)

            player_name = self.device_name if self.device_name is not None else self.name

            cv.imshow(player_name, frame)
            cv.displayOverlay(
                player_name, f'{player_name}: Frame {self.current_index}')

            if fps == 0:
                while True:
                    keypress = cv.waitKey()
                    # Don't let meta key increment frames
                    if not keypress == 235:
                        break
            else:
                keypress = cv.waitKey(1000//fps)

            if keypress == ord('q') or keypress == 27:
                cv.destroyAllWindows()
                break
            if keypress == ord('s'):
                self.save_tiff(frame)

    def to_mp4(self, filename='output.mp4', fps=30, four_cc="MP4V"):
        """
        Converts a list of frames to an mp4 video file

        Parameters:
        frames: generator of frames
        filename: name of the output video file
        fps: frames per second of the output video
        """

        # Check if opencv is installed
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError('OpenCV is required for this function')

        self.current_index = self.start_frame
        frame = self.get_frame(self.current_index)

        # Get the shape of the first frame
        shape = frame.shape
        shape = (shape[1], shape[0])

        if len(shape) == 3:
            isColor = True
            shape = (shape[1], shape[0], 3)
        else:
            isColor = False

        # Create a video writer object
        fourcc = cv.VideoWriter_fourcc(*four_cc)
        out = cv.VideoWriter(filename, fourcc, fps, shape, isColor=isColor)

        out.write(frame)
        for frame in self:
            out.write(frame)

        out.release()

    def __iter__(self):
        self.current_index = self.start_frame - 1
        return self

    def get_previous(self):
        if self.current_index > self.start_frame:
            self.current_index -= 1
        return next(self)

    def get_frame(self, n_frame):

        n_frame = int(n_frame)

        d_frame = n_frame - self.current_index
        self.dt = 1./self.fps * d_frame
        self.t = 1./self.fps * n_frame

        frame_size_bytes = self.frame_size_bytes
        bit_depth = self.bit_depth
        mmap = self.mmap
        output_dtype = self.output_dtype
        start_frame = self.start_frame
        n_frames = self.n_frames
        im_shape = self.im_shape
        output_dtype_itemsize = self.output_dtype_itemsize

        if n_frame >= start_frame + n_frames:
            raise StopIteration

        frame_offset_bytes = int(n_frame * frame_size_bytes)
        mmap.seek(frame_offset_bytes)
        if mmap.tell() + frame_size_bytes > os.path.getsize(self.mraw):
            raise IOError("NOOOOO")
        data = mmap.read(frame_size_bytes)
        data = np.frombuffer(data, dtype=np.uint8)

        if bit_depth == 12:

            fst_uint8, mid_uint8, lst_uint8 = data.reshape(
                (-1, 3)).astype(np.uint16).T

            fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)

            snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8

            mat = np.reshape(np.concatenate(
                (fst_uint12[:, None], snd_uint12[:, None]), axis=1), im_shape)

        elif bit_depth == 8 or bit_depth == 16:
            mat = data.reshape(im_shape)

        bitshift_delta = output_dtype_itemsize*8 - bit_depth

        if self.scale:
            if bitshift_delta > 0:
                mat <<= bitshift_delta
            elif bitshift_delta < 0:
                mat >>= -bitshift_delta

        if self.apply_hflip:
            mat = np.fliplr(mat)
        if self.apply_vflip:
            mat = np.flipud(mat)

        mat = mat.astype(output_dtype)

        self.im_shape = mat.shape

        if self.brighten:
            # if not hasattr(self, 'brighten_mean'):
            brighten_mean = mat.mean()
            brighten_std = mat.std()

            # Rescale image from -4 to 4 sigma
            mat = (mat - brighten_mean)/brighten_std
            half_width = 2**mat.itemsize/2-1
            mat = half_width * (1 + mat/4)

            mat = np.clip(mat, min=0,
                          max=2**bit_depth-1).astype(self.output_dtype)

        if hasattr(self, 'image_processing_fn'):
            mat = self.image_processing_fn(mat)

        if self.output_resolution is not None:
            # mat = self.pad_to(mat, resolution=self.output_resolution,
            #                   center_offset=self.center_offset)
            mat = self.shift(mat)

        if self.overlay_frame_index:
            self.info_overlay(mat, n_frame)

        return mat

    def info_overlay(self, frame, n_frame):

        overlays = [
            f'Device: {self.device_name}',
            f'Frames since trigger: {self.start_offset+n_frame}',
            f'fps: {int(self.fps)}',
            f'Frame: {n_frame}',
            f't: {self.t:.4f}s',
        ]

        font_face = cv.FONT_HERSHEY_PLAIN
        font_scale = 1
        font_thickness = 1
        spacing = 2
        padding = 10
        line_height = cv.getTextSize(
            ' ', font_face, font_scale, font_thickness)[0][1]
        y_max = frame.shape[0] - int(len(overlays)
                                     * line_height * spacing + padding * 2)
        x_max = int(max([cv.getTextSize(
            overlay, font_face, font_scale, font_thickness)[0][0] for overlay in overlays]) + padding * 2
        )

        cv.rectangle(frame, (0, y_max), (x_max, frame.shape[0]), color=[
            30, 50, 30], thickness=-1)

        for i, overlay in enumerate(overlays):
            y = int(frame.shape[0]-i*line_height*spacing-padding)
            frame[:] = cv.putText(frame, overlay, (padding, y),
                                  cv.FONT_HERSHEY_PLAIN, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    def shift(self, frame):
        sensorXpos = self.sensorXpos
        sensorYpos = self.sensorYpos

        blank = np.zeros(self.output_resolution, dtype=self.output_dtype)

        blank[sensorYpos:sensorYpos+frame.shape[0],
              sensorXpos:sensorXpos+frame.shape[1]] = frame
        return blank

    def get_current_frame(self):
        return self.get_frame(self.current_index)

    def __next__(self):
        frame = self.get_frame(self.current_index+1)
        self.current_index += 1
        return frame

    # def pad(self, frame, pad):
    #     # follows (before, after) format. For both axes, use ( (before_x,after_x),(before_y,after_y) )
    #     return np.pad(frame, pad)

    # def pad_to(self, frame, resolution, center_offset=(0, 0)):

    #     # uses numpy coordinates so it goes (top,bottom), (left,right)
    #     # center_offset uses normal xy coordinates (since that is what camera reports)

    #     assert len(resolution) == 2

    #     hor_res, vert_res = resolution
    #     vert_len, hor_len = frame.shape[:2]

    #     hor_center = hor_res // 2 + center_offset[0]
    #     vert_center = vert_res // 2 + center_offset[1]

    #     assert vert_res >= vert_len
    #     assert hor_res >= hor_len

    #     left_pad = hor_center - hor_len // 2
    #     right_pad = hor_res - hor_center - hor_len // 2
    #     bottom_pad = vert_center - vert_len // 2
    #     top_pad = vert_res - vert_center - vert_len // 2

    #     return np.pad(frame, ((top_pad, bottom_pad), (left_pad, right_pad)))

    def save_tiff(self, frame):
        import cv2 as cv

        files = glob.glob(
            f'/tmp/hscimproc_data/{self.name.replace(" ", "_")}*')
        cv.imwrite(
            f'/tmp/hscimproc_data/{self.name.replace(" ", "_")}{len(files)}.tiff', frame)

    def set_output_resolution(self, resolution):
        self.output_resolution = resolution

    def set_center_offset(self, center_offset):
        self.center_offset = center_offset

    def to_tmp_file(self, frame: np.array):
        """
        Take a numpy array/image, save to temporary folder and return the filename
        """
        hashname = f'/tmp/hscimproc_data/{hashlib.sha1(frame.tobytes()).hexdigest()}.tiff'
        cv.imwrite(hashname, frame)
        return hashname

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()


class RawFrameGenerator(FrameGenerator):

    def __init__(self,
                 mraw,
                 start_frame=0,
                 output_dtype=np.uint8,
                 n_frames=None,
                 scale=True,
                 hflip=False,
                 vflip=False,
                 brighten=False,
                 name='Unknown Frame Generator'
                 ):

        # determine the image shape so it doesn't need to be passed
        (im_shape,
         total_frames,
         fps,
         bit_depth,
         start_offset,
         sensorXpos,
         sensorYpos,
         deviceName
         ) = self.get_metadata(mraw)

        self.mmap = open(mraw, 'rb')

        self.brighten = brighten

        if n_frames is None:
            n_frames = total_frames - start_frame
        elif n_frames > total_frames - start_frame:
            print(
                'Warning: n_frames is greater than the total number of frames in the file')
            n_frames = total_frames - start_frame

        px_per_frame = np.prod(im_shape)

        output_dtype_itemsize = np.array([0], dtype=output_dtype).itemsize

        if bit_depth == 12 or bit_depth == 8:
            dtype = np.uint8
        elif bit_depth == 16:
            dtype = np.uint16

        self.frame_size_bytes = int(px_per_frame * bit_depth / 8)
        self.bit_depth = bit_depth
        self.px_per_frame = px_per_frame
        self.output_dtype = output_dtype
        self.bit_depth = bit_depth
        self.n_frames = n_frames
        self.start_frame = start_frame
        self.total_frames = total_frames
        self.output_dtype_itemsize = output_dtype_itemsize
        self.dtype = dtype
        self.im_shape = im_shape
        self.apply_hflip = hflip
        self.apply_vflip = vflip
        self.scale = scale
        self.mraw = mraw
        self.fps = float(fps)
        self.start_offset = start_offset
        self.device_name = deviceName

        self.sensorXpos = sensorXpos
        self.sensorYpos = sensorYpos
        self.set_center_offset((sensorXpos, sensorYpos))

        self.current_index = start_frame
        self.output_resolution = None
        # self.center_offset = (0, 0)

        super().__init__(name)

    def get_metadata(self, filename):

        # find the metadatata file
        mraw_folder = os.path.abspath(os.path.relpath(f'{filename}/..'))
        files = os.listdir(mraw_folder)
        files = [f for f in files if f.startswith(
            filename.split('/')[-1].split('.')[0])]

        xml_file = [f for f in files if f.split('.')[-1].startswith('cih')][0]
        xml_file = os.path.join(mraw_folder, xml_file)

        with open(xml_file, 'rb') as f:
            xml_data = f.read()
            # Find the start of the XML content and decode it
            start = xml_data.find(b'<')
            # xml_data = xml_data[start:].decode('utf-8')
            xml_data = xml_data[start:]

        # root = xml.etree.ElementTree.fromstring(xml_data)

        root = lxml.etree.fromstring(xml_data)

        imageDataInfo = root.find('imageDataInfo')
        resolution = imageDataInfo.find('resolution')
        resolution = resolution.find(
            'width').text, resolution.find('height').text
        resolution = int(resolution[1]), int(resolution[0])
        total_frames = int(root.find('frameInfo').find('totalFrame').text)
        fps = int(root.find('recordInfo').find('recordRate').text)
        start_offset = int(root.find('frameInfo').find('startFrame').text)
        bitDepth = int(imageDataInfo.find('effectiveBit').find('depth').text)

        sensorXpos = int(imageDataInfo.find(
            'segmentPos').find('sensorXpos').text)//2
        sensorYpos = int(imageDataInfo.find(
            'segmentPos').find('sensorYpos').text)//2

        deviceName = root.find('deviceInfo').find('deviceName').text

        f = None
        sleep(0.5)

        return resolution, total_frames, fps, bitDepth, start_offset, sensorXpos, sensorYpos, deviceName


class StandardFormatFrameGenerator(FrameGenerator):

    def __init__(self,
                 pattern,
                 fps=1,
                 start_frame=0,
                 n_frames=None,
                 sort_fn=lambda x: int(re.findall('[0-9]+', x)[-1]),
                 nostop=False,
                 hflip=False,
                 vflip=False,
                 brighten=False,
                 name='Unknown Frame Generator'
                 ):
        """
        Generator for frames in a standard image format

        Parameters:

        pattern: pattern to match the images. e.g. 'images/*.png'
        sort_fn: function to sort the images. Default is to sort by the last number in the filename
        nostop: if True, ignore bad image file reads and continue
        start_frame: frame to start at
        n_frames: number of frames to read

        """

        files = glob.glob(pattern)

        if not files:
            raise ValueError('No files found matching pattern', pattern)

        files.sort(key=sort_fn)

        if not n_frames:
            n_frames = np.inf

        # n_frames = min(n_frames,len(files)-start_frame)
        n_frames = len(files)

        # files = files[start_frame:start_frame+n_frames]

        self.files = files
        self.pattern = pattern
        self.start_frame = start_frame
        self.n_frames = n_frames
        self.sort_fn = sort_fn
        self.nostop = nostop
        self.apply_hflip = hflip
        self.apply_vflip = vflip
        self.brighten = brighten
        self.fps = fps
        self.current_index = start_frame

        super().__init__(name)

    def __next__(self):
        self.current_index += 1

        if self.current_index > len(self.files):
            raise StopIteration

        return self.get_frame(self.current_index)

    def get_frame(self, idx):

        if idx >= len(self.files):
            raise StopIteration

        f = self.files[idx]

        try:
            with Image.open(f) as im:
                im = np.array(im)

                if self.apply_hflip:
                    im = self.hflip(im)
                if self.apply_vflip:
                    im = self.vflip(im)

                self.im_shape = im.shape

                return im
        except FileNotFoundError:
            print('Warning: Could not open file', f)

            if not self.nostop:
                print(
                    'Stopping due to bad image file read. To ignore bad file reads, pass nostop=True')
                return None


class AlignedFrameGenerator(FrameGenerator):
    # def __init__(self,parent,name='Unknown Frame Generator'):
    #     super().__init__(name=name)

    def set_parent(self, parent):
        self.parent = parent

        if isinstance(self, AlignedStandardFormatFrameGenerator):
            self.get_frame_fn = super(
                StandardFormatFrameGenerator, self).get_frame
        else:
            self.get_frame_fn = super().get_frame

    def get_frame(self, n_frame, **kwargs):
        """
        Take a frame index from another FrameGenerator and return the corresponding frame for this one.

        Consider that the frames are functions of time that started from the same trigger

        t1 = (start_frame + i_1)*fps1
        t2 = (start_frame2 + i_2)*fps2

        If those times are equal this means that

        i_2 = (start_frame+i_1)*fps1/fps2 - start_frame2
        """

        sf1 = self.parent.start_offset
        sf2 = self.start_offset
        fps1 = self.fps
        fps2 = self.parent.fps

        i = (sf1 + n_frame)*fps1/fps2 - sf2

        if i >= 0 and i <= self.total_frames and i % 1 == 0:
            return self.get_frame_fn(i, **kwargs)
        else:
            return None


class AlignedRawFrameGenerator(RawFrameGenerator, AlignedFrameGenerator):
    # AlignedFrameGenerator sets the RawFrameGenerator get_frame function for this class
    pass


class AlignedStandardFormatFrameGenerator(StandardFormatFrameGenerator, AlignedFrameGenerator):
    # AlignedFrameGenerator sets the StandardFormatFrameGenerator get_frame function for this class
    pass


class FrameGeneratorCollection:

    def __init__(self, frame_generators: list[FrameGenerator]):

        self.frame_generators = frame_generators
        self.current_index = frame_generators[0].current_index
        self.t = 0.
        # self.dt = frame_generators[0].dt

    def play(self, fps=30):

        assert fps >= 0

        try:
            import cv2 as cv
        except ImportError:
            raise ImportError("You need OpenCV to play back a frame generator")

        if fps == 0:
            delay = 0
        else:
            delay = 1000 // fps

        for frames in self:
            for j, frame in enumerate(frames):
                if frame is not None:
                    player = self.frame_generators[j]
                    player_name = player.device_name if player.device_name is not None else player.name

                    if player.overlay_frame_index:
                        cv.putText(frame, str(player.current_index), (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                    cv.imshow(player_name, frame)
                    cv.displayOverlay(
                        player_name, f'{player_name}: Frame {self.frame_generators[j].current_index}')

            while True:
                key = cv.waitKey(delay)

                if key == ord('q'):
                    cv.destroyAllWindows()
                    return

                # Don't let meta key increment frames
                if not key == 235:
                    break

    def __iter__(self):
        return self

    def __next__(self):

        frames = []

        for frame_gen in self.frame_generators:
            # idx = frame_gen.current_index
            frames.append(next(frame_gen))
            # frame_gen.current_index += 1

        self.t = self.frame_generators[0].t
        self.dt = self.frame_generators[0].dt
        self.current_index = self.frame_generators[0].current_index

        self.is_multiview = True if sum(
            [im is not None for im in frames]) > 1 else False

        return frames

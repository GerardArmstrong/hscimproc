import numpy as np
import lxml.etree
import os
import glob
import re
from PIL import Image
from time import sleep
import hashlib
import cv2 as cv

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

if not os.path.exists('/tmp/hscimproc'):
    os.mkdir('/tmp/hscimproc')


class FrameGenerator:

    def __init__(self,name = 'Unknown Frame Generator'):
        self.aligned_frame_generator = None
        self.brighten = False
        self.name = name

    def raw_frame_generator(self,mraw,start_frame=0,output_dtype=np.uint8,n_frames=None, scale=True,hflip=False,vflip=False,brighten=False):

        # determine the image shape so it doesn't need to be passed
        (im_shape,
        total_frames,
        fps,
        bit_depth, start_offset) = self.get_metadata(mraw)

        self.mmap = open(mraw,'rb')

        self.brighten = brighten

        if n_frames is None:
            n_frames = total_frames - start_frame
        elif n_frames > total_frames - start_frame:
            print('Warning: n_frames is greater than the total number of frames in the file')
            n_frames = total_frames - start_frame

        px_per_frame = np.prod(im_shape)

        output_dtype_itemsize = np.array([0],dtype=output_dtype).itemsize

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
        self.fps = fps
        self.start_offset = start_offset

        self.current_index = start_frame
        self.output_resolution = None
        self.center_offset = (0,0)

        return self

    def get_metadata(self,filename):

        # find the metadatata file
        mraw_folder = os.path.abspath(os.path.relpath(f'{filename}/..'))
        files = os.listdir(mraw_folder)
        files = [f for f in files if f.startswith(filename.split('/')[-1].split('.')[0])]

        xml_file = [f for f in files if f.split('.')[-1].startswith('cih')][0]
        xml_file = os.path.join(mraw_folder,xml_file)
        
        with open(xml_file,'rb') as f:
            xml_data = f.read()
            # Find the start of the XML content and decode it
            start = xml_data.find(b'<')
            # xml_data = xml_data[start:].decode('utf-8')
            xml_data = xml_data[start:]

        # root = xml.etree.ElementTree.fromstring(xml_data)

        root = lxml.etree.fromstring(xml_data)

        imageDataInfo = root.find('imageDataInfo')
        resolution = imageDataInfo.find('resolution')
        resolution = resolution.find('width').text,resolution.find('height').text
        resolution = int(resolution[1]),int(resolution[0])
        total_frames = int(root.find('frameInfo').find('totalFrame').text)
        fps = int(root.find('recordInfo').find('recordRate').text)
        start_offset = int(root.find('frameInfo').find('startFrame').text)
        bitDepth = int(imageDataInfo.find('effectiveBit').find('depth').text)
        f = None
        sleep(0.5)
        
        return resolution, total_frames, fps, bitDepth, start_offset

    def hflip(self,im):
        return np.flip(im,1)
    
    def vflip(self,im):
        return np.flip(im,0)

    def play(self,fps=30,**kwargs) -> None:

        assert fps >= 0

        #Check if opencv is installed
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError('OpenCV is required for FrameGenerator playback')

        print('Press q or esc to stop playback')

        for frame in self:

            # if self.output_resolution is not None:
            #     frame = self.pad_to(frame,self.output_resolution,center_offset=self.center_offset)

            cv.imshow('Player1',frame)

            if fps == 0:
                keypress = cv.waitKey()
            else:
                keypress = cv.waitKey(1000//fps)
                #Also allow user to press 'q' button or 'esc' button and stop the video
            if keypress == ord('q') or keypress == 27:
                break
            if keypress == ord('s'):
                self.save_tiff(frame)


    def to_mp4(self,frames,filename='output.mp4',fps=30,four_cc="MP4V"):

        """
        Converts a list of frames to an mp4 video file

        Parameters:
        frames: generator of frames
        filename: name of the output video file
        fps: frames per second of the output video
        """

        #Check if opencv is installed
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError('OpenCV is required for this function')

        frame = next(frames)

        #Get the shape of the first frame
        shape = frame.shape
        shape = (shape[1],shape[0])

        if len(shape) == 3:
            isColor = True
            shape = (shape[1],shape[0],3)
        else:
            isColor = False

        #Create a video writer object
        fourcc = cv.VideoWriter_fourcc(*four_cc)
        out = cv.VideoWriter(filename,fourcc, fps, shape,isColor=isColor)

        out.write(frame)
        for frame in frames:
            out.write(frame)

        out.release()

    def standard_format_frame_generator(self,pattern,start_frame=0,n_frames=None,sort_fn=lambda x: re.findall('[0-9]+',x)[-1],nostop=False,hflip=False,vflip=False):

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
            raise ValueError('No files found matching pattern',pattern)

        files.sort(key=sort_fn)

        if not n_frames:
            n_frames = np.inf   

        n_frames = min(n_frames,len(files)-start_frame)

        files = files[start_frame:start_frame+n_frames]

        for f in files:
            try:
                with Image.open(f) as im:
                    im = np.array(im)
            
                    if hflip:
                        im = self.hflip(im)
                    if vflip:
                        im = self.vflip(im)

                    yield im
            except:
                print('Warning: Could not open file',f)

                if not nostop:
                    print('Stopping due to bad image file read. To ignore bad file reads, pass nostop=True')
                    break

        return self
            
    def __iter__(self):
        self.current_index = 0
        return self
    
    def get_previous(self):
        if self.current_index > self.start_frame:
            self.current_index -= 1
        return next(self)

    def get_frame(self,n_frame):

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
        data = np.frombuffer(data,dtype=np.uint8)

        if bit_depth == 12:
            
            fst_uint8, mid_uint8, lst_uint8 = data.reshape((-1,3)).astype(np.uint16).T
            
            fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
            
            snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
            
            mat =  np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), im_shape)

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

        if self.output_resolution is not None:
            mat = self.pad_to(mat,resolution=self.output_resolution,center_offset=self.center_offset)

        if self.brighten:
            _max = mat.max()
            mat = mat * (2**(output_dtype_itemsize*8) / _max)
            mat = mat.astype(output_dtype) 

        return mat
    
    def get_current_frame(self):
        return self.get_frame(self.current_index)
    
    def __next__(self):
        frame =  self.get_frame(self.current_index)
        self.current_index += 1

        return frame
    
    def pad(self,frame,pad):
        # follows (before, after) format. For both axes, use ( (before_x,after_x),(before_y,after_y) )
        return np.pad(frame,pad)
    
    def pad_to(self,frame,resolution,center_offset=(0,0)):

        # uses numpy coordinates so it goes (top,bottom), (left,right)
        # center_offset uses normal xy coordinates (since that is what camera reports)

        assert len(resolution) == 2

        hor_res, vert_res = resolution
        vert_len, hor_len = frame.shape[:2]

        hor_center = hor_res // 2 + center_offset[0]
        vert_center = vert_res // 2 + center_offset[1] 

        assert vert_res >= vert_len
        assert hor_res >= hor_len

        left_pad = hor_center - hor_len // 2
        right_pad = hor_res - hor_center - hor_len // 2
        bottom_pad = vert_center - vert_len // 2
        top_pad = vert_res - vert_center - vert_len // 2

        return np.pad(frame,((top_pad,bottom_pad),(left_pad,right_pad)))
    
    def save_tiff(self,frame):
        import cv2 as cv

        files = glob.glob(f'/tmp/hscimproc/{self.name.replace(" ","_")}*')
        cv.imwrite(f'/tmp/hscimproc/{self.name.replace(" ","_")}{len(files)}.tiff',frame)

    def set_output_resolution(self,resolution):
        self.output_resolution = resolution

    def set_center_offset(self,center_offset):
        self.center_offset = center_offset

    def to_tmp_file(self,frame: np.array):
        """
        Take a numpy array/image, save to temporary folder and return the filename
        """
        hashname = f'/tmp/hscimproc/{hashlib.sha1(frame.tobytes()).hexdigest()}.tiff'
        cv.imwrite(hashname,frame)
        return hashname

    def __del__(self):
        if hasattr(self,'mmap'):
            self.mmap.close()

class AlignedFrameGenerator(FrameGenerator):
    def __init__(self,parent,name='Unknown Frame Generator'):
        self.parent = parent
        super().__init__(name=name)

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
            return super().get_frame(i,**kwargs)
        else:
            return None
        
class FrameGeneratorCollection:

    def __init__(self,frame_generator: FrameGenerator,aligned_frame_generator: AlignedFrameGenerator):

        self.frame_generator = frame_generator
        self.aligned_frame_generator = aligned_frame_generator
        
    def play(self,fps=30):

        assert fps >= 0

        try:
            import cv2 as cv
        except ImportError:
            raise ImportError("You need OpenCV to play back a frame generator")

        if fps == 0:
            delay = 0
        else:
            delay = 1000 // fps

        for (frame1,frame2) in self:

            cv.imshow('Player1',frame1)
            if frame2 is not None:
                cv.imshow('Player2',frame2)

            key = cv.waitKey(delay)

            if key == ord('q'):
                break

    def __iter__(self):
        return self
    
    def __next__(self):
        idx = self.frame_generator.current_index + 1
        self.frame_generator.current_index = idx
        frame1 = self.frame_generator.get_frame(idx)
        frame2 = self.aligned_frame_generator.get_frame(idx)
        return (frame1, frame2)

if __name__ == '__main__':
    fg = FrameGenerator(name='East')
    fg.raw_frame_generator('/srv/smb/12dec/run1946_schlieren_C001H001S0001/run1946_schlieren_C001H001S0001.mraw',hflip=True)
    fg.set_output_resolution((1024,1024))
    fg.set_center_offset((0,0))

    fg2 = AlignedFrameGenerator(fg,name='Top')
    fg2.raw_frame_generator('/srv/smb/12dec/run1946_top_C002H001S0001/run1946_top_C002H001S0001.mraw',hflip=True,vflip=True)
    fg2.set_output_resolution((1024,1024))
    fg2.brighten = True

    fg_collection = FrameGeneratorCollection(fg,fg2)

    fg_collection.play(0)

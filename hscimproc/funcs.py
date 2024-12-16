import numpy as np
import xml.etree.ElementTree
import lxml.etree
import os
import glob
import re
from PIL import Image
from time import sleep

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

class FrameGenerator:

    def __init__(self):
        pass

    def raw_frame_generator_int16(self,mraw,start_frame=0,n_frames=None, scale=False,hflip=False,vflip=False):

        # determine the image shape so it doesn't need to be passed
        (im_shape,
        total_frames,
        fps,
        bit_depth) = self.get_metadata(mraw)


        if n_frames is None:
            n_frames = total_frames - start_frame
        elif n_frames > total_frames - start_frame:
            print('Warning: n_frames is greater than the total number of frames in the file')
            n_frames = total_frames - start_frame

        px_per_frame = np.prod(im_shape)

        # return a generator of image frames
        for n_frame in range(start_frame,start_frame+n_frames):

            frame_offset_bytes = int(bit_depth/8 * n_frame * px_per_frame)
            if bit_depth == 12:
                data = np.memmap(mode='r',filename=mraw,dtype=np.uint8,offset=frame_offset_bytes,shape=(int(px_per_frame*bit_depth/8),)) 
                self.data = data
                fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
                fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
                snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
                mat =  np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), im_shape)
            elif bit_depth == 8:
                mat = np.memmap(mode='r',filename=mraw,dtype=np.uint8,offset=frame_offset_bytes,shape=im_shape)
            else:
                mat = np.memmap(mode='r',filename=mraw,dtype=np.uint16,offset=frame_offset_bytes,shape=im_shape)
            matcpy = mat.copy()

            if bit_depth == 12 and scale:
                matcpy <<= 4

            if hflip:
                matcpy = self.hflip(matcpy)
            if vflip:
                matcpy= self.vflip(matcpy)

            yield matcpy

    def raw_frame_generator_int8(self,mraw,start_frame=0,n_frames=None,hflip=False,vflip=False):

        # create a 16 bit generator and bit shift down

        # determine the image shape so it doesn't need to be passed
        (im_shape,
        total_frames,
        fps,
        bit_depth) = self.get_metadata(mraw)

        bitshift = bit_depth - 8

        frame_gen = self.raw_frame_generator_int16(mraw,start_frame=start_frame,n_frames=n_frames,scale=False,hflip=hflip,vflip=vflip)

        for frame in frame_gen:
            frame = (frame >> bitshift).astype(np.uint8)
            self.frame = frame
            yield(frame)

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
        bitDepth = int(imageDataInfo.find('effectiveBit').find('depth').text)
        f = None
        sleep(0.5)
        
        return resolution, total_frames, fps, bitDepth

    def hflip(self,im):
        return np.flip(im,1)
    
    def vflip(self,im):
        return np.flip(im,0)

    def play(self,frames,fps=30,**kwargs) -> None:

        assert fps >= 0

        #Check if opencv is installed
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError('OpenCV is required for this function')

        print('Press q or esc to stop playback')

        for frame in frames:
            cv.imshow('Playback',frame)
            if fps == 0:
                keypress = cv.waitKey()
            else:
                keypress = cv.waitKey(1000//fps)
                #Also allow user to press 'q' button or 'esc' button and stop the video
            if keypress == ord('q') or keypress == 27:
                break



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
            


    def mraw_one_frame(self,mraw,n):
        return next(self.raw_frame_generator_int8(mraw,n,1))

    def one_frame(self,fname,n):
        return next(self.standard_format_frame_generator(fname,n,1))

#%% 
import cv2 as cv
import os
import numpy as np

def identify_blob():
    


im = cv.imread('/tmp/blender_tracking/TOP_render.png',cv.IMREAD_GRAYSCALE)
# sift = cv.SIFT.create()
# sift = cv.ORB().create()
# kp,desc = sift.detectAndCompute(im,None)

im2 = cv.imread('/tmp/blender_tracking/screeny.png',cv.IMREAD_GRAYSCALE)
# kp2,desc2 = sift.detectAndCompute(im2,None)

# bfmatcher = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
# matches = bfmatcher.match(desc,desc2)
# matches = sorted(matches, key = lambda x:x.distance)

# # Define a threshold for filtering good matches
# # (Lower distance indicates better match)
# threshold = 0.75 * np.median([m.distance for m in matches])
# # Filter matches based on the threshold
# good_matches = [m for m in matches if m.distance < threshold]

im1rgb = cv.cvtColor(im,cv.COLOR_GRAY2BGR)
im2rgb = cv.cvtColor(im2,cv.COLOR_GRAY2BGR)


# cv.drawKeypoints(im1rgb,kp,im1rgb,(0,255,0))
# cv.drawKeypoints(im2rgb,kp2,im2rgb,(0,255,0))

# img3 = cv.drawMatches(im,kp,im2,kp2,good_matches,im2rgb,(0,0,255),flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# # cv.imshow('2',im2rgb)
# # cv.imshow('1',im1rgb)
# cv.imshow('3',img3)

# cv.waitKey()
# cv.destroyAllWindows()




_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
dparams = cv.aruco.DetectorParameters()
dparams.useAruco3Detection = True
dparams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
det = cv.aruco.ArucoDetector(_dict,dparams)

cnrs, ids, rej = det.detectMarkers(im2)

# cnrs = np.vstack(cnrs)
# ids = np.array(ids)
# im1rgb = cv.cvtColor(im2,cv.COLOR_GRAY2BGR)
cv.aruco.drawDetectedMarkers(im2rgb,cnrs,ids,(0,0,255))
# cv.imshow('',im1rgb)
# cv.waitKey()
# cv.destroyAllWindows()

bd = cv.SimpleBlobDetector().create()
bd.Params.maxArea = 100
bd.Params.minArea = 4
# bd.Params.filterByConvexity = True

ret, threshim = cv.threshold(im2,128,255,cv.THRESH_OTSU+cv.THRESH_BINARY)

resp = bd.detect(threshim)

cv.drawKeypoints(im2rgb,resp,im2rgb,(0,0,255))
cv.imshow('',im2rgb)
cv.waitKey()
cv.destroyAllWindows()

# %%

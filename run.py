import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt # for plotting

# COMPARE_METHODS = (
#     ('Correlation', cv2.cv.CV_COMP_CORREL),
#     ('Chi-Square', cv2.cv.CV_COMP_CHISQR),
#     ('Intersection', cv2.cv.CV_COMP_INTERSECT),
#     ('Hellinger', cv2.cv.CV_COMP_BHATTACHARYYA)
# )

COMPARE_METHODS = (
    ('Correlation', cv2.cv.CV_COMP_CORREL),
)


def load_video(path):
    MAX_LONG_SIZE = 400
    resize = 1
    cap = cv2.VideoCapture()
    cap.open(path)

    if not cap.isOpened():
        print "Fatal error - could not open video %s." % (path)
        sys.exit()
    else:
        print "Parsing video %s ..." % (path)
        width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        print "Video Resolution (width x height): %d x %d" % (width, height)
        long_side = max(width,height)
        if long_side > MAX_LONG_SIZE:
            resize = math.pow(2,long_side/MAX_LONG_SIZE)
        resize_width = int(width/resize)
        resize_height = int(height/resize)
    return cap, resize_width, resize_height


def histogram(frame):
    ## calcHist: calculate histogram for B,G,R channels in range[0,256] and 256 bins without mask
    # hist = cv2.calcHist([frame], [0,1,2], None, [256, 256, 256], [0,256,0,256,0,256])
    b = cv2.calcHist([frame], [0], None, [5], [0,256])
    g = cv2.calcHist([frame], [1], None, [5], [0,256])
    r = cv2.calcHist([frame], [2], None, [5], [0,256])
    hist = (cv2.normalize(b),cv2.normalize(g),cv2.normalize(r))

    return hist


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4 / colour blue / thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1 / colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


def SIFT(img1, img2):
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    return kp1, des1, kp2, des2


def SURF(img1, img2):
    # Create SURF object
    # Here I set Hessian Threshold to 400
    surf = cv2.SURF(1000)

    # Find keypoints and descriptors directly
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    return kp1, des1, kp2, des2


def FREAK(img1, img2):
    # Create FREAK object
    freak = cv2.FREAK()

    # Find keypoints and descriptors directly
    kp1, des1 = freak.detectAndCompute(img1,None)
    kp2, des2 = sufreakrf.detectAndCompute(img2,None)

    return kp1, des1, kp2, des2

def BRIEF(img1, img2):
    # Initiate STAR detector
    star = cv2.FeatureDetector_create("STAR")

    # Initiate BRIEF extractor
    brief = cv2.DescriptorExtractor_create("BRIEF")

    # find the keypoints with STAR
    kp1 = star.detect(img1,None)
    kp2 = star.detect(img2,None)

    # compute the descriptors with BRIEF
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    des1 = np.asarray(des1,np.float32)
    des2 = np.asarray(des2,np.float32)

    return kp1, des1, kp2, des2


def homography(good,kp1,kp2):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    kp1_t = cv2.perspectiveTransform(src_pts,M)

    return ((kp1_t-dst_pts)**2).sum()/len(good)

def match(img1, img2):
    ## img1: queryImage, img2: trainImage
    MIN_MATCH_COUNT = 20
    MAX_DIST = 10

    # kp1, des1, kp2, des2 = SIFT(img1, img2)
    kp1, des1, kp2, des2 = SURF(img1, img2)
    # kp1, des1, kp2, des2 = FREAK(img1, img2)
    # kp1, des1, kp2, des2 = BRIEF(img1, img2)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        dist = homography(good,kp1,kp2)
        if dist > MAX_DIST:
            print "Distance too large - %d (pixel)" % (dist)
            return False

    else:
        img3 = drawMatches(img1,kp1,img2,kp2,good[:10])
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        return False

    return True


def main():
    if len(sys.argv) < 2:
        print "Error - file name must be specified as first argument."
        return

    cap, resize_width, resize_height = load_video(sys.argv[1])
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print 'fps:', fps
    # sample = int(fps/15) + 1
    index = 0
    while True:
        (rv1, im1) = cap.read()   # im is a valid image if and only if rv is true
        (rv2, im2) = cap.read()
        index = index + 1
        ## sampling
        # if index % sample != 0:
        #     continue
        if not rv1 or not rv2 or index == 1000:
            break
        im1 = cv2.resize(im1,(resize_width,resize_height))
        im2 = cv2.resize(im2,(resize_width,resize_height))

        feature_match = match(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))

        hist1 = histogram(im1)
        hist2 = histogram(im2)
        for name, method in COMPARE_METHODS:
            diff_b = cv2.compareHist(hist1[0], hist2[0], method)
            diff_g = cv2.compareHist(hist1[1], hist2[1], method)
            diff_r = cv2.compareHist(hist1[2], hist2[2], method)
            diff = diff_b*diff_b + diff_g*diff_g + diff_r*diff_r
            print 'hist', diff

        if not feature_match and diff < 2.5 :
            print index
            cv2.imshow('frame %s' % (index-1), im1)
            cv2.imshow('frame %s' % (index), im2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    main()

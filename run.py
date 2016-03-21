import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt # for plotting

sys.path.append('/usr/local/lib/python2.7/site-packages')

# COMPARE_METHODS = (
#     ('Correlation', cv2.cv.CV_COMP_CORREL),
#     ('Chi-Square', cv2.cv.CV_COMP_CHISQR),
#     ('Intersection', cv2.cv.CV_COMP_INTERSECT),
#     ('Hellinger', cv2.cv.CV_COMP_BHATTACHARYYA)
# )


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
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        print "Video Resolution (width x height): %d x %d" % (width, height)
        long_side = max(width,height)
        if long_side > MAX_LONG_SIZE:
            resize = math.pow(2,long_side/MAX_LONG_SIZE)
        resize_width = int(width/resize)
        resize_height = int(height/resize)
    return cap, width, height, resize_width, resize_height


def division(frame):
    #TODO: divide each frame into 4x4 windows
    length, width, ch = frame.shape
    blocks = [frame[:length/2, :width/2], frame[length/2:, :width/2], frame[:length/2, width/2:], frame[length/2:, width/2:]]
    return blocks


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
    MIN_MATCH_COUNT = 4
    MAX_DIST = 20

    # kp1, des1, kp2, des2 = SIFT(img1, img2)
    kp1, des1, kp2, des2 = SURF(img1, img2)
    # kp1, des1, kp2, des2 = FREAK(img1, img2)
    # kp1, des1, kp2, des2 = BRIEF(img1, img2)

    good = []
    if len(kp1) >= 2 and len(kp2) >= 2:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # print good[0].distance
        good_sort = sorted(good, key=lambda x : x.distance)
        # print good_sort[0].distance
        dist = homography(good_sort[0:MIN_MATCH_COUNT],kp1,kp2)
        if dist > MAX_DIST:
            print "Distance too large - %d (pixel)" % (dist)
            return False

    else:
        img3 = drawMatches(img1,kp1,img2,kp2,good[:10])
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
        return False

    return True


def save2png(filename, img):
    cv2.imwrite(filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def save2video(filename, frames, fps, frame_size):

    # out = cv2.VideoWriter('%s.mpg' % (filename), cv2.cv.CV_FOURCC('P','I','M','1'), fps, frame_size)
    out = cv2.VideoWriter('%s.mp4' % (filename), cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), fps, frame_size)
    if len(frames) < fps:
        save2png('%s.png' % (filename), frames[0])
    else:
        for frame in frames:
            out.write(frame)
    out.release()


def main():
    if len(sys.argv) < 2:
        print "Error - file name must be specified as first argument."
        return

    cap, width, height, resize_width, resize_height = load_video(sys.argv[1])
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print 'fps:', fps
    # sample = int(fps/15) + 1

    # shots = [0]
    num_shot = 0
    # buff = []
    index = 0
    while True:
        (rv1, im1) = cap.read()   # im is a valid image if and only if rv is true
        (rv2, im2) = cap.read()
        index = index + 1
        ## sampling
        # if index % sample != 0:
        #     continue
        if not rv1 or not rv2:
            break
        # buff.append(im2)
        im1 = cv2.resize(im1,(resize_width,resize_height))
        im2 = cv2.resize(im2,(resize_width,resize_height))

        blocks1 = division(im1)
        blocks2 = division(im2)
        feature_match = match(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
        largest_diff = 0
        sum_diff = 0
        for i in range(4):
            hist1 = histogram(blocks1[i])
            hist2 = histogram(blocks2[i])
            diff_r = cv2.compareHist(hist1[0], hist2[0], cv2.cv.CV_COMP_CHISQR)
            diff_g = cv2.compareHist(hist1[1], hist2[1], cv2.cv.CV_COMP_CHISQR)
            diff_b = cv2.compareHist(hist1[2], hist2[2], cv2.cv.CV_COMP_CHISQR)
            diff = diff_b + diff_g + diff_r
            if diff > largest_diff:
                largest_diff = diff
            sum_diff += diff
        sum_diff -= largest_diff

        if not feature_match and sum_diff > 10:
            print index
            print 'sum_diff', sum_diff
            # shots.append(index)
            save2png('frame_%s.png' % (index-1), im1)
            save2png('frame_%s.png' % (index), im2)
            # save2video('shot_%s' % (num_shot), buff, fps, (width, height))
            num_shot += 1
            # buff = []

            # cv2.imshow('frame %s' % (index-1), im1)
            # cv2.imshow('frame %s' % (index), im2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # cap.release()
    # shots.append(index)

    # print "Video total length: %d frame unit turned into %d shots" % (index, num_shot)
    # for i in range(len(shots)-1):
        # print "shot %s: %d frame unit" % (i, shots[i+1]-shots[i])

if __name__ == "__main__":
    main()

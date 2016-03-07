import sys
import cv2
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
    cap = cv2.VideoCapture()
    cap.open(path)

    if not cap.isOpened():
        print "Fatal error - could not open video %s." % (path)
        sys.exit()
    else:
        print "Parsing video %s ..." % (path)
    return cap


def video2frames(cap):
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    print "Video Resolution (width x height): %d x %d" % (width, height)

    frames = []
    while True:
        (rv, im) = cap.read()   # im is a valid image if and only if rv is true
        if not rv or len(frames) == 1000:
            break
        frames.append(im)
        # Do stuff with im here.
    print "Read %d frames from video." % len(frames)
    return frames


def histogram(frame):
    ## calcHist: calculate histogram for B,G,R channels in range[0,256] and 256 bins without mask
    # hist = cv2.calcHist([frame], [0,1,2], None, [256, 256, 256], [0,256,0,256,0,256])
    hist = cv2.calcHist([frame], [0], None, [5], [0,256])

    # color = ('b', 'g', 'r')
    # for ch, col in enumerate(color):
        ## calcHist: calculate histogram for B,G,R channels in range[0,256] without mask
        # hist = cv2.calcHist([frame], [ch], None, [256], [0,256])
        ## plot histogram
        # plt.plot(hist, color=col)
        # plt.xlim([0,256])  # x-axis
    # plt.show()

    return hist


def main():
    if len(sys.argv) < 2:
        print "Error - file name must be specified as first argument."
        return

    cap = load_video(sys.argv[1])
    # Do stuff with cap here.
    frames = video2frames(cap)
    # hist = histogram(frames[0])
    for i in range(len(frames)-1):
        hist1 = histogram(frames[i])
        hist2 = histogram(frames[i+1])
        for name, method in COMPARE_METHODS:
            diff = cv2.compareHist(hist1, hist2, method)
            print diff

            if diff < 0.8:
                cv2.imshow('frame %s' % (i), frames[i])
                cv2.imshow('frame %s' % (i+1), frames[i+1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    cap.release()


if __name__ == "__main__":
    main()

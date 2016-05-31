import sys, glob, argparse
import numpy as np
from scipy.spatial.distance import pdist
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm

SAMPLE_RATE = 20


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
        long_side = max(width, height)
        if long_side > MAX_LONG_SIZE:
            resize = math.pow(2, long_side/MAX_LONG_SIZE)
        resize_width = int(width/resize)
        resize_height = int(height/resize)
        total_frame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        print 'total_frame', total_frame
    return cap, resize_width, resize_height


def load_frames(file):
    cap, resize_width, resize_height = load_video(file)
    total_frame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    index = 0
    raw_frames = []
    while True:
        rv, im = cap.read()
        if not rv or index > total_frame:
            break
        index = index + 1
        raw_frames.append(im)
    frames = [cv2.resize(raw_frames[i], (resize_width, resize_height)) for i in range(0, len(raw_frames), SAMPLE_RATE)]
    print 'sample', len(frames), 'from', len(raw_frames)
    print 'raw', raw_frames[0].shape, 'resize', frames[0].shape
    return raw_frames, frames, cap


def dictionary(des, N):
    print 'start em training'
    em = cv2.EM(N)
    em.train(des)

    return np.float32(em.getMat("means")), \
        np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]


def PCA(des):
    mean, eigenvectors = cv2.PCACompute(des, maxComponents=64)
    compressed_des = cv2.PCAProject(des, mean, eigenvectors)
    return compressed_des


def image_descriptors(file):
    # img = cv2.imread(file, 0)
    # img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((1, 128))

    ## Reduce dimension with PCA
    # des = PCA(des)
    return des


def video_descriptors(frames):
    print("Calculating descriptors. # of frames: ", len(frames))
    return np.concatenate([image_descriptors(frame) for frame in frames])


def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk


def likelihood_statistics(samples, means, covs, weights):
    gaussians, s0, s1, s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)

    g = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True) for k in range(0, len(weights))]
    for index, x in samples:
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2


def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k] + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k]) for k in range(0, len(w))])


def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))


def fisher_vector(samples, means, covs, w):
    s0, s1, s2 = likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv


def generate_gmm(frames, N, folder=""):
    words = video_descriptors(frames)
    print("Training GMM of size", N)
    means, covs, weights = dictionary(words, N)
    ## Throw away gaussians with weights that are too small:
    th = 1.0 / N
    means = np.float32([m for k, m in zip(range(0, len(weights)), means) if weights[k] > th])
    covs = np.float32([m for k, m in zip(range(0, len(weights)), covs) if weights[k] > th])
    weights = np.float32([m for k, m in zip(range(0, len(weights)), weights) if weights[k] > th])

    print 'generate gmm'
    np.save("%s/means.gmm" % folder, means)
    np.save("%s/covs.gmm" % folder, covs)
    np.save("%s/weights.gmm" % folder, weights)
    print 'gmm saved'
    return means, covs, weights


def fisher_features(frames, gmm):
    features = np.float32([fisher_vector(image_descriptors(frame), *gmm) for frame in frames])
    return features


def load_gmm(folder=""):
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    return map(lambda file: np.load(file), map(lambda s: folder + "/" + s, files))


def kernel_matrix(features):
    kernel_matrix = pdist(features, 'cosine')
    return kernel_matrix


def similarity_matrix(features):
    similarity = []
    for i in range(len(features)-1):
        similarity.append(pdist([features[i], features[i+1]]))
    similarity = np.matrix(similarity)
    print similarity.shape
    return similarity


def merge(raw_frames, similarity, cap, k=10, e=False):
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    cluster = similarity.shape[0] + 1
    while cluster > k:
        argmin = np.argmin(similarity)
        similarity[argmin] = 100
        cluster = cluster - 1
    print '# of clips:', cluster
    num_shot = 0
    i = 0
    buff = []
    if e:
        save2png('frame_%s.png' % (i*SAMPLE_RATE), raw_frames[min(i*SAMPLE_RATE,len(raw_frames)-1)])
    while i != similarity.shape[0] - 1:
        if similarity[i] == 100:
            if not e:
                for idx in range(SAMPLE_RATE):
                    buff.append(raw_frames[min(i*SAMPLE_RATE+idx,len(raw_frames)-1)])
        else:
            if e:
                save2png('frame_%s.png' % ((i+1)*SAMPLE_RATE), raw_frames[min((i+1)*SAMPLE_RATE,len(raw_frames)-1)])
            else:
                for idx in range(SAMPLE_RATE):
                    buff.append(raw_frames[min(i*SAMPLE_RATE+idx,len(raw_frames)-1)])
                save2video('shot_%s' % (num_shot), buff, fps, (width, height))
                num_shot = num_shot + 1
                buff = []
        i = i + 1
    if e:
        save2png('frame_%s.png' % (i*SAMPLE_RATE), raw_frames[min(i*SAMPLE_RATE,len(raw_frames)-1)])
    else:
        save2video('shot_%s' % (num_shot), buff, fps, (width, height))
        num_shot = num_shot + 1


def save2png(filename, img):
    cv2.imwrite(filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def save2video(filename, frames, fps, frame_size):
    if len(frames) < fps:
        pass
        # print '%s.png' % (filename)
        # save2png('%s.png' % (filename), frames[0])
    else:
        print '%s.mp4' % (filename)

        out = cv2.VideoWriter('%s.mp4' % (filename), cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), fps, frame_size)
        for frame in frames:
            out.write(frame)
        out.release()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--video", help="Path to the video file", default='test.mp4')
    parser.add_argument('-f', "--folder", help="Working folder", default='.')
    parser.add_argument('-g', "--loadgmm", help="Load Gmm dictionary", action='store_true', default=False)
    parser.add_argument('-n', "--number", help="Number of words in dictionary", default=128, type=int)
    parser.add_argument('-k', "--k", help="Number of clips wanted", default=5)
    parser.add_argument('-e', "--eval", help="Eval mode, turn on to save the boundary frames only", default=False)
    args = parser.parse_args()
    return args



args = get_args()
raw_frames, frames, cap = load_frames(args.video)
gmm = load_gmm(args.folder) if args.loadgmm else generate_gmm(frames, args.number, args.folder)
fisher_features = fisher_features(frames, gmm)
similarity = similarity_matrix(fisher_features)
merge(raw_frames, similarity, cap, int(args.k), args.eval)

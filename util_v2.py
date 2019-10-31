from __future__ import print_function
import cv2 as cv
import numpy as np
from pil import Image
import pil
import torchvision
import statistics
import torch
from Network import UNet
import argparse
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
from torchvision import transforms
import cv2
#import cv2

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def faster_rcnn(img):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image = pil.Image.open(img)
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    output = model([image_tensor])
    new = output[0]['boxes']
    new = new[output[0]['scores'] > 0.4]
    label = output[0]['labels']
    label = label[output[0]['scores'] > 0.4]
    new = new[label == 1]
    #print(new)
    return new

def mask_rcnn_v2(img):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image = pil.Image.open(img)
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    output = model([image_tensor])
    new = output[0]['masks'][:, 0]
    new = new[output[0]['scores'] > 0.4]
    label = output[0]['labels']
    label = label[output[0]['scores'] > 0.4]
    new = new[label == 1]
    msk = np.sum(new.mul(255).byte().cpu().numpy(), axis=0)
    msk[msk > 0] = 254
    cv2.imwrite('test_vahid.jpg', np.array(msk, dtype=np.uint8))
    m = np.nonzero(msk)
    lst = []
    for i in range(len(m[0])):
        lst.append([m[0][i], m[1][i]])
    return lst

def mask_rcnn(img):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    image = pil.Image.open(img)
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    output = model([image_tensor])
    new = output[0]['masks'][:, 0]
    new = new[output[0]['scores'] > 0.4]
    label = output[0]['labels']
    label = label[output[0]['scores'] > 0.4]
    new = new[label == 1]
    msk = np.sum(new.mul(255).byte().cpu().numpy(), axis=0)
    msk[msk > 0] = 254
    return msk

def parameterization(j, image_lst, parameter, Min_list):
    lst = []
    A = homography_matrix(image_lst[j-1], image_lst[j])
    for x in range(len(parameter)):
        C = A.dot(np.array([int(parameter[x]), Min_list[int(parameter[x])], 1]))
        if C[0]/C[2] > 0 and C[0]/C[2]< len(Min_list):
            lst.append(C[0]/C[2])
    return lst


def homography_matrix(img1, img2):
    #point = []
    im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    #orb = cv.ORB_create(MAX_FEATURES)
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=2000000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)
    msk = mask_rcnn(img1)
    p1 = []
    p2 = []
    test = []
    for i, match in enumerate(matches):
        if msk[int(keypoints1[match.queryIdx].pt[1]), int(keypoints1[match.queryIdx].pt[0])] == 0 and msk[int(keypoints1[match.queryIdx].pt[1])+1, int(keypoints1[match.queryIdx].pt[0])+1] == 0:
            # points1[i, :] = keypoints1[match.queryIdx].pt
            # points2[i, :] = keypoints2[match.trainIdx].pt
            p1.append([keypoints1[match.queryIdx].pt[1],keypoints1[match.queryIdx].pt[0]])
            p2.append([keypoints2[match.trainIdx].pt[1],keypoints2[match.trainIdx].pt[0]])
        else:
            test.append([keypoints1[match.queryIdx].pt[1],keypoints1[match.queryIdx].pt[0]])

        #print(points1[i][0] - points2[i][0])
    #print(len(p1))
    points1 = np.asarray(p1, dtype=np.float32)
    points2 = np.asarray(p2, dtype=np.float32)
    # for j in range(len(p1)):
    #     im1[int(p1[j][0]),int(p1[j][1])]=[0,0,255]
    #
    # cv.imwrite("E://Frames/test_alignment/test4.jpg", im1)
    # exit(0)
    # Find homography
    matrix, mask = cv.findHomography(points1, points2, cv.RANSAC)
    a = np.array(matrix)
    # print("works")
    # exit(0)
    return a

def homography_matrix_v2(img1, img2):
    #point = []
    im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    #orb = cv.ORB_create(MAX_FEATURES)
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=2000000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        # print(keypoints1[match.queryIdx].pt[0])
        # exit(0)
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        #print(points1[i][0] - points2[i][0])
    # Find homography
    matrix, mask = cv.findHomography(points1, points2, cv.RANSAC)
    #print('Hello')
    a = np.array(matrix)
    return a

def billboard_point_tracker_homography(xx, yy, img1, img2):
    #point = []
    im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    #orb = cv.ORB_create(MAX_FEATURES)
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=2000000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        #print(points1[i][0] - points2[i][0])
    # Find homography
    matrix, mask = cv.findHomography(points1, points2, cv.RANSAC)
    A = np.array(matrix)
    #B = np.array([[points1[0][0],points1[0][1],1]]).transpose()
    B = np.array([[xx, yy, 1]]).transpose()
    C = A.dot(B)
    # print(C)
    # print(C[0][0], C[1][0])
    # exit(0)
    #point.append(C[0][0])
    # print(points2)
    # print(C)
    # exit(0)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, matrix, (width, height))

    return C[0][0]/C[2][0], C[1][0]/C[2][0], C[2][0]

def static_point_detecor_homography(img1, img2):
    #point = []
    im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    #orb = cv.ORB_create(MAX_FEATURES)
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=2000000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        #print(points1[i][0] - points2[i][0])
    # Find homography
    matrix, mask = cv.findHomography(points1, points2, cv.RANSAC)
    A = np.array(matrix)
    #B = np.array([[points1[0][0],points1[0][1],1]]).transpose()
    B = np.array([[647, 553, 1]]).transpose()
    C = A.dot(B)
    # print(C)
    # print(C[0][0], C[1][0])
    # exit(0)
    #point.append(C[0][0])
    # print(points2)
    # print(C)
    # exit(0)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, matrix, (width, height))

    return C[0][0]/C[2][0], C[1][0]/C[2][0], C[2][0]


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        #print(points1[i][0] - points2[i][0])
    # Find homography
    matrix, mask = cv.findHomography(points1, points2, cv.RANSAC)
    # A = np.array(matrix)
    # B = np.array([[points1[0][0],points1[0][1],1]]).transpose()
    # C = A.dot(B)
    # print(points2)
    # print(C)
    # exit(0)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, matrix, (width, height))

    return points1, points2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='CP1_v3.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'CP1_v3.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.25)

    return parser.parse_args()
def predict_img(net,
                full_img,
                scale_factor=0.25,
                out_threshold=0.1,
                use_dense_crf=True,
                use_gpu=True):
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    if img.shape[2] == 4:
        img = img[:,:,0:3]

    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)

    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )

        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)
    #
    return full_mask > out_threshold
def frame_difference_tracker():
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture("video_second_set.avi")
    # Variable for color to draw optical flow track
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    args = get_args()
    net = UNet(n_channels=3, n_classes=1)
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")
    vidcap = cv.VideoCapture("video_second_set.avi")
    success, image = vidcap.read()
    cv.imwrite("frame%d.jpg" % 0, image)
    img = Image.open("frame0.jpg")
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       use_dense_crf=not args.no_crf,
                       use_gpu=not args.cpu)
    numpy_image = np.array(img)
    opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
    min_list = []
    max_list = []
    D_list = []
    M_list = []
    for x in range(0, opencv_image.shape[1]):  # looping through each column
        lst = []
        for y in range(0, opencv_image.shape[0]):  # looping through each rows
            if mask[y, x] > 0:
                lst.append(y)
        if np.any(mask[:, x] > 0):
            min_list.append(min(lst))
            max_list.append(max(lst))
            # print(max_list[x], min_list[x])
            ind = max_list.index(max(lst))
            if max_list[ind] < min_list[ind]:
                k = min_list[ind]
                min_list[ind] = max_list[ind]
                max_list[ind] = k
            D = max_list[ind] - min_list[ind]
            D_list.append(D)
            if D > 2 / 3 * statistics.mean(D_list) and D < 4 / 3 * statistics.mean(D_list):
                T = (max_list[ind] + min_list[ind]) / 2
                M_list.append(T)
    # Mean_D = statistics.mean(D_list)
    im_dst = opencv_image
    min_list = []
    max_list = []
    D_list = []
    Mean_list = []
    # Four corners of the billboard in destination image.
    for x in range(0, opencv_image.shape[1]):  # looping through each column
        lst = []
        for y in range(0, opencv_image.shape[0]):  # looping through each rows
            if mask[y, x] > 0:
                lst.append(y)
        if np.any(mask[:, x] > 0):
            max_pt = (x, max(lst))
            min_pt = (x, min(lst))
            # print(max_pt, min_pt)
            min_list.append(min(lst))
            max_list.append(max(lst))
            # print(max_list[x], min_list[x])
            ind = max_list.index(max(lst))
            if max_list[ind] < min_list[ind]:
                k = min_list[ind]
                min_list[ind] = max_list[ind]
                max_list[ind] = k
            D = max_list[ind] - min_list[ind]
            D_list.append(D)
            if D > 2 / 3 * statistics.mean(D_list) and D < 4 / 3 * statistics.mean(D_list):
                T = (max_list[ind] + min_list[ind]) / 2
                Mean_list.append(T)

            if D != statistics.mean(D_list) and len(Mean_list) < len(M_list):
                D = statistics.mean(D_list)
                # T = Mean_list[-1]
                X = len(Mean_list)
                # print(X)
                # print(len(M_list))
                T = (M_list[X] + Mean_list[-1]) / 2
                # T = (max_list[ind] + min_list[ind])/2
                max_list[ind] = T + D / 2
                min_list[ind] = T - D / 2
            D1 = 0.1 * D
            max_list[ind] = max_list[ind] - D1
            min_list[ind] = min_list[ind] + 1.5 * D1
    length = len(max_list) - 1
    ind = length
    D = max_list[ind] - min_list[ind]
    if D != statistics.mean(D_list) and len(Mean_list) == len(M_list):
        D_mean = statistics.mean(D_list)
        X = len(Mean_list)
        T = (M_list[X - 1] + Mean_list[X - 2]) / 2
        # T = (max_list[ind] + min_list[ind])/2
        max_list[ind] = T + D_mean / 2
        min_list[ind] = T - D_mean / 2
    D1 = 0.1 * D
    max_list[ind] = max_list[ind] - D1
    min_list[ind] = min_list[ind] + 1.5 * D1
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    length = len(prev)
    # print(length)
    delete_these = []
    for i in range(length):
        # if not( prev[i][0][1]> max_list[int(prev[i][0][0])] or prev[i][0][1] < min_list[int(prev[i][0][0])]):
        if not (prev[i][0][1] < min_list[int(prev[i][0][0])]):
            # print("Hello")
            delete_these.append(i)
    prev = np.delete(prev, delete_these, axis=0)
    # elements in prev should be remained only if their y value is between min_list and max_list

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)

    dif = []
    while (cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        if frame is None:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        # print(next)
        if next is None:
            good_new = good_old
        else:
            good_new = next[status == 1]
        # good_new = next[status == 1]
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        # print(a,b)
        #print(c - a)
        dif.append(c - a)
        output = cv.add(frame, mask)
        # cv.imshow("mask_only", mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        #cv.imshow("sparse optical flow", output)
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    #print(len(dif))
    cap.release()
    cv.destroyAllWindows()
    return dif


def static_point_detector():
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture("video_second_set.avi")
    # Variable for color to draw optical flow track
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    args = get_args()
    net = UNet(n_channels=3, n_classes=1)
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")
    vidcap = cv.VideoCapture("video_second_set.avi")
    success, image = vidcap.read()
    cv.imwrite("frame%d.jpg" % 0, image)
    img = Image.open("frame0.jpg")
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       use_dense_crf=not args.no_crf,
                       use_gpu=not args.cpu)
    numpy_image = np.array(img)
    opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
    min_list = []
    max_list = []
    D_list = []
    M_list = []
    for x in range(0, opencv_image.shape[1]):  # looping through each column
        lst = []
        for y in range(0, opencv_image.shape[0]):  # looping through each rows
            if mask[y, x] > 0:
                lst.append(y)
        if np.any(mask[:, x] > 0):
            min_list.append(min(lst))
            max_list.append(max(lst))
            # print(max_list[x], min_list[x])
            ind = max_list.index(max(lst))
            if max_list[ind] < min_list[ind]:
                k = min_list[ind]
                min_list[ind] = max_list[ind]
                max_list[ind] = k
            D = max_list[ind] - min_list[ind]
            D_list.append(D)
            if D > 2 / 3 * statistics.mean(D_list) and D < 4 / 3 * statistics.mean(D_list):
                T = (max_list[ind] + min_list[ind]) / 2
                M_list.append(T)
    # Mean_D = statistics.mean(D_list)
    im_dst = opencv_image
    min_list = []
    max_list = []
    D_list = []
    Mean_list = []
    # Four corners of the billboard in destination image.
    for x in range(0, opencv_image.shape[1]):  # looping through each column
        lst = []
        for y in range(0, opencv_image.shape[0]):  # looping through each rows
            if mask[y, x] > 0:
                lst.append(y)
        if np.any(mask[:, x] > 0):
            max_pt = (x, max(lst))
            min_pt = (x, min(lst))
            # print(max_pt, min_pt)
            min_list.append(min(lst))
            max_list.append(max(lst))
            # print(max_list[x], min_list[x])
            ind = max_list.index(max(lst))
            if max_list[ind] < min_list[ind]:
                k = min_list[ind]
                min_list[ind] = max_list[ind]
                max_list[ind] = k
            D = max_list[ind] - min_list[ind]
            D_list.append(D)
            if D > 2 / 3 * statistics.mean(D_list) and D < 4 / 3 * statistics.mean(D_list):
                T = (max_list[ind] + min_list[ind]) / 2
                Mean_list.append(T)

            if D != statistics.mean(D_list) and len(Mean_list) < len(M_list):
                D = statistics.mean(D_list)
                # T = Mean_list[-1]
                X = len(Mean_list)
                # print(X)
                # print(len(M_list))
                T = (M_list[X] + Mean_list[-1]) / 2
                # T = (max_list[ind] + min_list[ind])/2
                max_list[ind] = T + D / 2
                min_list[ind] = T - D / 2
            D1 = 0.1 * D
            max_list[ind] = max_list[ind] - D1
            min_list[ind] = min_list[ind] + 1.5 * D1
    length = len(max_list) - 1
    ind = length
    D = max_list[ind] - min_list[ind]
    if D != statistics.mean(D_list) and len(Mean_list) == len(M_list):
        D_mean = statistics.mean(D_list)
        X = len(Mean_list)
        T = (M_list[X - 1] + Mean_list[X - 2]) / 2
        # T = (max_list[ind] + min_list[ind])/2
        max_list[ind] = T + D_mean / 2
        min_list[ind] = T - D_mean / 2
    D1 = 0.1 * D
    max_list[ind] = max_list[ind] - D1
    min_list[ind] = min_list[ind] + 1.5 * D1
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    length = len(prev)
    # print(length)
    delete_these = []
    for i in range(length):
        # if not( prev[i][0][1]> max_list[int(prev[i][0][0])] or prev[i][0][1] < min_list[int(prev[i][0][0])]):
        if not (prev[i][0][1] < min_list[int(prev[i][0][0])]):
            # print("Hello")
            delete_these.append(i)
    prev = np.delete(prev, delete_these, axis=0)
    # elements in prev should be remained only if their y value is between min_list and max_list

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)

    point = []
    while (cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        if frame is None:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        # print(next)
        if next is None:
            good_new = good_old
        else:
            good_new = next[status == 1]
        # good_new = next[status == 1]
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        # print(a,b)
        #print(c - a)
        point.append(a)
        #output = cv.add(frame, mask)
        # cv.imshow("mask_only", mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        #cv.imshow("sparse optical flow", output)
        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    #print(len(dif))
    cap.release()
    cv.destroyAllWindows()
    return point



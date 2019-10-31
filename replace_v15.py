import argparse
import os
import cv2
import numpy as np
import torch
import statistics
import math
import torch.nn.functional as F

from pil import Image, ImageDraw

from Network import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
from utils import plot_img_and_mask
from util_v2 import static_point_detecor_homography, billboard_point_tracker_homography

from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable

# we consider the fixed points to start putting billboards before and after that point in order to make the billbord constant
point = [647]
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


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output
    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    #       >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
    #       >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
    #       >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# def rob_alg(lst):
#     while len(lst) > 5:
#         d = []
#         for i, val in enumerate(lst):
#             # print(i)
#             if i < len(lst) - 2:
#                 p1 = np.array([i, lst[i]])
#                 p2 = np.array([i + 2, lst[i + 2]])
#                 p3 = np.array([i + 1, lst[i + 1]])
#                 diff = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
#                 d.append(diff)
#         index = d.index(min(d))
#         print(len(lst))
#         lst.remove(lst[index])
#         print(len(lst))
#     return lst


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    import cv2
    image_lst = []
    vidcap = cv2.VideoCapture(in_files[0])
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        img_name = "frame"+str(count)+".jpg"
        image_lst.append(img_name)
        # print('Read a new frame: ', success)
        count += 1
    #sub = []
    base = 0
    kappa = 100
    rr = 0
    for j in range(len(image_lst)):
        #for i, fn in enumerate(image_lst[j]):
        # print("\nPredicting image {} ...".format(image_lst[j]))

        img = Image.open(image_lst[j])
        if j - base > 0:
            c, d, e = static_point_detecor_homography(image_lst[base], image_lst[j])
            if c > 100:
                point.append(c)
            else:
                base = j - 1
                if j - base > 0:
                    c, d, e = static_point_detecor_homography(image_lst[base], image_lst[j])
                    point.append(c)


        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=not args.no_crf,
                           use_gpu=not args.cpu)
        # print(mask)

        is_billboard_y, is_billboard_x = np.where(mask > 0)
        print(is_billboard_y.shape, is_billboard_x.shape)

        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        original_image = np.copy(opencv_image)
        # im_pil = Image.fromarray(img1)
        # opencv_image[mask>0] = 0
        # cv2.imshow('window_name', opencv_image)
        # cv2.waitKey(0)
        # plot_img_and_mask(img, mask)
        # exit(0)
        # Read source image.
        im_src = cv2.imread('framex.jpg')
        # Four corners of the billboard in source image
        img_gray = np.mean(im_src, axis=2)
        b, a = np.where(img_gray > 0)
        pts_src = np.array([[min(a), min(b)], [max(a), min(b)], [max(a), max(b)], [min(a), max(b)]])
        #print(pts_src)

        # Read destination image.
        # im_dst = cv2.imread('billboard1.jpg')
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
        D1 = 0.15 * D
        max_list[ind] = max_list[ind] - D1
        min_list[ind] = min_list[ind] + 1.5 * D1
        angle_list = []
        for x in range(0, length):
            if x > 0 and x < length - 200:
                # print((180 / np.pi) * angle_between(np.array([0, 1])-np.array([1, 1]), np.array([1, 1])))
                # exit(0)
                ang = (180 / np.pi) * angle_between(
                    np.array([x + 100, min_list[x + 100]]) - np.array([x, min_list[x]]),
                    np.array([x + 200, min_list[x + 200]]) - np.array([x + 100, min_list[x + 100]]))
                angle_list.append(ang)
                # print(ang)
                # if ang > 4:
                #     print('Here', x+100)
        ind_angle = angle_list.index(max(angle_list))
        # print(ind_angle+100)
        # print(rob_alg(min_list))
        # exit(0)
        #kappa = 100
        if j > 0:
            cc1, dd1, ee1 = billboard_point_tracker_homography(int(point[j-1]), min_list[int(point[j-1])],image_lst[j - 1], image_lst[j])
            cc2, dd2, ee2 = billboard_point_tracker_homography(int(point[j - 1]) + int(kappa), min_list[int(point[j - 1]) + int(kappa)],
                                                               image_lst[j - 1], image_lst[j])
            print(cc2 - cc1)
            rr = rr + cc2 - cc1 - kappa
            kk = kappa - round(cc2 - cc1)
            kappa = round(cc2 - cc1)
            if kk == 0:
                if abs(rr) > 1:
                    kappa = kappa + round(rr)
                    rr = 0
            else:
                rr = 0
            # kappa = round(cc2-cc1)
            # if abs(rr) > 1:
            #     kappa = kappa + round(rr)
            #     rr = 0
        for x in range(0, length):
            if x % int(kappa) == 0 and x + int(point[j]) + int(kappa) < length:
                #sub.append([j,x])
                pts_dst = np.array(
                    [[x + int(point[j]), min_list[x + int(point[j])]],
                     [x + int(point[j]) + int(kappa), min_list[x + int(point[j]) + int(kappa)]],
                     [x + int(point[j]) + int(kappa), max_list[x + int(point[j]) + int(kappa)]],
                     [x + int(point[j]), max_list[x + int(point[j])]]])
                # print(pts_dst)


                # Calculate Homography
                h, status = cv2.findHomography(pts_src, pts_dst)

                # Warp source image to destination based on homography
                temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 4)

                im_dst = im_dst + temp
                # im_dst = cv2.add(im_dst, temp)
                # show the process step by step
                # cv2.imshow("test", im_dst)
                # cv2.waitKey(0)
            if x % int(kappa) == 0 and x + int(point[j]) + int(kappa) >= length and x + int(point[j])< length:
                pts_dst = np.array(
                    [[x + int(point[j]), min_list[x + int(point[j])]], [len(min_list) - 1, min_list[len(min_list) - 1]],
                     [len(min_list) - 1, max_list[len(min_list) - 1]], [x + int(point[j]), max_list[x + int(point[j])]]])
                # print(pts_dst)
                # print(max(a)-min(a))
                cropped_src = np.array([[min(a), min(b)], [min(a) + ((length - (x + int(point[j]))) / int(kappa)) * (max(a) - min(a)), min(b)],
                                        [min(a) + ((length - (x + int(point[j]))) / int(kappa)) * (max(a) - min(a)), max(b)], [min(a), max(b)]])

                # Calculate Homography
                h, status = cv2.findHomography(cropped_src, pts_dst)

                # Warp source image to destination based on homography
                temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 4)
                im_dst = im_dst + temp
                # show the process step by step
                # cv2.imshow("test", im_dst)
                # cv2.waitKey(0)
            if x % int(kappa) == 0 and int(point[j]) - int(kappa) - x > 0:
                pts_dst2 = np.array(
                    [[int(point[j]) - int(kappa) - x, min_list[int(point[j]) - int(kappa) - x]],
                     [int(point[j]) - x, min_list[int(point[j]) - x]],
                     [int(point[j]) - x, max_list[int(point[j]) - x]],
                     [int(point[j]) - int(kappa) - x, max_list[int(point[j]) - int(kappa) - x]]])

                # Calculate Homography
                h2, status2 = cv2.findHomography(pts_src, pts_dst2)

                # Warp source image to destination based on homography
                temp2 = cv2.warpPerspective(im_src, h2, (im_dst.shape[1], im_dst.shape[0]))

                cv2.fillConvexPoly(im_dst, pts_dst2.astype(int), 0, 4)

                im_dst = im_dst + temp2

            if x % int(kappa) == 0 and int(point[j]) - int(kappa) - x <= 0 and int(point[j]) - x>0:
                pts_dst2 = np.array(
                    [[0, min_list[0]],
                     [int(point[j]) - x, min_list[int(point[j]) - x]],
                     [int(point[j]) - x, max_list[int(point[j]) - x]],
                     [0, max_list[0]]])

                cropped_src2 = np.array([[max(a) - ((int(point[j]) - x) / int(kappa)) * (max(a) - min(a)), min(b)], [max(a), min(b)],
                                        [max(a), max(b)], [max(a) - ((int(point[j]) - x) / int(kappa)) * (max(a) - min(a)), max(b)]])
                # print(cropped_src2)
                # print((int(point[j]) - x) / int(kappa))
                # print(int(point[j]) - x)
                # print(length)
                # exit(0)

                # Calculate Homography
                h2, status2 = cv2.findHomography(cropped_src2, pts_dst2)

                # Warp source image to destination based on homography
                temp2 = cv2.warpPerspective(im_src, h2, (im_dst.shape[1], im_dst.shape[0]))

                cv2.fillConvexPoly(im_dst, pts_dst2.astype(int), 0, 4)

                im_dst = im_dst + temp2

        alpha = 0.25  # amount to mix between [0, 1]
        # cv2.imwrite('original_image.jpg', original_image)
        # cv2.imwrite('im_dst.jpg', im_dst)
        im_dst[:] = alpha * original_image + (1 - alpha) * im_dst

        # cv2.imshow('warpped', im_dst)

        # Radius of circle
        radius = 20

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px


        if j - base > 0:
            # Center coordinates
            center_coordinates = (int(c), int(d))
            # Make one pixel red
            im_dst = cv2.circle(im_dst, center_coordinates, radius, color, thickness)
        else:
            # Center coordinates
            center_coordinates = (647, 553)
            # Make one pixel red
            im_dst = cv2.circle(im_dst, center_coordinates, radius, color, thickness)

        cv2.imwrite("E:/Frames/frame%d.jpg" % int(j+1068), im_dst)
        # cv2.waitKey(0)



image_folder = 'E:/Frames'
video_name = 'E:/Frames/video_second_set_replaced_v4.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 50, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
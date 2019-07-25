import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from pil import Image, ImageDraw

from Network import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks
from utils import plot_img_and_mask

from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable


def predict_img(net,
                full_img,
                scale_factor=0.25,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=True):
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

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
    parser.add_argument('--model', '-m', default='CP1.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'CP1.pth')")
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

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=not args.no_crf,
                           use_gpu=not args.cpu)
        #print(mask)

        is_billboard_y, is_billboard_x = np.where(mask > 0)
        print(is_billboard_y.shape,is_billboard_x.shape)
        import cv2
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        # im_pil = Image.fromarray(img1)
        # opencv_image[mask>0] = 0
        # cv2.imshow('window_name', opencv_image)
        #cv2.waitKey(0)
        plot_img_and_mask(img, mask)
        #exit(0)
        # Read source image.
        im_src = cv2.imread('frame1.jpg')
        # Four corners of the billboard in source image
        img_gray = np.mean(im_src, axis=2)
        b, a = np.where(img_gray > 0)
        pts_src = np.array([[min(a), min(b)], [max(a), min(b)], [max(a), max(b)], [min(a), max(b)]])

        # Read destination image.
        #im_dst = cv2.imread('billboard1.jpg')
        im_dst = opencv_image
        min_list = []
        max_list = []
        # Four corners of the billboard in destination image.
        for x in range(0, opencv_image.shape[1]):  # looping through each column
            lst = []
            for y in range(0, opencv_image.shape[0]):  # looping through each rows
                if mask[y, x] > 0:
                    lst.append(y)
            max_pt = (x, max(lst))
            min_pt = (x, min(lst))
            print(max_pt, min_pt)
            min_list.append(min(lst))
            max_list.append(max(lst))

        for x in range(0, opencv_image.shape[1]):
            if x % 100 == 0 and x+99 < len(min_list):
                if x == 0:
                    pts_dst = np.array(
                        [[x, min_list[x]], [x + 99, min_list[x + 99]], [x + 99, max_list[x + 99]], [x, max_list[x]]])
                    #print(pts_dst)
                else:
                    pts_dst = np.array(
                        [[x-1, min_list[x-1]], [x + 99, min_list[x + 99]], [x + 99, max_list[x + 99]], [x-1, max_list[x-1]]])
                    #print(pts_dst)

                # Calculate Homography
                h, status = cv2.findHomography(pts_src, pts_dst)

                # Warp source image to destination based on homography
                temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 4)

                im_dst = im_dst + temp
                #im_dst = cv2.add(im_dst, temp)
                # show the process step by step
                # cv2.imshow("test", im_dst)
                # cv2.waitKey(0)
            if x % 100 == 0 and x+99 >= len(min_list):
                pts_dst = np.array(
                    [[x-1, min_list[x-1]], [len(min_list)-1, min_list[len(min_list)-1]], [len(min_list)-1, max_list[len(min_list)-1]], [x-1, max_list[x-1]]])
                #print(pts_dst)
                # Calculate Homography
                h, status = cv2.findHomography(pts_src, pts_dst)

                # Warp source image to destination based on homography
                temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 4)
                im_dst = im_dst + temp
                # show the process step by step
                # cv2.imshow("test", im_dst)
                # cv2.waitKey(0)


        #exit(0)

        # catching the edges
        # imgray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(imgray, 127, 255, 0)
        #
        # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # Just to show all rectangles are found
        # cv2.drawContours(opencv_image, contours, -1, (255, 255, 255), 5)
        # cv2.imwrite("a.png", opencv_image)
        # gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        # using Hough Transform
        # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        #
        # lines = cv2.HoughLines(edges, 5, np.pi / 180, 200)
        # for rho, theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #
        #     cv2.line(opencv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #
        # cv2.imwrite('houghlines.jpg', opencv_image)
        # exit(0)
        # fit line using all data
        # # using ransac
        # from skimage.measure import LineModelND, ransac
        # from matplotlib import pyplot as plt
        # model = LineModelND()
        # model.estimate(gray)
        # # robustly fit line only using inlier data with RANSAC algorithm
        # model_robust, inliers = ransac(gray, LineModelND, min_samples=2, residual_threshold=1,
        #                                max_trials=1000)
        # outliers = inliers == False
        #
        # # generate coordinates of estimated models
        # line_x = np.arange(-250, 250)
        # line_y = model.predict_y(line_x)
        # line_y_robust = model_robust.predict_y(line_x)
        # # data1= np.vstack((line_x,line_y)).T
        # # data2= np.vstack((line_x,line_y_robust)).T
        #
        # # img1 = Image.fromarray(data1, 'RGB')
        # # img2 = Image.fromarray(data2, 'RGB')
        # # print(img1)
        #
        # # fig, ax = plt.subplots()
        # # ax.plot(gray[inliers, 0], gray[inliers, 1], '.b', alpha=0.6,
        # #         label='Inlier data')
        # # ax.plot(gray[outliers, 0], gray[outliers, 1], '.r', alpha=0.6,
        # #         label='Outlier data')
        # # ax.plot(line_x, line_y, '-k', label='Line model from all data')
        # # ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
        # # ax.legend(loc='lower left')
        # # plt.show()
        # exit(0)

        # pts_dst = np.array([[0, 304], [500, 288], [500, 335], [0, 346]])
        #
        # # Calculate Homography
        # h, status = cv2.findHomography(pts_src, pts_dst)
        #
        # # Warp source image to destination based on homography
        # temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        #
        # cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
        #
        # im_dst = im_dst + temp
        # # iteration
        # pts_dst = np.array([[501, 288], [600, 295], [600, 342], [501, 335]])
        #
        # # Calculate Homography
        # h, status = cv2.findHomography(pts_src, pts_dst)
        #
        # # Warp source image to destination based on homography
        # temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        #
        # cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
        #
        # im_dst = im_dst + temp
        # # iteration
        # pts_dst = np.array([[601, 295], [1100, 359], [1100, 411], [601, 342]])
        #
        # # Calculate Homography
        # h, status = cv2.findHomography(pts_src, pts_dst)
        #
        # # Warp source image to destination based on homography
        # temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        #
        # cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
        #
        # im_dst = im_dst + temp
        # # iteration
        # pts_dst = np.array([[1100, 359], [1919, 482], [1919, 537], [1100, 411]])
        #
        # # Calculate Homography
        # h, status = cv2.findHomography(pts_src, pts_dst)
        #
        # # Warp source image to destination based on homography
        # temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        #
        # cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
        #
        # im_dst = im_dst + temp

        # Display images
        # cv2.imshow("Source Image", im_src)
        # cv2.imshow("Destination Image", im_dst)
        # cv2.imshow("Warped Source Image", im_out)
        cv2.imshow('warpped', im_dst)
        cv2.imwrite('replaced.jpg', im_dst)
        cv2.waitKey(0)
        exit(0)
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))
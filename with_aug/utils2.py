import os
import random
import numpy as np
import matplotlib.pyplot as plt
#import pydensecrf.densecrf as dcrf
from PIL import Image
import torchvision
from torchvision import transforms




def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=1, final_height=None, is_mask=False):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    if is_mask:
        # convert to numpy array, remove alpha channel, convert back to pilimag
        # otherwise pilimg resize will make the image transparent (all black)
        pilimg = np.array(pilimg)
        pilimg = pilimg[:, :, :3]  # remove alpha channel
        img = Image.fromarray(pilimg)
        #pilimg = Image.fromarray(pilimg)
        #pilimg = pilimg.resize((newW//8, newH//8))
        #img = pilimg.crop((0, 0, newW//8, newH//8))
        # img = pilimg.torchvision.transforms.RandomCrop((newW // 8, newH // 8), padding=None, pad_if_needed=False,
        #                                                fill=0, padding_mode='constant')
    else:
        #pilimg = pilimg.resize((newW // 8, newH // 8))
        img = pilimg.crop((0, 0, newW // 1, newH // 1))
        # img = pilimg.torchvision.transforms.RandomCrop((newW // 8, newH // 8), padding=None, pad_if_needed=False,
        #                                                fill=0, padding_mode='constant')

    # plot_img_and_mask(img, img)
    # print(is_mask)
    # exit(0)
    #img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.1):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
# def rle_encode(mask_image):
#     pixels = mask_image.flatten()
#     # We avoid issues with '1' at the start or end (at the corners of
#     # the original image) by setting those pixels to '0' explicitly.
#     # We do not expect these to be non-zero for an accurate mask,
#     # so this should not harm the score.
#     pixels[0] = 0
#     pixels[-1] = 0
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     runs[1::2] = runs[1::2] - runs[:-1:2]
#     return runs


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()


# def dense_crf(img, output_probs):
#     h = output_probs.shape[0]
#     w = output_probs.shape[1]
#
#     output_probs = np.expand_dims(output_probs, 0)
#     output_probs = np.append(1 - output_probs, output_probs, axis=0)
#
#     d = dcrf.DenseCRF2D(w, h, 2)
#     U = -np.log(output_probs)
#     U = U.reshape((2, -1))
#     U = np.ascontiguousarray(U)
#     img = np.ascontiguousarray(img)
#
#     d.setUnaryEnergy(U)
#
#     d.addPairwiseGaussian(sxy=20, compat=3)
#     d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
#
#     Q = d.inference(5)
#     Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
#
#     return Q

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-8] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale, is_mask=False):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale, is_mask=is_mask)

        if is_mask:
            # change to mask 0 or 1, 1 is for billboards
            im = im.any(axis=2, keepdims=True).astype(im.dtype)
            # import cv2
            # cv2.imwrite("test.png", im*255)

        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '_vis.PNG', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_coords.PNG', scale, is_mask=True)


    return zip(imgs_normalized, masks)


# def get_full_img_and_mask(id, dir_img, dir_mask):
#     im = Image.open(dir_img + id + '_vis.PNG')
#     mask = Image.open(dir_mask + id + '_coords.PNG')
#     return np.array(im), np.array(mask)
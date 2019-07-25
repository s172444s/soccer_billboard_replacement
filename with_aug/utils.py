import os
import random
import numpy as np
import matplotlib.pyplot as plt
#import pydensecrf.densecrf as dcrf
from PIL import Image
import cv2
from numpy import newaxis
import torchvision
from torchvision import transforms
import random




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


def resize_and_crop(pilimg, scale=0.25, final_height=None, is_mask=False):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    #trans = transforms.Compose([transforms.RandomResizedCrop(270)])
    if is_mask:
        # convert to numpy array, remove alpha channel, convert back to pilimag
        # otherwise pilimg resize will make the image transparent (all black)
        pilimg = np.array(pilimg)
        pilimg = pilimg[:, :, :3]  # remove alpha channel
        pilimg = Image.fromarray(pilimg)
        pilimg = pilimg.resize((newW, newH))
        img = pilimg.crop((0, 0, newW, newH))
        #img = trans(pilimg)
    else:
        pilimg = pilimg.resize((newW, newH))
        img = pilimg.crop((0, 0, newW, newH))
        #img = trans(pilimg)

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


#extra functions

def resize_and_crop_img_and_mask(pilimg_img, pilimg_mask, scale=1, final_height=None, is_mask=False):
    w = pilimg_img.size[0]
    h = pilimg_img.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    #trans = transforms.Compose([transforms.RandomResizedCrop(270)])
    if is_mask:
        # convert to numpy array, remove alpha channel, convert back to pilimag
        # otherwise pilimg resize will make the image transparent (all black)
        pilimg_mask = np.array(pilimg_mask)
        pilimg_mask = pilimg_mask[:, :, :3]  # remove alpha channel
        pilimg_mask = Image.fromarray(pilimg_mask)
        #print(pilimg_mask)
        # pilimg = pilimg.resize((newW//4, newH//4))
        # img = pilimg.crop((0, 0, newW // 4, newH // 4))
        #mask = trans(pilimg_mask)
    #else:
        # pilimg = pilimg.resize((newW // 4, newH // 4))
        # img = pilimg.crop((0, 0, newW // 4, newH // 4))
        #img = trans(pilimg_img)
    #img = trans(pilimg_img)
    #mask = trans(pilimg_mask)
    # plot_img_and_mask(img, img)
    # print(is_mask)
    # exit(0)
    #img = img.crop((0, diff // 2, newW, newH - diff // 2))

    im, mask = np.array(pilimg_img, dtype=np.float32), np.array(pilimg_mask, dtype=np.float32)

    # change to mask 0 or 1, 1 is for billboards
    mask = mask.any(axis=2, keepdims=True).astype(mask.dtype)
    # import cv2
    # cv2.imwrite("test.png", im*255)
    width = im.shape[1]
    height = im.shape[0]
    is_billboard_y, is_billboard_x = np.where(mask[:, :, 0] == 1)
    n_billboard_pixels = len(is_billboard_x)
    if n_billboard_pixels>0:
        random_pixel = np.random.randint(0, n_billboard_pixels)
        random_pixel_x = is_billboard_x[random_pixel]
        random_pixel_y = is_billboard_y[random_pixel]
        # target_width = min(abs(random_pixel_x - 270), abs(random_pixel_y - 270))
        # target_height = min(abs(random_pixel_x - 270), abs(random_pixel_y - 270))
        target_width = 540
        target_height = 540
        if target_width == 0 | target_height==0:
            x0 = np.random.randint(0, width - 540)
            y0 = np.random.randint(0, height - 540)
        else:
            x0 = random_pixel_x
            y0 = random_pixel_y
        # x0 = random_pixel_x
        # y0 = random_pixel_y
    else:
        target_width = 540
        target_height = 540
        x0 = np.random.randint(0, width - target_width)
        y0 = np.random.randint(0, height - target_height)
        # debugging
    # import cv2
    # test = np.copy(mask)
    # test = np.concatenate([test, test, test], axis=2)
    # print(random_pixel_x, random_pixel_y)
    # cv2.circle(test, (random_pixel_x, random_pixel_y), 10, (1, 0, 1), thickness=-1)
    # cv2.imwrite("_random_pixel.png", test*255)
    # exit(0)




    # width = im.shape[1]
    # height = im.shape[0]

    # target_width = 270
    # target_height = 270

    # x0 = np.random.randint(0, width-target_width)
    # y0 = np.random.randint(0, height-target_height)
    # x0 = random_pixel_x
    # y0 = random_pixel_y

    im = im[y0:y0+target_height, x0:x0+target_width]
    mask = mask[y0:y0+target_height, x0:x0+target_width]
    import cv2
    from numpy import newaxis
    im = cv2.resize(im, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
    mask = mask[:, :, newaxis]

    #print(im.shape, mask.shape, "HERE")

    return im, mask

def distorter(width, height, img, mask):
    max_side = max(height, width)
    x_max = width / max_side
    y_max = height / max_side

    # "normalized" coordinates
    x_normalized = np.empty([height, width], np.float32)
    y_normalized = np.empty([height, width], np.float32)

    # scale to be [0, x_max] X [0, y_max]
    x_normalized[:, :] = np.linspace(0, x_max, width, endpoint=False)[np.newaxis, :]
    y_normalized[:, :] = np.linspace(0, y_max, height, endpoint=False)[:, np.newaxis]

    # shift so that [0, 0] is the middle of the image
    x_normalized = x_normalized - x_max / 2
    y_normalized = y_normalized - y_max / 2

    # do the transformation on normalized coordinates
    c = np.array([0.0, 0.0], dtype=np.float32)
    k = 0.1
    cx = c[0]
    cy = c[1]
    radius_squared = (x_normalized - cx) ** 2 + (y_normalized - cy) ** 2
    radius = np.sqrt(radius_squared)

    f = 1.0 / (1 - k * radius)

    x_transformed = (x_normalized - cx) * f + cx
    y_transformed = (y_normalized - cy) * f + cy
    #x_transformed, y_transformed = random_perspective(x_normalized, y_normalized)

    # return to original lookup coordinates
    x_lookup = max_side * (x_transformed + x_max / 2)
    y_lookup = max_side * (y_transformed + y_max / 2)

    interpolation = cv2.INTER_LINEAR
    borderMode = cv2.BORDER_CONSTANT
    transformed_img = cv2.remap(img, x_lookup, y_lookup, interpolation=interpolation, borderMode=borderMode)
    transformed_mask = cv2.remap(mask, x_lookup, y_lookup, interpolation=interpolation, borderMode=borderMode)
    return transformed_img, transformed_mask

def perspector(width, height, img, mask):
    max_side = max(height, width)
    x_max = width / max_side
    y_max = height / max_side

    # "normalized" coordinates
    x_normalized = np.empty([height, width], np.float32)
    y_normalized = np.empty([height, width], np.float32)

    # scale to be [0, x_max] X [0, y_max]
    x_normalized[:, :] = np.linspace(0, x_max, width, endpoint=False)[np.newaxis, :]
    y_normalized[:, :] = np.linspace(0, y_max, height, endpoint=False)[:, np.newaxis]

    # shift so that [0, 0] is the middle of the image
    x_normalized = x_normalized - x_max / 2
    y_normalized = y_normalized - y_max / 2
    x_transformed = (x_normalized + 0.1 * (y_normalized ** 2)) / (1 + 0.01 * x_normalized + 0.1 * (x_normalized ** 2))
    y_transformed = (y_normalized + 0.1 * (x_normalized ** 2)) / (1 + 0.01 * y_normalized + 0.1 * (y_normalized ** 2))
    x_lookup = max_side * (x_transformed + x_max / 2)
    y_lookup = max_side * (y_transformed + y_max / 2)

    interpolation = cv2.INTER_LINEAR
    borderMode = cv2.BORDER_CONSTANT
    transformed_img = cv2.remap(img, x_lookup, y_lookup, interpolation=interpolation, borderMode=borderMode)
    transformed_mask = cv2.remap(mask, x_lookup, y_lookup, interpolation=interpolation, borderMode=borderMode)
    return transformed_img, transformed_mask

def resize_and_crop_img_and_mask_v2(pilimg_img, pilimg_mask, id, scale=1, final_height=None, is_mask=False):
    h = pilimg_img.size[0]
    w = pilimg_img.size[1]
    im, mask = np.array(pilimg_img, dtype=np.float32), np.array(pilimg_mask, dtype=np.float32)
    # change to mask 0 or 1, 1 is for billboards
    mask = mask.any(axis=2, keepdims=True).astype(mask.dtype)
    rdm = np.random.randint(0,4)
    if rdm == 0:
        target_Width = np.random.randint(400, 1080)
        target_Height = np.random.randint(500, 1920)

        no_board = True
        ctr = 0
        while(no_board == True):
            ctr += 1
            x0 = np.random.randint(target_Width // 2, w - target_Width // 2)
            y0 = np.random.randint(target_Height // 2, h - target_Height // 2)
            #x0 = int(546)
            #y0 = int(1800)
            diff = abs(x0-y0)
            if diff > 400:
                y0 = x0+400
            new_im = im[y0 + 1 - target_Height // 2:y0 - 1 + target_Height // 2, x0 + 1 - target_Width // 2:x0 - 1 + target_Width // 2]
            new_mask = mask[y0 - target_Height // 2:y0 + target_Height // 2, x0 - target_Width // 2:x0 + target_Width // 2]
            if (new_mask.any() == False) & (ctr < 10):
                continue
            # print("x0 "+str(x0))
            # print("y0 "+str(y0))
            # print(id)
            #cv2.imwrite('created/' + id + '_image.PNG', np.asarray(new_im))
            #cv2.imwrite('created/' + id + '_mask.PNG', np.asarray(mask))
            new_im = cv2.resize(new_im, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
            new_mask = cv2.resize(new_mask, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
            new_mask = new_mask[:, :, newaxis]
            msk = np.copy(new_mask)
            msk[msk==1] = 255
            #print("bill_board")
            #cv2.imwrite('created/' + id + '_image.PNG', np.asarray(new_im))
            #cv2.imwrite('created/' + id + '_mask.PNG', np.asarray(msk))
            no_board = False
    elif rdm == 1:
        new_im = cv2.resize(im, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
        new_mask = cv2.resize(mask, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
        new_mask = new_mask[:, :, newaxis]
        msk = np.copy(new_mask)
        msk[msk == 1] = 255
        # cv2.imwrite('created/' + id + '_image.PNG', np.asarray(new_im))
        # cv2.imwrite('created/' + id + '_mask.PNG', np.asarray(msk))
    elif rdm == 2:
        im, mask = distorter(h,w,im,mask)
        new_im = cv2.resize(im, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
        new_mask = cv2.resize(mask, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
        new_mask = new_mask[:, :, newaxis]
        msk = np.copy(new_mask)
        msk[msk == 1] = 255
        # cv2.imwrite('created/' + id + '_image.PNG', np.asarray(new_im))
        # cv2.imwrite('created/' + id + '_mask.PNG', np.asarray(msk))
    elif rdm == 3:
        im, mask = perspector(h, w, im, mask)
        new_im = cv2.resize(im, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
        new_mask = cv2.resize(mask, dsize=(540, 540), interpolation=cv2.INTER_LINEAR)
        new_mask = new_mask[:, :, newaxis]
        msk = np.copy(new_mask)
        msk[msk == 1] = 255
        #cv2.imwrite('created/' + id + '_image.PNG', np.asarray(new_im))
        #cv2.imwrite('created/' + id + '_mask.PNG', np.asarray(msk))
    #cv2.imwrite('created/' + id + '_image.PNG', np.asarray(new_im))
    #cv2.imwrite('created/' + id + '_mask.PNG', np.asarray(msk))
    return new_im, new_mask


def resize_and_crop_for_predict(pilimg, scale=1, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    trans = transforms.Compose([transforms.RandomResizedCrop(540)])
    # pilimg = pilimg.resize((newW // 4, newH // 4))
    # img = pilimg.crop((0, 0, newW // 4, newH // 4))
    img = trans(pilimg)

    # plot_img_and_mask(img, img)
    # print(is_mask)
    # exit(0)
    #img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def to_cropped_imgs_and_mask(ids, dir_img, suffix_img, dir_mask, suffix_mask, scale, is_mask=False):
    """From a list of tuples, returns the correct cropped img"""
    # while(True): #for i in range(10000000):
    #     id, pos = get_random_id_pos()

    #print(type(ids), len(ids))

    for i in range(10):
        print(i)
        for id, pos in ids:
            #print(id, pos)

            im, mask = resize_and_crop_img_and_mask_v2(Image.open(dir_img + id + suffix_img),
                                                       Image.open(dir_mask + id + suffix_mask), id, scale=scale,
                                                       is_mask=is_mask)

            # if is_mask:
            #     # change to mask 0 or 1, 1 is for billboards
            #     mask = mask.any(axis=2, keepdims=True).astype(mask.dtype)
            #     # import cv2
            #     # cv2.imwrite("test.png", im*255)

            im_switched = hwc_to_chw(im)  # switches channels axis
            mask_switched = hwc_to_chw(mask)

            #im_normalized = normalize(im_switched)  # normalizes image [0, 1]
            # mask already in range [0, 1]

            #final_im = im_normalized
            final_im = im_switched
            final_mask = mask_switched
            # # debug
            # import cv2
            # # multiply by 255 to save image in range [0, 255]
            # cv2.imwrite("_test_0.png", final_im[0, :, :]*255)  # only save one channel, image will look grey scale
            # cv2.imwrite("_test_1.png", final_mask[0, :, :]*255)
            # #
            # exit(0)

            # yield get_square(final_im, pos), get_square(final_mask, pos)
            yield final_im, final_mask



def get_imgs_and_masks_both(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    imgs_and_masks = to_cropped_imgs_and_mask(ids, dir_img, '_vis.PNG', dir_mask, '_coords.PNG', scale, is_mask=True)

    # need to transform from HWC to CHW
    # imgs_switched = map(hwc_to_chw, imgs)
    # imgs_normalized = map(normalize, imgs_switched)
    #masks = to_cropped_imgs(ids, dir_mask, '_coords.PNG', scale, is_mask=True)
    #return zip(imgs_normalized, masks)

    return imgs_and_masks

# def get_full_img_and_mask(id, dir_img, dir_mask):
#     im = Image.open(dir_img + id + '_vis.PNG')
#     mask = Image.open(dir_mask + id + '_coords.PNG')
#     return np.array(im), np.array(mask)
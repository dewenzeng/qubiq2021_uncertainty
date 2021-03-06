import math
import numbers
import random
import torch
import torchvision.transforms.functional as tf
from PIL import Image, ImageOps
import cv2

# zbabby(2019/2/21)
# All of the augmentation for PIL image
# modified by Dewen Zeng to handle list of labels

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, imgs, masks):
        if isinstance(imgs[0], np.ndarray):
            imgs = [Image.fromarray(img, mode="L") for img in imgs]
            masks = [Image.fromarray(mask, mode="L") for mask in masks]
            self.PIL2Numpy = True

        assert imgs[0].size == masks[0].size
        for a in self.augmentations:
            imgs, masks = a(imgs, masks)

        if self.PIL2Numpy:
            imgs = [np.array(img) for img in imgs]
            masks = [np.array(mask, dtype=np.uint8) for mask in masks]

        return imgs, masks

class ToTensor(object):
    def __call__(self, imgs, masks):
        return [tf.to_tensor(img) for img in imgs], [torch.from_numpy(np.array(mask)).long() for mask in masks]

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs, masks):
        return [tf.normalize(img,self.mean,self.std) for img in imgs], masks

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, masks):
        return [tf.resize(img,self.size) for img in imgs], [tf.resize(mask,self.size) for mask in masks]


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        return [tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)) for img in imgs], masks


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        return [tf.adjust_saturation(img,
                                    random.uniform(1 - self.saturation,
                                                   1 + self.saturation)) for img in imgs], masks
class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        return [tf.adjust_hue(img, random.uniform(-self.hue,
                                                 self.hue)) for img in imgs], masks

class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        return [tf.adjust_brightness(img,
                                    random.uniform(1 - self.bf,
                                                   1 + self.bf)) for img in imgs], masks

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        return [tf.adjust_contrast(img,
                                  random.uniform(1 - self.cf,
                                                 1 + self.cf)) for img in imgs], masks


class RandomHorizontallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, masks):
        if random.random() < self.p:
            return (
                [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs],
                [mask.transpose(Image.FLIP_LEFT_RIGHT) for mask in masks],
            )
        return (imgs, masks)


class RandomVerticallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, masks):
        if random.random() < self.p:
            return (
                [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs],
                [mask.transpose(Image.FLIP_TOP_BOTTOM) for mask in masks],
            )
        return (imgs, masks)


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        return (
            [img.resize(self.size, Image.BILINEAR) for img in imgs],
            [mask.resize(self.size, Image.NEAREST) for mask in masks],
        )

class RandomZoom(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        if random.random() < 0.5:
            new_size = (int(imgs[0].size[0]*self.size[0]), int(imgs[0].size[1]*self.size[1]))
            return (
                [img.resize(new_size, Image.BILINEAR) for img in imgs],
                [mask.resize(new_size, Image.NEAREST) for mask in masks],
            )
        return (imgs, masks)


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset  # tuple (delta_x, delta_y), 0~1

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        x_offset = int((2 * (random.random() - 0.5) * self.offset[0])*imgs[0].size[0])
        y_offset = int((2 * (random.random() - 0.5) * self.offset[1])*imgs[0].size[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_imgs = [tf.crop(img,
                              y_crop_offset,
                              x_crop_offset,
                              img.size[1] - abs(y_offset),
                              img.size[0] - abs(x_offset)) for img in imgs]

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            [tf.pad(cropped_img,
                   padding_tuple,
                   padding_mode='reflect') for cropped_img in cropped_imgs],
            [tf.affine(mask,
                      translate=(-x_offset, -y_offset),
                      scale=1.0,
                      angle=0.0,
                      shear=0.0,
                      fillcolor=0) for mask in masks])

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree #  -180 and 180

    def __call__(self, imgs, masks):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            [tf.affine(img,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.NEAREST,
                      fillcolor=(0, 0, 0) if len(img.size) == 3 else 0,
                      shear=0.0) for img in imgs],
            [tf.affine(mask,
                      translate=(0, 0),
                      scale=1.0,
                      angle=rotate_degree,
                      resample=Image.NEAREST,
                      fillcolor=0,
                      shear=0.0) for mask in masks])

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        w, h = imgs[0].size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return imgs, masks
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                [img.resize((ow, oh), Image.BILINEAR) for img in imgs],
                [mask.resize((ow, oh), Image.NEAREST) for mask in masks],
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                [img.resize((ow, oh), Image.BILINEAR) for img in imgs],
                [mask.resize((ow, oh), Image.NEAREST) for mask in masks],
            )


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs, masks):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]
            masks = [ImageOps.expand(mask, border=self.padding, fill=0) for mask in masks]

        assert imgs[0].size == masks[0].size
        w, h = imgs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return imgs, masks
        if w < tw or h < th:
            return (
                [img.resize((tw, th), Image.BILINEAR) for img in imgs],
                [mask.resize((tw, th), Image.NEAREST) for mask in masks],
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs],
            [mask.crop((x1, y1, x1 + tw, y1 + th)) for mask in masks],
        )


class RandomSizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        for attempt in range(10):
            area = imgs[0].size[0] * imgs[0].size[1]
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
                x1 = random.randint(0, imgs[0].size[0] - w)
                y1 = random.randint(0, imgs[0].size[1] - h)

                imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
                masks = [mask.crop((x1, y1, x1 + w, y1 + h)) for mask in masks]
                assert imgs[0].size == (w, h)

                return (
                    [img.resize((self.size, self.size), Image.BILINEAR) for img in imgs],
                    [mask.resize((self.size, self.size), Image.NEAREST) for mask in masks] ,
                )

        # Notice, we must guarantee crop to the expected size
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(imgs, masks))

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size
        w, h = imgs[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs],
            [mask.crop((x1, y1, x1 + tw, y1 + th)) for mask in masks],
        )

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, imgs, masks):
        assert imgs[0].size == masks[0].size

        w = int(random.uniform(0.5, 2) * imgs[0].size[0])
        h = int(random.uniform(0.5, 2) * imgs[0].size[1])

        imgs, masks = (
            [img.resize((w, h), Image.BILINEAR) for img in imgs],
            [mask.resize((w, h), Image.NEAREST) for mask in masks],
        )

        return self.crop(*self.scale(imgs, masks))

class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, imgs, masks):
        return ([ImageOps.expand(img, border=self.padding, fill=self.fill) for img in imgs],
                [ImageOps.expand(mask, border=self.padding, fill=self.fill) for mask in masks])


class RandomElasticTransform(object):
    def __init__(self, alpha = 3, sigma=0.07, img_type='L'):
        self.alpha = alpha
        self.sigma = sigma
        self.img_type = img_type

    def _elastic_transform(self, imgs, masks):

        # convert to numpy
        imgs = [np.array(img) for img in imgs]  # hxwxc
        masks = [np.array(mask) for mask in masks]

        shape1=imgs[0].shape

        alpha = self.alpha*shape1[0]
        sigma = self.sigma*shape1[0]

        x, y = np.meshgrid(np.arange(shape1[0]), np.arange(shape1[1]), indexing='ij')
        blur_size = int(4 * sigma) | 1
        dx = cv2.GaussianBlur((np.random.rand(shape1[0], shape1[1]) * 2 - 1), ksize=(blur_size, blur_size),sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(shape1[0], shape1[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

        if (x is None) or (y is None):
            x, y = np.meshgrid(np.arange(shape1[0]), np.arange(shape1[1]), indexing='ij')

        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        # convert map
        map_x, map_y = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

        imgs = [cv2.remap(img, map_y, map_x, interpolation=cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT).reshape(shape1) for img in imgs]
        masks =  [cv2.remap(mask, map_y, map_x, interpolation=cv2.INTER_NEAREST, borderMode = cv2.BORDER_CONSTANT).reshape(shape1) for mask in masks]

        return ([Image.fromarray(img,mode=self.img_type) for img in imgs], [Image.fromarray(mask, mode='L') for mask in masks])

    def __call__(self, imgs, masks):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        if random.random() < 0.5:
            return self._elastic_transform(imgs, masks)
        else:
            return (imgs, masks)



import cv2
import numpy as np
import SimpleITK as sitk

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)

    return imgs

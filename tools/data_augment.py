import time

import imgaug as ia
from scipy import ndimage


class MyBoundingBox(ia.BoundingBox):
    def __init__(self, store, x1=0, y1=0, x2=1, y2=1, bounding_box=None):
        if bounding_box is None:
            super(MyBoundingBox, self).__init__(x1, y1, x2, y2)
        else:
            super(MyBoundingBox, self).__init__(bounding_box.x1,
                                                bounding_box.y1,
                                                bounding_box.x2,
                                                bounding_box.y2)
        self.store = store


def argument(imsl_img, num_per_resolution, is_augment):
    ia.seed(int(time.time()))

    image = ndimage.imread(imsl_img.image_path, mode="RGB")
    bbd_list = []
    for region in imsl_img.objects:
        bbd_list.append(
            MyBoundingBox(
                x1=region.xmin,
                y1=region.ymin,
                x2=region.xmax,
                y2=region.ymax,
                store=str(region.id)
            )
        )

    if len(bbd_list) == 0:
        return [], []

    image_list = []
    bounding_box = []

    bbs = ia.BoundingBoxesOnImage(bbd_list, shape=image.shape)

    image_list.append(image)
    bounding_box.append(bbs)

    if is_augment:
        flag = True
        while flag:
            try:
                (images, bndboxs) = do_argument(image, bbs, num_per_resolution)

                image_list.extend(images)
                bounding_box.extend(bndboxs)
                flag = False
            except Exception as e:
                print("%s  Error!!  %s" % (imsl_img.image_path, e))

    return image_list, bounding_box


def do_argument(image, bounding_box, num_per_size):
    scale_seq_list = [
        ia.augmenters.Scale({"width": 2000, "height": "keep-aspect-ratio"}),
        ia.augmenters.Scale({"width": 1440, "height": "keep-aspect-ratio"}),
        ia.augmenters.Scale({"width": 1024, "height": "keep-aspect-ratio"}),
        ia.augmenters.Scale({"width": 800, "height": "keep-aspect-ratio"}),
        ia.augmenters.Scale({"width": 640, "height": "keep-aspect-ratio"}),
        ia.augmenters.Scale({"width": 320, "height": "keep-aspect-ratio"}),
    ]

    proj_seq = ia.augmenters.PerspectiveTransform(0.12, 0.12)

    blur_seq = ia.augmenters.OneOf([
        ia.augmenters.GaussianBlur(sigma=(0.2, 3.0)),
        ia.augmenters.AverageBlur(k=(2, 11.0)),
        ia.augmenters.MedianBlur(k=(1.0, 5.0))
    ])

    sharp_seq = ia.augmenters.Sharpen(alpha=(0, 1), lightness=(0.5, 2))

    bright_seq = ia.augmenters.Multiply(mul=(0.25, 2), per_channel=0.2)

    image_list = []
    bounding_box_list = []

    for scale_seq in scale_seq_list:
        aug_seq = ia.augmenters.Sequential(
            [
                scale_seq,
                ia.augmenters.SomeOf(n=(1, None), children=[
                    bright_seq,
                    proj_seq,
                    ia.augmenters.OneOf([blur_seq, sharp_seq]),
                ], random_order=True)
            ]
        )

        for index in range(num_per_size):
            aug_seq_det = aug_seq.to_deterministic()

            image_aug = aug_seq_det.augment_image(image)
            bbs_aug = aug_seq_det.augment_bounding_boxes([bounding_box])[0]
            image_list.append(image_aug)

            bbs_aug_list = []
            for bb_aug, bb in zip(bbs_aug.bounding_boxes, bounding_box.bounding_boxes):
                bbs_aug_list.append(
                    MyBoundingBox(store=bb.store, bounding_box=bb_aug)
                )

            mybbs_aug = ia.BoundingBoxesOnImage(bbs_aug_list, shape=image.shape)

            bounding_box_list.append(mybbs_aug)

            # width, height, depth = image_aug.shape
            # print('*** Augment in resolution %dx%dx%d, index %d' % (width, height, depth, index))

    return filter_bounding_box(image_list, bounding_box_list)


def filter_bounding_box(image_list, bounding_box_list):
    new_bounding_box_list = []

    for index in range(len(image_list)):
        image = image_list[index]
        bbs = bounding_box_list[index]

        new_bbs = [bb for bb in bbs.bounding_boxes if bb.is_fully_within_image(image.shape)]
        new_bounding_box_list.append(ia.BoundingBoxesOnImage(new_bbs, shape=image.shape))

    return image_list, new_bounding_box_list

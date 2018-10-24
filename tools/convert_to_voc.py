import os
import shutil

from scipy import misc

import data_augment
from ProgressBar import ProgressBar

annotations_template = '<annotation>\n' + \
                       '\t<folder>signboard</folder>\n' + \
                       '\t<filename>%s.jpg</filename>\n' + \
                       '\t<source>\n' + \
                       '\t\t<database>My Database</database>\n' + \
                       '\t\t<annotation>VOC2007</annotation>\n' + \
                       '\t\t<image>flickr</image>\n' + \
                       '\t\t<flickrid>NULL</flickrid>\n' + \
                       '\t</source>\n' + \
                       '\t<onwer>\n' + \
                       '\t\t<flickrid>NULL</flickrid>\n' + \
                       '\t\t<name>SmartYi</name>\n' + \
                       '\t</onwer>\n' + \
                       '\t<size>\n' + \
                       '\t\t<width>%d</width>\n' + \
                       '\t\t<height>%d</height>\n' + \
                       '\t\t<depth>%d</depth>\n' + \
                       '\t</size>\n' + \
                       '\t<segmented>0</segmented>\n'

object_template = '\t<object>\n' + \
                  '\t\t<name>%s</name>\n' + \
                  '\t\t<pose>Unspecified</pose>\n' + \
                  '\t\t<truncated>0</truncated>\n' + \
                  '\t\t<difficult>0</difficult>\n' + \
                  '\t\t<bndbox>\n' + \
                  '\t\t\t<xmin>%d</xmin>\n' + \
                  '\t\t\t<ymin>%d</ymin>\n' + \
                  '\t\t\t<xmax>%d</xmax>\n' + \
                  '\t\t\t<ymax>%d</ymax>\n' + \
                  '\t\t</bndbox>\n' + \
                  '\t</object>\n'

annotation_final = '</annotation>'


def format_annotation(output_dir, imsl_image, image_list, bounding_box, binary):
    for index in range(len(image_list)):
        image = image_list[index]
        bbs = bounding_box[index]

        if len(bbs.bounding_boxes) == 0:
            continue

        annotation = dict()
        annotation['name'] = '%s-%d' % (imsl_image.name, index)
        annotation['height'], annotation['width'], annotation['depth'] = image.shape
        annotation['file'] = '%s/JPEGImages/%s.jpg' % (output_dir, annotation['name'])
        annotation['xml'] = '%s/Annotations/%s.xml' % (output_dir, annotation['name'])

        object_regions = []

        for bb in bbs.bounding_boxes:
            region = dict()
            region['xmin'] = bb.x1
            region['ymin'] = bb.y1
            region['xmax'] = bb.x2
            region['ymax'] = bb.y2
            region['store_num'] = bb.store
            object_regions.append(region)

        annotation['regions'] = object_regions

        # print('Writing image %s' % annotation['file'])

        misc.imsave(annotation['file'], arr=image)

        with open(annotation['xml'], mode='w') as annotation_xml:
            annotation_xml.writelines(
                annotations_template % (annotation['name'],
                                        annotation['width'],
                                        annotation['height'],
                                        annotation['depth']))
            for region in annotation['regions']:
                name = "landmark" if binary else region['store_num']
                annotation_xml.writelines(
                    object_template % (name,
                                       region['xmin'], region['ymin'],
                                       region['xmax'], region['ymax'])
                )

            annotation_xml.writelines(annotation_final)


def convert_to_voc(dataset, output_dir, is_augment=False, binary=False):
    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        print(e)

    os.mkdir(output_dir)
    os.mkdir("%s/JPEGImages" % output_dir)
    os.mkdir("%s/Annotations" % output_dir)

    bar = ProgressBar(total=len(dataset.images))

    for image in dataset.images:
        bar.move()
        bar.log("Converting image %s to voc format..." % image.image_path)
        images_list, bounding_box = data_augment.argument(image, num_per_resolution=3, is_augment=is_augment)
        format_annotation(output_dir, image, images_list, bounding_box, binary)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import warnings
import random

from convert_to_voc import convert_to_voc


class IMSLPosition:
    def __init__(self, x, y, scale):
        self._x = x
        self._y = y
        self._scale = scale

    def __str__(self):
        return "(x: %f, y: %f, scale: %f)" % (self._x, self._y, self._scale)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def scale(self):
        return self._scale


class IMSLLandmark:
    def __init__(self, id, name, loc_x, loc_y, scale):
        self._id = id
        self._name = name
        self._pos = IMSLPosition(loc_x, loc_y, scale)

    def __str__(self):
        return "(id: %d, name: %s, pos: %s)" % (self._id, self._name, str(self._pos))

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def position(self):
        return self._pos


class IMSLMapModel:
    def __init__(self, start_x, start_y, end_x, end_y, scale):
        self._start = IMSLPosition(start_x, start_y, scale)
        self._end = IMSLPosition(end_x, end_y, scale)

    def __str__(self):
        return "(start: %s, end: %s)" % (str(self._start), str(self._end))


class IMSLRoI:
    def __init__(self, id, xmin, ymin, xmax, ymax, scale):
        self._id = id
        self._min_pos = IMSLPosition(xmin, ymin, scale=scale)
        self._max_pos = IMSLPosition(xmax, ymax, scale=scale)

    def __str__(self):
        return "(id: %d, min: %s, max: %s)" % (self._id, str(self._min_pos), str(self._max_pos))

    @property
    def id(self):
        return self._id

    @property
    def xmin(self):
        return self._min_pos.x

    @property
    def ymin(self):
        return self._min_pos.y

    @property
    def xmax(self):
        return self._max_pos.x

    @property
    def ymax(self):
        return self._max_pos.y


class IMSLImage:
    def __init__(self, image_path, width, height, depth, object_num, objects):
        self._image_path = image_path
        self._width = width
        self._height = height
        self._depth = depth
        self._objects = objects
        if not (len(self._objects) == object_num):
            warnings.warn("Object number mismatch", ResourceWarning)

    @property
    def name(self):
        return os.path.basename(self.image_path).split(".")[-2]

    @property
    def image_path(self):
        return self._image_path

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def depth(self):
        return self._depth

    @property
    def objects(self):
        return self._objects

    @property
    def object_num(self):
        return len(self.objects)


class IMSLMap:
    def __init__(self, id, name, width, height, scale,
                 landmark_num, model_num, landmarks, mapmodels):
        self._id = id
        self._name = name
        self._width = width
        self._height = height
        self._scale = scale

        self._landmarks = landmarks
        self._map_models = mapmodels

        if not (len(landmarks) == landmark_num
                and len(mapmodels) == model_num):
            warnings.warn("Landmark or Model number mismatch", ResourceWarning)

    @property
    def scale(self):
        return self._scale

    @property
    def id(self):
        return self._id

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def map_models(self):
        return self._map_models

    @property
    def landmark_num(self):
        return len(self.landmarks)

    @property
    def model_num(self):
        return len(self.map_models)


class IMSLDataset:
    def __init__(self, name, root_path, map_dir, landmark_dir):
        self._name = name
        self._maps = []
        self._images = []
        try:
            self.read_map(root_path, map_dir)
            self.read_landmarks(root_path, landmark_dir)
        except Exception as e:
            print(e)

    @property
    def name(self):
        return self._name

    @property
    def maps(self):
        return self._maps

    @property
    def images(self):
        return self._images

    def read_map(self, root_path, map_dir):
        map_json_file = os.path.join(root_path, self._name, map_dir, "map_info.json")

        with open(map_json_file, 'r', encoding="utf-8") as map_f:
            map_info = json.load(map_f)
            landmark_list = []
            for landmark in map_info['landmarks']:
                imsl_landmark = IMSLLandmark(id=landmark['id'],
                                             name=landmark['name'],
                                             loc_x=landmark['x'],
                                             loc_y=landmark['y'],
                                             scale=map_info['map']['scale'])
                landmark_list.append(imsl_landmark)

            model_list = []
            for model in map_info['map_models']:
                imsl_mapmodel = IMSLMapModel(start_x=model['start_x'],
                                             start_y=model['start_y'],
                                             end_x=model['end_x'],
                                             end_y=model['end_y'],
                                             scale=map_info['map']['scale'])
                model_list.append(imsl_mapmodel)

            map_info = IMSLMap(id=map_info['id'],
                               name=map_info['name'],
                               width=map_info['map']['width'],
                               height=map_info['map']['height'],
                               scale=map_info['map']['scale'],
                               landmark_num=map_info['map']['landmark_num'],
                               model_num=map_info['map']['model_num'],
                               landmarks=landmark_list,
                               mapmodels=model_list)
            self._maps.append(map_info)

    def read_landmarks(self, root_path, landmark_dir):
        json_file = os.path.join(root_path, self._name, landmark_dir, "info.json")

        with open(json_file, 'r') as f:
            info = json.load(f)
            map_info = self.get_map_from_id(info['map_id'])

            image_list = info['image_list']
            if not len(image_list) == info['image_num']:
                warnings.warn("Image number mismatch", ResourceWarning)

            imsl_image_list = []

            for image in image_list:
                anno_json_file = os.path.join(root_path, self._name, landmark_dir,
                                              "Annotations", "%s.json" % image)
                with open(anno_json_file, 'r') as anno_f:
                    anno_json = json.load(anno_f)
                    object_list = []
                    if not len(anno_json['objects']) == anno_json['object_num']:
                        warnings.warn("object number mismatch", ResourceWarning)

                    for obj in anno_json['objects']:
                        imsl_obj = IMSLRoI(id=obj['id'], xmin=obj['bndbox']['xmin'],
                                           ymin=obj['bndbox']['ymin'],
                                           xmax=obj['bndbox']['xmax'],
                                           ymax=obj['bndbox']['ymax'],
                                           scale=map_info.scale)
                        object_list.append(imsl_obj)

                    imsl_image = IMSLImage(image_path=os.path.join(root_path, self.name, landmark_dir,
                                                                   "JPEGImages", anno_json['image_name']),
                                           width=anno_json['image_size']['width'],
                                           height=anno_json['image_size']['height'],
                                           depth=anno_json['image_size']['depth'],
                                           object_num=len(object_list),
                                           objects=object_list)
                    imsl_image_list.append(imsl_image)

            self._images.extend(imsl_image_list)

    def get_map_from_id(self, map_id):
        for m in self._maps:
            if m.id == map_id:
                return m
        warnings.warn("Not found map_id: %d" % map_id, RuntimeWarning)
        return None

    def get_landmark_from_id(self, landmark_id):
        for m in self._maps:
            for l in m.landmarks:
                if l.id == landmark_id:
                    return l
        warnings.warn("Not found landmark_id: %d" % landmark_id, RuntimeWarning)
        return None

    def combine(self, other, newname):
        self._name = newname
        self._maps.extend(other.maps)
        self._images.extend(other.images)

    def get_landmark_list(self):
        landmark_list = []
        for m in self._maps:
            for l in m.landmarks:
                landmark_list.append(str(l.id) + " " + str(l.name))
        return landmark_list

    def get_landmark_map(self):
        landmark_map = dict()
        for m in self._maps:
            for l in m.landmarks:
                landmark_map[l.id] = str(l.name)
        return landmark_map

    def extract_images_from_id(self, id):
        image_list = []
        for image in self.images:
            obj_list = []
            for obj in image.objects:
                if obj.id == str(id):
                    obj_list.append(obj)
            if not len(obj_list) == 0:
                im = IMSLImage(image.image_path, image.width,
                               image.height, image.depth,
                               len(obj_list), obj_list)
                image_list.append(im)

        return image_list

    def select_sub_dataset(self, selected_landmark, newname):
        self._name = newname
        new_imsl_image_list = []

        for image in self._images:
            object_list = []
            for obj in image.objects:
                if str(obj.id) in selected_landmark:
                    object_list.append(obj)

            if not len(object_list) == 0:
                imsl_image = IMSLImage(image_path=image.image_path,
                                       width=image.width,
                                       height=image.height,
                                       depth=image.depth,
                                       object_num=len(object_list),
                                       objects=object_list)
                new_imsl_image_list.append(imsl_image)
        self._images.clear()
        self._images.extend(new_imsl_image_list)


if __name__ == '__main__':
    gogo_dataset = IMSLDataset('newgogo', '/home/huangjianjun/LandmarkData/Format', 'map', 'landmark_1')
    # gogo_dataset = IMSLDataset('oldgogo', '/home/huangjianjun/LandmarkData/Format', 'map', 'landmark_1')
    # zhengjia_dataset = IMSLDataset('zhengjia', '/home/huangjianjun/LandmarkData/Format', 'map', 'landmark_1')
    # gogo_dataset.combine(zhengjia_dataset, "oldgogo_and_zhengjia")


    # select some landmarks and write their ids to file
    # also write the not-selected landmarks' ids to another file
    #selected_landmarks_file = "/mnt/UserData/Mingkuan/Public/VOCdevkit/newgogo_selected_landmarks.txt"
    #not_selected_landmarks_file = "/mnt/UserData/Mingkuan/Public/VOCdevkit/newgogo_not_selected_landmarks.txt"

    # selected_number = int(len(gogo_dataset.get_landmark_list()) * 3 / 4 )
    # print(selected_number)
    # with open(selected_landmarks_file, "w") as sf, open(not_selected_landmarks_file, "w") as nsf:
    #     selected = random.sample(gogo_dataset.get_landmark_list(), selected_number)
    #
    #     for lm in gogo_dataset.get_landmark_list():
    #         id = lm.split()[0]
    #         if lm in selected:
    #             sf.write(str(id) + "\n")
    #         else:
    #             nsf.write(str(id) + "\n")


    #landmark = []
    #with open(not_selected_landmarks_file, "r") as f:
     #   landmark.extend([line.strip() for line in f.readlines()])


    #gogo_dataset.select_sub_dataset(landmark, "sub_gogo")
    # a = gogo_dataset.get_landmark_list()
    # for ai in a:
    #     print(ai)

    # image_list = gogo_dataset.extract_images_from_id(id=1001)

    print(gogo_dataset.get_landmark_list())
    with open("newgogo_landmark_list.txt", "w", encoding="utf-8") as f:
        for landmark in gogo_dataset.get_landmark_list():
            f.write(landmark + "\n")

    # convert_to_voc(gogo_dataset, '/home/huangjianjun/LandmarkData/VOCdevkit/VOC2007_new_gogo_detailed_id',
    #                is_augment=True, binary=False)

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
import random
import cv2
import json
import codecs

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from typing import Dict, List

def read_voc(root_path):
    """
    :param root_path:
    :return: images [image : dict]
    image :
    {
        "path" :
        "width" :
        "height" :
        "objects" : [object : dict]
    }
    """
    annotations_path = os.path.join(root_path, "Annotations")
    jpegimages_path = os.path.join(root_path, "JPEGImages")

    images = []

    for root, dirs, files in os.walk(annotations_path):
        for file in files:
            if int(file.split('-')[-1].split('.')[0]) != 0:
                continue
            image = dict()
            image["path"] = os.path.join(jpegimages_path, "%s.jpg" % file.split(".")[0])
            tree = ET.parse(os.path.join(annotations_path, file))
            root = tree.getroot()
            s = root.find("size")
            image["width"] = int(s.find("width").text)
            image["height"] = int(s.find("height").text)
            objs = root.findall("object")
            objs_list = []
            for obj in objs:
                bbox = obj.find("bndbox")
                obj_ = dict()
                obj_["xmin"] = int(bbox.find("xmin").text)
                obj_["ymin"] = int(bbox.find("ymin").text)
                obj_["xmax"] = int(bbox.find("xmax").text)
                obj_["ymax"] = int(bbox.find("ymax").text)
                obj_["name"] = obj.find("name")
                objs_list.append(obj_)
            image["objects"] = objs_list

            images.append(image)
    return images

def read_landmark_detail(file, frames_dict, vedio_id):
    with open(file, "r") as f:
        for line in f.readlines():
            line.strip()
            elements = line.split()
            bndbox = {}
            bndbox["xmin"] = int(elements[2])
            bndbox["ymin"] = int(elements[3])
            bndbox["xmax"] = int(elements[4])
            bndbox["ymax"] = int(elements[5])
            id = vedio_id + '_' + elements[1]
            if id not in frames_dict:
                frames_dict[id] = []
            if bndbox not in frames_dict[id]:
                frames_dict[id].append(bndbox)

def get_frames_objects_dict(video_label_path, target_file):
    frames_dict = {}

    for root, dirs, files in os.walk(video_label_path):
        if target_file in files:
            read_landmark_detail(os.path.join(root, target_file), frames_dict, root.split('/')[-3])

    return frames_dict

def convert_to_dataset(frames_dict):
    ROOT_PATH = "/mnt/UserData/Mingkuan/Public/Format/videos/newgogo"
    images = []

    keys = list(frames_dict.keys())
    keys.sort()
    filtered_keys = keys[::10]

    for k in filtered_keys:
        vedio_id = k.split('_')[0]
        img_file_name = k.split('_')[1] + ".jpg"

        image = {}
        image["path"] = os.path.join(ROOT_PATH, vedio_id, "frames", img_file_name)
        # print(image["path"])
        im = Image.open(image["path"])
        image["width"] = int(im.size[0])
        image["height"] = int(im.size[1])
        image["objects"] = frames_dict[k]

        images.append(image)

    return images

def convert_with_frame_rate(frames_dict, frame_rate):
    keys = list(frames_dict.keys())
    keys.sort()
    filtered_keys = keys[::frame_rate]
    filtered_frames_dict = {}
    for k in filtered_keys:
        filtered_frames_dict[k] = frames_dict[k]
    return filtered_frames_dict

def load_ground_truth():
    """
    :return: ground_truths, samples, len(frames_dict) + len(newgogo_test_images)
    ground_truths : {path : [objects_num, (xmin, ymin, xmax, ymax)]}
    """
    ground_truths = {}
    samples = 0

    root_path = "/home/huangjianjun/LandmarkData/VideoLabel/output"
    target_file = "landmark_detail.txt"
    frames_dict = get_frames_objects_dict(root_path, target_file)
    frames_dict = convert_with_frame_rate(frames_dict, 10)
    ROOT_PATH = "/home/huangjianjun/LandmarkData/Format/videos/newgogo/"
    for key in frames_dict.keys():
        vedio_id = key.split('_')[0]
        img_file_name = key.split('_')[1] + ".jpg"

        path = os.path.join(ROOT_PATH, vedio_id, "frames", img_file_name)
        ground_truths[path] = frames_dict[key]
        samples += len(frames_dict[key])

    VOC_DIR = "/home/huangjianjun/LandmarkData/VOCdevkit/"
    newgogo_test_images = read_voc(VOC_DIR + "VOC2007_binary_new_gogo_not_selected")
    newgogo_test_images.extend(read_voc(VOC_DIR + "VOC2007_binary_new_gogo_selected"))
    for im in newgogo_test_images:
        ground_truths[im["path"]] = im["objects"]
        samples += len(im["objects"])

    return ground_truths, samples, len(frames_dict) + len(newgogo_test_images)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def load_rcnn_detection_result(path):
    detection_result = []
    upper_score = -1
    lower_score = 2
    with open(path, "r") as f:
        for line in f.readlines():
            line.strip()
            elements = line.split()
            roi = {}
            roi["path"] = elements[0]
            roi["score"] = float(elements[1])
            if roi["score"] > upper_score:
                upper_score = roi["score"]
            if roi["score"] < lower_score:
                lower_score = roi["score"]
            roi["xmin"] = int(elements[2])
            roi["ymin"] = int(elements[3])
            roi["xmax"] = int(elements[4])
            roi["ymax"] = int(elements[5])
            detection_result.append(roi)
    return detection_result, upper_score, lower_score

def load_rcnn_detection_result_to_dict(path):
    detection_result, upper_score, lower_score = load_rcnn_detection_result(path)
    dr_dict = dict()
    for dr in detection_result:
        if dr["path"] not in dr_dict:
            dr_dict[dr["path"]] = []

        dr_dict[dr["path"]].append([dr["score"], dr["xmin"], dr["ymin"], dr["xmax"], dr["ymax"]])

    return dr_dict, upper_score, lower_score


def compute_iou(bbx1, bbx2):
    """
    :param bbx1: [xmin ymin xmax ymax]
    :param bbx2: [xmin ymin xmax ymax]
    :return: iou
    """
    x1 = np.maximum(bbx1[0], bbx2[0])
    y1 = np.maximum(bbx1[1], bbx2[1])
    x2 = np.minimum(bbx1[2], bbx2[2])
    y2 = np.minimum(bbx1[3], bbx2[3])
    intersection = np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0)
    area1 = (bbx1[2] - bbx1[0] + 1) * (bbx1[3] - bbx1[1] + 1)
    area2 = (bbx2[2] - bbx2[0] + 1) * (bbx2[3] - bbx2[1] + 1)
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

# def compute_match(target, ground_truths, iou_threshold):
#     for gt in ground_truths[target["path"]]:
#         iou = compute_iou(np.array([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]),
#                        np.array([target["xmin"], target["ymin"], target["xmax"], target["ymax"]]))
#         print("iou : %f" % iou)
#         if iou > iou_threshold:
#             return 1
#
#     return 0

# def compute_match(target, ground_truths, iou_thresholds):
#     iou_thresholds = np.array(iou_thresholds)
#     for gt in ground_truths[target["path"]]:
#         iou = compute_iou(np.array([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]),
#                        np.array([target["xmin"], target["ymin"], target["xmax"], target["ymax"]]))
#         matches = (iou_thresholds < iou).astype(int)
#         if np.sum(matches) > 0:
#             return matches
#
#     return np.zeros(iou_thresholds.shape)

def compute_match(target, ground_truths, iou_threshold):
    for gt in ground_truths[target["path"]]:
        iou = compute_iou(np.array([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]),
                       np.array([target["xmin"], target["ymin"], target["xmax"], target["ymax"]]))
        if iou > iou_threshold:
            return True

    return False

def compute_match_id(target, ground_truths, id_dict, iou_threshold):
    if target["path"] == "/home/huangjianjun/LandmarkData/VOCdevkit/VOC2007_binary_new_gogo_selected/JPEGImages/GOPR0859-0.jpg":
        print([target["xmin"], target["ymin"], target["xmax"], target["ymax"]])
        print(ground_truths[target["path"]])

    for gt in ground_truths[target["path"]]:
        iou = compute_iou(np.array([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]),
                       np.array([target["xmin"], target["ymin"], target["xmax"], target["ymax"]]))
        if target[
            "path"] == "/home/huangjianjun/LandmarkData/VOCdevkit/VOC2007_binary_new_gogo_selected/JPEGImages/GOPR0859-0.jpg":
            print("gt")
            print(gt)
            print(iou)
        if iou > iou_threshold:
            _, name = os.path.split(target["path"])
            dict_key = os.path.join("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_1/Annotations/", name)
            if name == "GOPR0859-0.jpg":
                print(id_dict[dict_key])
            for b in id_dict[dict_key]:
                if target["path"] == "/home/huangjianjun/LandmarkData/VOCdevkit/VOC2007_binary_new_gogo_selected/JPEGImages/GOPR0859-0.jpg":
                    print("b")
                    print(b)
                if b["xmin"] == gt["xmin"] and b["ymin"] == gt["ymin"] and b["xmax"] == gt["xmax"] and b["ymax"] == gt["ymax"]:
                    return b["id"]
            return "Error"
    return -1

def voc_ap(rec, prec, use_07_metric=True):
    """
    :param rec: increasing
    :param prec:
    :param use_07_metric:
    :return:
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # max operation is to make sure the precision is not decreasing
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_ap_score_thres():
    ground_truths, samples_num, images_num = load_ground_truth()
    faster_rcnn_detection_result, f_upper_score, f_lower_score = load_rcnn_detection_result(
        "/home/huangjianjun/Faster_RCNN_new/output/detect_result_allnewgogo.txt")
    mask_rcnn_detection_result, m_upper_score, m_lower_score = load_rcnn_detection_result(
        "/home/huangjianjun/Mask_RCNN-master/mrcnn_detect_results_allnewgogo.txt")


    score_thresholds = np.array([i * 0.00001 + 0.0009 for i in range(10)])  # [0.0, 0.1 ... 0.9]
    f_det = np.zeros(score_thresholds.size)
    f_tp = np.zeros(score_thresholds.size)
    for i, dr in enumerate(faster_rcnn_detection_result):
        print("%d/%d" % (i, len(faster_rcnn_detection_result)))
        f_det += (score_thresholds < dr["score"]).astype(int)
        if compute_match(dr, ground_truths, 0.5) == 0:
            f_tp += np.zeros(score_thresholds.size)
        else:
            print("score : %f" % dr["score"])
            print((score_thresholds < dr["score"]).astype(int))
            f_tp += (score_thresholds < dr["score"]).astype(int)

        print(f_tp)

    m_tp = np.zeros(score_thresholds.size)
    m_det = np.zeros(score_thresholds.size)
    for i, dr in enumerate(mask_rcnn_detection_result):
        # print("dr")
        # print(dr)
        # print("gt")
        # print(ground_truths[dr["path"]])
        print("%d/%d" % (i, len(mask_rcnn_detection_result)))
        m_det += (score_thresholds < dr["score"]).astype(int)
        t = dr["xmin"]
        dr["xmin"] = dr["ymin"]
        dr["ymin"] = t
        t = dr["xmax"]
        dr["xmax"] = dr["ymax"]
        dr["ymax"] = t
        if compute_match(dr, ground_truths, 0.5) == 0:
            m_tp += np.zeros(score_thresholds.size)
        else:
            print("score : %f" % dr["score"])
            print((score_thresholds < dr["score"]).astype(int))
            m_tp += (score_thresholds < dr["score"]).astype(int)

        print(m_tp)

        # view the image
        # # load the image
        # print(dr["path"])
        # image = cv2.imread(dr["path"])
        #
        # # draw the ground-truth bounding box along with the predicted
        # # bounding box
        # # green dr
        # cv2.rectangle(image, tuple([dr["xmin"], dr["ymin"]]),
        #               tuple([dr["xmax"], dr["ymax"]]), (0, 255, 0), 2)
        #
        # for gt in ground_truths[dr["path"]]:
        #     cv2.rectangle(image, tuple([gt["xmin"], gt["ymin"]]),
        #               tuple([gt["xmax"], gt["ymax"]]), (0, 0, 255), 2)
        #
        # # compute the intersection over union and display it
        # # iou = bb_intersection_over_union(detection.gt, detection.pred)
        # # cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
        # #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # # print("{}: {:.4f}".format(detection.image_path, iou))
        #
        # # show the output image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)

    print("there are %d images" % images_num)
    print("total bounding boxes' num: %d" % samples_num)
    print("faster rcnn detection result info:")
    print("len : %d" % len(faster_rcnn_detection_result))
    print("upper_score : %f" % f_upper_score)
    print("lower_score : %f" % f_lower_score)
    print("mask rcnn detection result info:")
    print("len : %d" % len(mask_rcnn_detection_result))
    print("upper_score : %f" % m_upper_score)
    print("lower_score : %f" % m_lower_score)
    print(score_thresholds)


    f_recall = f_tp / samples_num
    f_precision = f_tp / f_det
    f_ap = voc_ap(f_recall, f_precision, True)
    print("f_tp")
    print(f_tp)
    print("f_det")
    print(f_det)
    print("faster rcnn recall:")
    print(f_recall)
    print("faster rcnn precision:")
    print(f_precision)
    print("2010 before faster rcnn ap:")
    print(f_ap)
    print("2010 after faster rcnn ap:")
    f_ap = voc_ap(f_recall, f_precision, False)
    print(f_ap)

    m_recall = m_tp / samples_num
    m_precision = m_tp / m_det
    m_ap = voc_ap(m_recall, m_precision)
    print("m_tp")
    print(m_tp)
    print("m_det")
    print(m_det)
    print("mask rcnn recall:")
    print(m_recall)
    print("mask rcnn precision:")
    print(m_precision)
    print("2010 before mask rcnn ap:")
    print(m_ap)
    print("2010 after mask rcnn ap:")
    m_ap = voc_ap(m_recall, m_precision, False)
    print(m_ap)

def compute_ap_iou_thres():
    ground_truths, samples_num, images_num = load_ground_truth()
    faster_rcnn_detection_result, f_upper_score, f_lower_score = load_rcnn_detection_result(
        "/home/huangjianjun/Faster_RCNN_new/output/detect_result_allnewgogo.txt")
    mask_rcnn_detection_result, m_upper_score, m_lower_score = load_rcnn_detection_result(
        "/home/huangjianjun/Mask_RCNN-master/mrcnn_detect_results_allnewgogo.txt")

    iou_thresholds = np.array([i * 0.05 + 0.5 for i in range(10)])  # [0.5, 0.55 ... 0.95]
    score_threshold = 0.0009
    print("iou_thresholds")
    print(iou_thresholds)

    # faster rcnn
    f_det = 0
    f_tp = np.zeros(iou_thresholds.size)
    for i, dr in enumerate(faster_rcnn_detection_result):
        print("%d/%d" % (i, len(faster_rcnn_detection_result)))
        if dr["score"] < score_threshold:
            continue
        f_det += 1
        f_tp += compute_match(dr, ground_truths, iou_thresholds)
        print(f_tp)

    # mask rcnn
    m_det = 0
    m_tp = np.zeros(iou_thresholds.size)
    for i, dr in enumerate(mask_rcnn_detection_result):
        print("%d/%d" % (i, len(mask_rcnn_detection_result)))
        if dr["score"] < score_threshold:
            continue
        t = dr["xmin"]
        dr["xmin"] = dr["ymin"]
        dr["ymin"] = t
        t = dr["xmax"]
        dr["xmax"] = dr["ymax"]
        dr["ymax"] = t
        m_det += 1
        m_tp += compute_match(dr, ground_truths, iou_thresholds)
        print(m_tp)

    print("there are %d images" % images_num)
    print("total bounding boxes' num: %d" % samples_num)
    print("faster rcnn detection result info:")
    print("len : %d" % len(faster_rcnn_detection_result))
    print("upper_score : %f" % f_upper_score)
    print("lower_score : %f" % f_lower_score)
    print("mask rcnn detection result info:")
    print("len : %d" % len(mask_rcnn_detection_result))
    print("upper_score : %f" % m_upper_score)
    print("lower_score : %f" % m_lower_score)

    f_recall = f_tp / samples_num
    f_precision = f_tp / f_det
    f_ap = voc_ap(f_recall, f_precision, True)
    print("f_tp")
    print(f_tp)
    print("f_det")
    print(f_det)
    print("faster rcnn recall:")
    print(f_recall)
    print("faster rcnn precision:")
    print(f_precision)
    print("2010 before faster rcnn ap:")
    print(f_ap)
    print("2010 after faster rcnn ap:")
    f_ap = voc_ap(f_recall, f_precision, False)
    print(f_ap)

    m_recall = m_tp / samples_num
    m_precision = m_tp / m_det
    m_ap = voc_ap(m_recall, m_precision)
    print("m_tp")
    print(m_tp)
    print("m_det")
    print(m_det)
    print("mask rcnn recall:")
    print(m_recall)
    print("mask rcnn precision:")
    print(m_precision)
    print("2010 before mask rcnn ap:")
    print(m_ap)
    print("2010 after mask rcnn ap:")
    m_ap = voc_ap(m_recall, m_precision, False)
    print(m_ap)

def plot_fm(f_data, m_data, x_axis):
    plt.plot(x_axis, f_data, color='green', label="faster rcnn")
    plt.plot(x_axis, m_data, color='red', label="mask rcnn")
    plt.ylabel("Precision")
    plt.xlabel("score tresholds")
    plt.legend()
    plt.savefig("pre_score_thre_1.jpg")
    plt.show()

def compute_ap(rcnn_detection_result, iou_threshold, ground_truths):
    y_truths = []
    y_scores = []

    for i, dr in enumerate(rcnn_detection_result):
        print("%d/%d" % (i, len(rcnn_detection_result)))
        # all new gogo mask RCNN should use nest code
        # My Miss
        # t = dr["xmin"]
        # dr["xmin"] = dr["ymin"]
        # dr["ymin"] = t
        # t = dr["xmax"]
        # dr["xmax"] = dr["ymax"]
        # dr["ymax"] = t
        y_truths.append(int(compute_match(dr, ground_truths, iou_threshold)))
        y_scores.append(dr["score"])

    return y_truths, y_scores

def plt_pr_curve(precision, recall, average_precision = -1, if_save = False):
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                    color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    if average_precision > -1:
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    if if_save:
        plt.savefig("PR_curve_f_all.jpg")

    plt.show()

def plt_something():
    ground_truths, samples_num, images_num = load_ground_truth()
    faster_rcnn_detection_result, f_upper_score, f_lower_score = load_rcnn_detection_result(
        "/home/huangjianjun/Faster_RCNN_new/output/detect_result_allnewgogo.txt")
    mask_rcnn_detection_result, m_upper_score, m_lower_score = load_rcnn_detection_result(
        "/home/huangjianjun/Mask_RCNN-master/mrcnn_detect_results_allnewgogo.txt")

    y_truths, y_scores = compute_ap(faster_rcnn_detection_result, 0.5, ground_truths)
    print(y_truths)
    print(y_scores)
    precision, recall, thres = precision_recall_curve(y_truths, y_scores)
    print(precision)
    print(recall)
    print(thres)
    average_precision = average_precision_score(y_truths, y_scores)
    print(average_precision)
    plt_pr_curve(precision, recall, average_precision, True)
    recall = recall[::-1]
    precision = precision[::-1]
    print("new:")
    print(voc_ap(recall, precision, False))
    print("is voc 07")
    print(voc_ap(recall, precision, True))

def plt_image(image, gts = None, drs = None, scores = None, file_name =None):
    """
    :param image:
    :param gt:
    :param dr: [num_instance, (x1, y1, x2, y2)] in image coordinates.
    :param scores: [num_instance, (score)]
    :return:
    """
    gt_color = (0, 0, 255)
    dr_color = (0, 255, 0)
    score_color = (255, 0, 0)
    thickness = 3

    image_shape = image.shape
    height = image_shape[0]
    width = image_shape[1]

    # if gts:
    #     for gt in gts:
    #         cv2.rectangle(image, (gt["xmin"], gt["ymin"]), (gt["xmax"], gt["ymax"]), gt_color, thickness=thickness)
    if gts:
        for gt in gts:
            cv2.rectangle(image, (gt[0], gt[1]), (gt[2], gt[3]), gt_color, thickness=thickness)

    if drs:
        for i, dr in enumerate(drs):
            x1, y1, x2, y2 = dr
            cv2.rectangle(image, (x1, y1), (x2, y2), dr_color, thickness=thickness)

            # Label
            cv2.putText(image, "{:.3f}".format(scores[i]), (x1, y1 + 25), cv2.FONT_HERSHEY_COMPLEX, 1, score_color)

    if file_name:
        cv2.imwrite(file_name, image)
    # cv2.imshow("Tm", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # fig, ax = plt.subplots(1, figsize=(width, height))
    # ax = plt.axes()
    # fig = plt.figure(figsize=(width, height), dpi=1)
    # ax = fig.add_axes([0,0, width, height])
    # ax.axis('off')
    # ax.imshow(image)
    #
    # for gt in gts:
    #     p = patches.Rectangle((gt["xmin"], gt["ymin"]), gt["xmax"] - gt["xmin"], gt["ymax"] - gt["ymin"], linewidth=2,
    #                       edgecolor=gt_color, facecolor='none')
    #     ax.add_patch(p)
    #
    # for i, dr in enumerate(drs):
    #     x1, y1, x2, y2 = dr
    #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7,
    #                       edgecolor=dr_color, facecolor='none')
    #
    #     # Label
    #     caption = "{:.3f}".format(scores[i])
    #     ax.text(x1, y1 - 5, caption,
    #             color='w', size=11, backgroundcolor="none")
    #     ax.add_patch(p)
    #
    # if file_name:
    #     plt.savefig("result_images/%s" % file_name, bbox_inches ="tight")
    # plt.show()

def filter_dr_with_score_thres(drs, score_thres):
    res = []
    for dr in drs:
        score = dr[0]
        if score > score_thres:
            res.append(dr)
    return  res

def build_result_images():
    """
    if there is Key Error, please run again
    Cuz not all images in frcnn have detection in mrcnn
    image with no detection result will not be writen into file
    """
    ground_truths, samples_num, images_num = load_ground_truth()
    mask_rcnn_detection_result, m_upper_score, m_lower_score = load_rcnn_detection_result_to_dict(
        "/home/huangjianjun/Mask_RCNN-master/mrcnn_detect_results.txt")
    faster_rcnn_detection_result, f_upper_score, f_lower_score = load_rcnn_detection_result_to_dict(
        "/home/huangjianjun/Faster_RCNN_new/output/detect_result.txt")

    mrcnn_to_build_images_list = random.sample(list(mask_rcnn_detection_result.items()), 20)
    frcnn_to_build_images_list = []

    for i in mrcnn_to_build_images_list:
        frcnn_to_build_images_list.append(faster_rcnn_detection_result[i[0]])

    for i, im in enumerate(mrcnn_to_build_images_list):
        path = im[0]
        gts = ground_truths[path]
        m_drs = im[1]
        plt_image(cv2.imread(path), gts, [[dr[1], dr[2], dr[3], dr[4]] for dr in m_drs], [dr[0] for dr in m_drs],
                  "mrcnn_result_images/" + str(i) + ".jpg")

        f_drs = frcnn_to_build_images_list[i]
        f_drs = filter_dr_with_score_thres(f_drs, 0.0012)
        plt_image(cv2.imread(path), gts, [[dr[1], dr[2], dr[3], dr[4]] for dr in f_drs], [dr[0] for dr in f_drs],
                  "frcnn_result_images/" + str(i) + ".jpg")

def build_gt_images_with_id(id):
    path = os.path.join("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_2/Landmarks/", str(id), "landmark.json")
    images = None
    with codecs.open(path, "r", "utf-8") as json_file:
        j = json.load(json_file)
        images = j["images"]

    for im in images:
        im_path = os.path.join("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_1/JPEGImages/", im["image_name"])
        if not os.path.exists(str(id)):
            os.mkdir(str(id))
        output_path = os.path.join(str(id), im["image_name"])
        plt_image(cv2.imread(im_path), [im["bndbox"]], None, None, output_path)

def load_ids(is_selected=True):
    path = "/home/huangjianjun/LandmarkData/VOCdevkit/"
    if is_selected:
        path = os.path.join(path, "newgogo_selected_landmarks.txt")
    else:
        path = os.path.join(path, "newgogo_not_selected_landmarks.txt")

    ids = []
    with open(path, "r") as f:
        for line in f.readlines():
            line.strip()
            ids.append(int(line))
    return ids

def build_images_with_id(id):
    print("this is id : %d" % id)
    mask_rcnn_detection_result_dict, _, __ = load_rcnn_detection_result_to_dict(
        "/home/huangjianjun/Mask_RCNN-master/mrcnn_detect_results_allnewgogo.txt")
    path = os.path.join("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_2/Landmarks/", str(id),
                        "landmark.json")
    selected_ids = load_ids(is_selected=True)
    not_selected_ids = load_ids(is_selected=False)

    key = "/home/huangjianjun/LandmarkData/VOCdevkit/"
    if id in selected_ids:
        key = os.path.join(key, "VOC2007_binary_new_gogo_selected", "JPEGImages")
    elif id in not_selected_ids:
        key = os.path.join(key, "VOC2007_binary_new_gogo_not_selected", "JPEGImages")

    images = None
    with codecs.open(path, "r", "utf-8") as json_file:
        j = json.load(json_file)
        images = j["images"]

    tp = 0
    for im in images:
        im_path = os.path.join("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_1/JPEGImages/",
                               im["image_name"])
        output_path = str(id)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        fname, fename = os.path.splitext(im["image_name"])
        im_name = fname + "-0" + fename
        output_path = os.path.join(output_path, im["image_name"])
        t_key = os.path.join(key, im_name)
        if t_key in mask_rcnn_detection_result_dict:
            m_drs = mask_rcnn_detection_result_dict[t_key]
            plt_image(cv2.imread(im_path), [im["bndbox"]], [[dr[1], dr[2], dr[3], dr[4]] for dr in m_drs],
                      [dr[0] for dr in m_drs], output_path)

            for dr in m_drs:
                gt = im["bndbox"]
                if compute_iou([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]], [dr[1], dr[2], dr[3], dr[4]]) > 0.5:
                    tp += 1
        else:
            plt_image(cv2.imread(im_path), [im["bndbox"]], None,
                      None, output_path)
    print("its tp is %d" % tp)

def load_landmark_ids_files(path):
    images = os.listdir(path)
    res = dict()
    for im in images:
        key = path + im
        with open(key) as f:
            annotation = json.load(f)
            name, _ = os.path.splitext(im)
            name += "-0.jpg"
            key = os.path.join(path, name)
            if key not in res:
                res[key] = []
            for bbox in annotation["objects"]:
                b = dict()
                b["id"] = int(bbox["id"])
                b["xmin"] = bbox["bndbox"]["xmin"]
                b["xmax"] = bbox["bndbox"]["xmax"]
                b["ymin"] = bbox["bndbox"]["ymin"]
                b["ymax"] = bbox["bndbox"]["ymax"]
                res[key].append(b)
    return res

def count_sth():
    ground_truths, samples_num, images_num = load_ground_truth()
    image_to_bbox_and_ids = load_landmark_ids_files("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_1/Annotations/")
    # print(list(image_to_bbox_and_ids.items()))
    mask_rcnn_detection_result, _, __ = load_rcnn_detection_result(
        "/home/huangjianjun/Mask_RCNN-master/mrcnn_detect_results_allnewgogo.txt")
    res = dict()

    with codecs.open("/home/huangjianjun/LandmarkData/VideoLabel/landmark_list.txt", "r", "utf-8") as f:
        for line in f.readlines():
            line.strip()
            line = line[:-1] #remove '\n'
            segments = line.split(" ", 1)
            dic = dict()
            dic["name"] = segments[1]
            res[int(segments[0])] = dic

    for root, dirs, files in os.walk("/home/huangjianjun/LandmarkData/Format/newgogo/landmark_2/Landmarks"):
        if "landmark.json" in files:
            path = os.path.join(root, "landmark.json")
            with codecs.open(path, "r", "utf-8") as f:
                j = json.load(f)
                res[int(j["landmark_id"])]["images_num"] = j["images_num"]
                res[int(j["landmark_id"])]["bbox_num"] = j["images_num"]
                res[int(j["landmark_id"])]["tp"] = 0

    # mask rcnn
    m_det = 0
    tp = 0
    for i, dr in enumerate(mask_rcnn_detection_result):
        print("%d/%d" % (i, len(mask_rcnn_detection_result)))
        if dr["path"].startswith("/home/huangjianjun/LandmarkData/Format/videos"):
            continue
        m_det += 1
        id = compute_match_id(dr, ground_truths, image_to_bbox_and_ids, 0.5)
        if id == "Error":
            break
        elif id == -1:
            continue
        else:
            if id not in res:
                res[id] = 0
            res[id]["tp"] += 1
            tp +=1

    # print(tp)
    # print(m_det)

    # with codecs.open("sth.json", "w", "utf-8") as f:
    #     json.dump(res, f, ensure_ascii = False, indent = 4)

def draw_hist():
    """
    use matplotlib.pyplot.hist to draw a histogram
    use json.load to read an UTF-8 file
    """
    import codecs
    recalls = []
    with codecs.open("sth.json", "r", "utf-8") as f:
        j = json.load(f)
        for it in j.values():
            if "tp" in it:
                rec = it["tp"] / it["bbox_num"]
                recalls.append(rec)
    plt.hist(recalls)
    plt.xlabel("Recall")
    plt.ylabel("Frequency")
    plt.xticks([_ * 0.1 for _ in range(11)])
    plt.savefig("recall_hist.jpg")
    plt.show()

from nms.cpu_nms import cpu_nms

def watch_nms_results():
    ground_truths, samples_num, images_num = load_ground_truth()
    mask_rcnn_detection_result_dict, _, __ = load_rcnn_detection_result_to_dict(
        "/home/huangjianjun/Mask_RCNN/mrcnn_detect_results_allnewgogo.txt")
    # drs = mask_rcnn_detection_result_dict["/home/huangjianjun/LandmarkData/VOCdevkit/VOC2007_binary_new_gogo_not_selected/JPEGImages/GOPR0953-0.jpg"]

    counter = 0;
    for path, drs in mask_rcnn_detection_result_dict.items():
        print(path)
        nms_form_drs = []
        for dr in drs:
            nms_form_drs.append([dr[1], dr[2], dr[3], dr[4], dr[0]])
        nms_form_drs = np.array(nms_form_drs).astype(np.float32)

        nms_res = cpu_nms(np.array(nms_form_drs).astype(np.float32), 0.1)

        if len(nms_res) < len(drs):
            counter += 1;
            gts_dict = ground_truths[path]
            gts = []
            for gt in gts_dict:
                gts.append([gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]])

            dr_show = []
            scores = []
            for dr in drs:
                dr_show.append([dr[1], dr[2], dr[3], dr[4]])
                scores.append(dr[0])

            nms_drs = [[int(dr[0]), int(dr[1]), int(dr[2]), int(dr[3])] for dr in nms_form_drs[nms_res, :]]
            nms_scores = [dr[4] for dr in nms_form_drs[nms_res, :]]

            elements = path.split("/")
            plt_image(cv2.imread(path), gts, dr_show, scores, "original_results/" + elements[-3] + "_" + elements[-1])
            plt_image(cv2.imread(path), gts, nms_drs, nms_scores, "nms_results/" + elements[-3] + "_" + elements[-1])

    print("%d / %d" % (counter, len(mask_rcnn_detection_result_dict.keys())))

import landmark_video as lv


def read_faster_rcnn_output_file(path):
    output = dict()

    with open(path, "r") as f:
        for line in f.readlines():
            line.strip()
            elements = line.split()
            key = int(elements[0])
            class_id = int(elements[1])
            score = float(elements[2])
            xmin = int(elements[3])
            ymin = int(elements[4])
            xmax = int(elements[5])
            ymax = int(elements[6])

            if key not in output:
                output[key] = []
            output[key].append(lv.LandmarkDataROI(xmin, ymin, xmax, ymax, class_id, None, score))
    return output


def calculate_iou(a: lv.LandmarkDataROI, b: lv.LandmarkDataROI):
    return compute_iou([a.xmin, a.ymin, a.xmax, a.ymax], [b.xmin, b.ymin, b.xmax, b.ymax])


def calculate_precision(ground_truth: Dict[int, List[lv.LandmarkDataROI]],
                        model_output: Dict[int, List[lv.LandmarkDataROI]], iou_threshold=0.5, score_threshold=0.1):
    tp = 0  # num of true positive
    total = 0  # total num of the model_output roi
    for key, roi_list in model_output.items():
        for roi in roi_list:
            if roi.score < score_threshold:
                continue

            total += 1
            if key not in ground_truth:
                continue

            for gt_roi in ground_truth[key]:
                # roi.class_id == gt_roi.class_id
                if calculate_iou(roi, gt_roi) > iou_threshold:
                    tp += 1
                    break
    print(tp)
    print(total)
    return tp / total


if __name__ == "__main__":
    model_output = read_faster_rcnn_output_file("/home/huangjianjun/Faster RCNN/experiments_output/id_newgogo/Faster_RCNN_output.txt")

    root_path = "/home/huangjianjun/LandmarkData/GoGoVideo"
    phone_name = "GoPro"
    gopro_video1 = lv.LandmarkDataVedio(video_directory_name="testcase2", phone_directory_name=phone_name,
                                        root_path=root_path,
                                        sample_rate=0.5)
    ground_truth = gopro_video1.frames_bbox

    precision = calculate_precision(ground_truth, model_output, iou_threshold=0.5, score_threshold=0.1)
    print(precision)

    # count_sth()
    # draw_hist()
    # for i in range(56):
    #     build_images_with_id(2001 + i)


    # watch_nms_results()


    # phone_name = "GoPro"
    # video_name = "testcase1"
    # with open(os.path.join("/home/huangjianjun/Faster_RCNN_new/", phone_name, video_name +"_m_no_correct.json"), "r") as f:
    #     j = json.load(f)
    #     for landmark in j:
    #         for region in landmark["regions"]:
    #             name = region["ori_frame_index"]
    #             rect = (region["rect"][1:len(region["rect"])-1]).split(",")
    #             gts = [[int(rect[0]), int(rect[1]),int(rect[2]),int(rect[3])]]
    #             plt_image(cv2.imread(
    #                 "/home/huangjianjun/LandmarkData/GoGoVideo/" + phone_name + "/" + video_name + "/front/" + name +".jpg"),
    #                 gts, None, None, "KCF_Results/" + video_name + "_" + str(landmark["keyframe_index"]) + "_" + str(region["frame_index"])
    #                                  + "_" + name + ".jpg")


#     plt_image(cv2.imread("/home/huangjianjun/LandmarkData/GoGoVideo/" + phone_name + "/" + video_name + "/front/" + "3175.jpg"),
#               [[313, 192, 410, 367],
#         [390, 14, 549, 242],
#         [346, 407, 417, 684]
# ], None, None,
#               "temp2.jpg")


    
    
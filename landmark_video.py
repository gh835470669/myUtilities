import os

class LandmarkDataROI:
    """
    a bounding box
    """
    def __init__(self, xmin, ymin, xmax, ymax, class_id, class_name, score=-1.0, image_path=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_id = class_id
        self.class_name = class_name
        self.score = score
        self.image_path = image_path


class LandmarkDataVedio:
    """
    Directory Structure:
    testcase1
    -- front : frame images .jpg
      -- [id.jpg for id in [001 to max frame id]]
    -- landmark
      -- [001 to max landmark id] this id is just from 1 to max landmark num in this vedio
        -- landmark_detail.txt :
          -- each line:  frame_id.jpg frame_id xmin ymin xmax ymax
        -- landmark_ground_truth.txt
        -- [id.jpg for id in [frames id]] these images are not frame images themselves, but the region image of the bounding box
      -- landmark.txt : store the other directory name and RCNN counter(i don't know what is this)
    """

    image_directory_name = "front"
    landmark_directory_name = "landmark"
    bbox_of_landmark_file_name = "landmark_detail.txt"
    landmark_id_file_name = "landmark_ground_truth.txt"

    video_id_base = {"newgogo": 2000, "oldgogo": 1000, "zhengjia": 3000}

    def __init__(self, video_directory_name, phone_directory_name="GoPro",
                 root_path="home/huangjianjun/LandmarkData/GoGoVideo", video_place="newgogo", sample_rate=1):
        self.__root_path = root_path
        self.__phone = phone_directory_name
        self.__name = video_directory_name

        self.__video_place = video_place

        self.sample_rate = sample_rate

        """
        what is a frame image's path
        a dict from id to image_path
        """
        self.frames_path = dict()

        """
        what bounding boxes does a frame have ?
        a dict from frame id to bounding boxes : LandmarkDataROI
        not all frame ids are in this dict, if a frame id is not in the dict, that means there is no bbox on that image
        """
        self.frames_bbox = dict()
        self.__read_frame_images_path()

        # what landmark does this video have
        self.landmark_ids = []

        # what frames does a landmark exist in
        self.landmark_frames_id = dict()
        self.__read_landmark_info()

        self.__sample()


    def images_directory(self):
        return os.path.join(self.__root_path, self.__phone, self.__name,
                            LandmarkDataVedio.image_directory_name)

    def landmarks_directory(self):
        return os.path.join(self.__root_path, self.__phone, self.__name,
                            LandmarkDataVedio.landmark_directory_name)

    def total_frames_num(self):
        return len(self.frames_path.values())

    def __read_frame_images_path(self):
        image_file_names = os.listdir(self.images_directory())

        for im_file in image_file_names:
            id = int(os.path.splitext(im_file)[0])
            self.frames_path[id] = os.path.join(self.images_directory(), im_file)

    def __read_landmark_info(self):
        # landmark_directories : ["001", "002" ... ]
        landmark_directories = os.listdir(self.landmarks_directory())
        landmark_directories = [item for item in landmark_directories if str.isdigit(item)]

        for landmark_dir in landmark_directories:
            # read the infomation of landmark_ground_truth.txt
            # contain the landmark id
            landmark_detail_file_path = os.path.join(self.landmarks_directory(), landmark_dir, LandmarkDataVedio.landmark_id_file_name)
            with open(landmark_detail_file_path, "r") as f:
                line = f.readline()
                line.strip()
                landmark_id = int(line) + LandmarkDataVedio.video_id_base[self.__video_place]
                self.landmark_ids.append(landmark_id)
                self.landmark_frames_id[landmark_id] = []

            # read the infomation of landmark_detail.txt
            with open(os.path.join(self.landmarks_directory(), landmark_dir, LandmarkDataVedio.bbox_of_landmark_file_name), "r") as f:
                for line in f.readlines():
                    line.strip()
                    elements = line.split()
                    frame_id = int(elements[1])
                    xmin = int(elements[2])
                    ymin = int(elements[3])
                    xmax = int(elements[4])
                    ymax = int(elements[5])
                    self.landmark_frames_id[landmark_id].append(frame_id)
                    if frame_id not in self.frames_bbox:
                        self.frames_bbox[frame_id] = []
                    self.frames_bbox[frame_id].append(LandmarkDataROI(xmin, ymin, xmax, ymax, landmark_id, None))

    def __sample(self):
        step_size = int((self.total_frames_num() - 1) / (self.sample_rate * self.total_frames_num() - 1))
        sampled_frame_ids = [i for i in range(1, self.total_frames_num() + 1,  step_size)]

        # sample frame_path
        new_frame_path = dict()
        for key, value in self.frames_path.items():
            if key in sampled_frame_ids:
                new_frame_path[key] = value
        self.frames_path = new_frame_path

        # sample frame_bbox
        new_frame_bbox = dict()
        for key, value in self.frames_bbox.items():
            if key in sampled_frame_ids:
                new_frame_bbox[key] = value
        self.frames_bbox = new_frame_bbox

        # sample landmark list
        new_landmark_frames_id = dict()
        for landmark_id, frames in self.landmark_frames_id.items():
            new_landmark_frames_id[landmark_id] = []
            for frame_id in frames:
                if frame_id in sampled_frame_ids:
                    new_landmark_frames_id[landmark_id].append(frame_id)
        self.landmark_frames_id = new_landmark_frames_id


def read_landmark_data_videos(root_path, phone_directory_name, video_num, sample_rate):
    """
    read a video data in Landmark Data, eg.GoGoVideo/GoPro/testcase1/
    :return:
    """
    videos = []
    for i in range(1, video_num + 1):
        video_directory_name = "testcase" + str(i)
        videos.append(LandmarkDataVedio(video_directory_name=video_directory_name,
                                        phone_directory_name=phone_directory_name,
                                        root_path=root_path,
                                        sample_rate=sample_rate))
    return videos


if __name__ == "__main__":
    root_path = "/home/huangjianjun/LandmarkData/GoGoVideo"
    phone_name = "GoPro"
    video_num = 1

    gopro_videos = read_landmark_data_videos(root_path=root_path, phone_directory_name=phone_name, video_num=video_num,
                                             sample_rate=0.5)

import os
import random

root = '/home/huangjianjun/LandmarkData/VOCdevkit/VOC2007_new_gogo_detailed_id/'
xml_file_path='Annotations'
txt_save_path='ImageSets/Main/'
trainval_percent=0.95 #trainval占整个数据集的百分比，剩下部分就是test所占百分比
train_percent=0.95 #train占trainval的百分比，剩下部分就是val所占百分比

files = os.listdir(os.path.join(root, xml_file_path))

print("total files num:")
print(len(files))

trainval_set = random.sample(files, int(len(files) * trainval_percent))
test_set = []
for file in files:
    if file not in trainval_set:
        test_set.append(file)



print("len of trainval_set")
print(len(trainval_set))
print("len of test")
print(len(test_set))

train_set = random.sample(trainval_set, int(len(trainval_set) * train_percent))
val_set = []
for file in trainval_set:
    if file not in train_set:
        val_set.append(file)

print("len of train_set")
print(len(train_set))
print("len of val_set")
print(len(val_set))

with open(os.path.join(root, txt_save_path, "test.txt"), "w") as f:
    for file in test_set:
        f.write(os.path.splitext(file)[0] + "\n")

with open(os.path.join(root, txt_save_path, "trainval.txt"), "w") as f:
    for file in trainval_set:
        f.write(os.path.splitext(file)[0] + "\n")

with open(os.path.join(root, txt_save_path, "train.txt"), "w") as f:
    for file in train_set:
        f.write(os.path.splitext(file)[0] + "\n")

with open(os.path.join(root, txt_save_path, "val.txt"), "w") as f:
    for file in val_set:
        f.write(os.path.splitext(file)[0] + "\n")


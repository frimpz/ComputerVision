import cv2
import numpy as np
from matplotlib import pyplot as plt

import CONSTANTS
import os


def frame_extraction():
    image_names = get_files(CONSTANTS.RAW_PATH + "videos")
    path = CONSTANTS.RAW_PATH + "videos/"

    for img_ct in image_names:

        video = cv2.VideoCapture(path + "/{}.mp4".format(img_ct))

        frame_dir = "data/images/{}".format(img_ct)
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        success, image = video.read()
        count = 1
        while success:
            cv2.imwrite(frame_dir + "/frame_{}.jpg".format(count), image)
            count += 1
            success, image = video.read()


def get_files(path):
    files = [file.split(".")[0] for file in os.listdir(path) if file.endswith(".mp4")]
    return files


def count_frames(path):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def read_target():
    target_dic = {}
    with open(CONSTANTS.RAW_PATH + "/gt.txt") as f:
        lines = [line.rstrip('\n').split(",") for line in f]

    for line in lines:
        if line[0] in target_dic:
            if line[1] in target_dic[line[0]]:
                target_dic[line[0]][line[1]].append(line)
            else:
                target_dic[line[0]][line[1]] = [line, ]
        else:
            target_dic[line[0]] = {line[1]: [line, ]}
    return target_dic


def get_vid(vid):
    if int(vid) < 10:
        return "00{}".format(vid)
    elif int(vid) < 100:
        return "0{}".format(vid)
    else:
        return vid


def draw_bounding_box():
    videos = read_target()
    for video in videos:
        print(video)
        _vid = get_vid(video)
        for frame in videos[video]:
            image = cv2.imread("data/images/{}/frame_{}.jpg".format(_vid, frame))
            for box in videos[video][frame]:
                cv2.rectangle(image, (int(box[3]), int(box[4])),
                              (int(box[3]) + int(box[5]),
                               int(box[4]) + int(box[6])),
                              (255, 0, 0), 4)
            bb_dir = "data/bb/{}".format(_vid)
            if not os.path.exists(bb_dir):
                os.makedirs(bb_dir)
            cv2.imwrite(bb_dir + "/frame_{}.jpg".format(frame), image)


def display_image(pth, _matplotlib=True):
    image = cv2.imread(pth)
    if _matplotlib:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
    else:
        cv2.imshow('image', image)
        cv2.waitKey(0)


def display_image_augmented(pth, data_augmentation):
    plt.figure(figsize=(10, 10))
    image = cv2.imread(pth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(3, 3, figsize=(12, 8))

    for row in range(0, 3):
        for col in range(0, 3):
            if row == 0 and col == 0:
                axs[row, col].imshow(image)
                axs[row, col].axis('off')
            else:
                augmented_image = data_augmentation(image)
                axs[row, col].imshow(augmented_image.numpy().astype("uint32"))
                axs[row, col].axis('off')
    plt.show()

# Count frames in Image
# print(count_frames(CONSTANTS.RAW_VIDEO + "/{}.mp4".format("001")))

# Extract Images
# frame_extraction()

# Apply Bounding Boxes
# draw_bounding_box()

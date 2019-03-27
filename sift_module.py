import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def capture_frame(video_name, frame_number):
    cap = cv2.VideoCapture(video_name)
    cap.set(1, frame_number)
    ret, frame = cap.read()
    return frame

def count_frames(video_name):
    cap = cv2.VideoCapture(video_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

def show():
    plt.show()

def gen_all_sift_features():
    listing = os.listdir(r'C:\Users\Arnaldo\Desktop\MoSIFT\train')
    
    for video in listing:
        video = r"C:/Users/Arnaldo/Desktop\MoSIFT/train/"+video
        number_of_frames = count_frames(video)
        all_kp = []
        all_dsc = []
        for i in range(number_of_frames-1):
            frame = capture_frame(video, i)
            gray_frame = to_gray(frame)
            gen_sift_features(gray_frame)
            frame_kp, frame_desc = gen_sift_features(gray_frame)
            all_kp.append(frame_kp)
            all_dsc.append(frame_desc)
    return all_kp, all_dsc

def gen_video_3d_points(video_name):
    number_of_frames = count_frames(video_name)
    data = []
    count_frame = 0
    for i in range(number_of_frames-1):
        frame = capture_frame(video_name, i)
        gray_frame = to_gray(frame)
        frame_kp, frame_desc = gen_sift_features(gray_frame)
        count_frame += 1
        print(count_frame)
        for j in frame_kp:
            info = [j.pt[0],j.pt[1],count_frame]
            data.append(info)       
    return data

def gen_data_set():
    listing = os.listdir(r'C:\Users\Arnaldo\Desktop\MoSIFT\dict')
    data = []
    count_frame = 0
    
    for video in listing:
        video = r"C:/Users/Arnaldo/Desktop\MoSIFT/dict/"+video
        data_video = gen_video_3d_points(video)
        data.extend(data_video)
        
    return data



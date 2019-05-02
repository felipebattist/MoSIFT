import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

def test_motion(sift_kp, sift_desc, optical_points, frame_2d_points):
    mosift_kp = []
    mosift_dsc = []
    mosift_2d_points = []
    
    for i in range(len(sift_kp)):
        distance_x = frame_2d_points[i].T[0] - optical_points[i].T[0]
        distance_y = frame_2d_points[i].T[1] - optical_points[i].T[1]
        
        if distance_x >3 or distance_y > 3:
            mosift_kp.append(sift_kp[i])
            mosift_dsc.append(sift_desc[i])
            x = int(frame_2d_points[i].T[0])
            y = int(frame_2d_points[i].T[1])
            info = [x, y]
            mosift_2d_points.append(info)
            
    return mosift_kp, mosift_dsc, mosift_2d_points
            

def gen_mosift_features(video_name):
    number_of_frames = count_frames(video_name)
    all_kp = []
    all_dsc = []
    all_3d_points = []
    
    for i in range(number_of_frames-2):
        frame_2d_points = []
        frame = capture_frame(video_name, i+1)
        old_frame = capture_frame(video_name, i)
        gray_frame = to_gray(frame)
        gray_old_frame = to_gray(old_frame)
        frame_kp, frame_desc = gen_sift_features(gray_old_frame)
        
        for j in frame_kp:
            info = [j.pt[0],j.pt[1]]
            frame_2d_points.append(info)
            
        frame_2d_points = np.array(frame_2d_points)
        frame_2d_points = np.float32(frame_2d_points[:, np.newaxis, :])
            
        optical_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_frame, frame_2d_points, None, **lk_params)
        frame_kp, frame_desc, frame_2d_points = test_motion(frame_kp, frame_desc, optical_points, frame_2d_points)
        
        for k in frame_2d_points:
            info = [k[0], k[1], i]
            all_3d_points.append(info)

        all_kp = all_kp+frame_kp
        all_dsc = all_dsc+frame_desc

        print(i)
      
    return all_kp, all_dsc, all_3d_points

def gen_data_set():
    listing = os.listdir(r'C:\Users\Arnaldo\Desktop\MoSIFT\dict')
    count_frame = 0
    
    for video in listing:
        video = r"C:/Users/Arnaldo/Desktop\MoSIFT/dict/"+video
        all_kp, all_dsc, all_3d_points = gen_mosift_features(video)
        dfdata_dict = pd.DataFrame(all_3d_points)
        dfdata_dict.to_csv('data_dict.csv',mode='a',index=False)
        
    return 0

import cv2 as cv
import util as ut
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import vbow as vb
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def capture_frame(video_name, frame_number):
    cap = cv.VideoCapture(video_name)
    cap.set(1, frame_number)
    ret, frame = cap.read()
    
    return frame

def count_frames(video_name):
    cap = cv.VideoCapture(video_name)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    return total_frames

def to_gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    
    return kp, desc

def test_motion(sift_kp, sift_desc, optical_points, frame_2d_points):
    mosift_kp = []
    mosift_dsc = []
    mosift_2d_points = []
    
    for i in range(len(sift_kp)):
        distance_x = frame_2d_points[i].T[0] - optical_points[i].T[0]
        distance_y = frame_2d_points[i].T[1] - optical_points[i].T[1]
        
        if distance_x > 1 or distance_y > 1:
            mosift_kp.append(sift_kp[i])
            mosift_dsc.append(sift_desc[i])
            x = int(frame_2d_points[i].T[0])
            y = int(frame_2d_points[i].T[1])
            info = [x, y]
            mosift_2d_points.append(info)
            
    return mosift_kp, mosift_dsc, mosift_2d_points

def gen_optical_dsc(x, y, gray_old_frame, gray_frame):
    descriptor = []
    neighbors = ut.gen_neighbors(x,y)
    neighbors = np.array(neighbors)
    neighbors = np.float32(neighbors[:, np.newaxis, :])
    optical_dsc = []
            
    optical_points, st, err = cv.calcOpticalFlowPyrLK(gray_old_frame, gray_frame, neighbors, None, **lk_params)
    count = 0
    flag = False
    teste = 0
    histogram = [0,0,0,0,0,0,0,0]
    for i in range(len(optical_points)):
        dx = optical_points[i].T[0]
        dy = optical_points[i].T[1]
        output = np.arctan(dy/dx)
        histogram = ut.gen_arc_hist(output, histogram)
        count+=1
        if count == 16:
            count = 0
            flag = True
            teste+=1
        if flag == True:
            optical_dsc+=histogram
            histogram = [0,0,0,0,0,0,0,0]
            flag = False
    return optical_dsc
    

def gen_mosift_features(video_name):
    number_of_frames = count_frames(video_name)
    all_kp = []
    all_dsc = []
    all_2d_points = []
    count_dsc = 0
    
    for i in range(number_of_frames-2):
    
        frame_2d_points = []
        frame = capture_frame(video_name, i+1)
        old_frame = capture_frame(video_name, i)
        gray_frame = frame
        gray_old_frame = old_frame
        frame_kp, frame_desc = gen_sift_features(gray_old_frame)
        
        for j in frame_kp:
            info = [j.pt[0],j.pt[1]]
            frame_2d_points.append(info)
            
        frame_2d_points = np.array(frame_2d_points)
        frame_2d_points = np.float32(frame_2d_points[:, np.newaxis, :])
            
        optical_points, st, err = cv.calcOpticalFlowPyrLK(gray_old_frame, gray_frame, frame_2d_points, None, **lk_params)
        frame_kp, frame_desc, frame_2d_points = test_motion(frame_kp, frame_desc, optical_points, frame_2d_points)
        
        all_kp = all_kp+frame_kp
        all_dsc = all_dsc+frame_desc
        all_2d_points = all_2d_points+frame_2d_points

        dsc_len = len(all_dsc)
        
        while count_dsc != dsc_len:
            optical_dsc = gen_optical_dsc(all_2d_points[count_dsc][0], all_2d_points[count_dsc][1], gray_old_frame, gray_frame)
            optical_dsc = np.array(optical_dsc)
            all_dsc[count_dsc] = np.concatenate((all_dsc[count_dsc],optical_dsc))
            count_dsc+=1
        print(i)
    return all_kp, all_dsc

def gen_frame_dsc(video_name, kmeans):
    number_of_frames = count_frames(video_name)
    all_kp = []
    all_dsc = []
    all_dsc_aux = []
    all_2d_points = []
    count_dsc = 0
    feature_vector = []
    all_feature_vector = []
    
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
            
        optical_points, st, err = cv.calcOpticalFlowPyrLK(gray_old_frame, gray_frame, frame_2d_points, None, **lk_params)
        frame_kp, frame_desc, frame_2d_points = test_motion(frame_kp, frame_desc, optical_points, frame_2d_points)
        
        all_kp = all_kp+frame_kp
        all_dsc = all_dsc+frame_desc
        all_2d_points = all_2d_points+frame_2d_points

        dsc_len = len(all_dsc)
        
        while count_dsc != dsc_len:
            optical_dsc = gen_optical_dsc(all_2d_points[count_dsc][0], all_2d_points[count_dsc][1], gray_old_frame, gray_frame)
            optical_dsc = np.array(optical_dsc)
            all_dsc[count_dsc] = np.concatenate((all_dsc[count_dsc],optical_dsc))
            all_dsc_aux.append(all_dsc[count_dsc])
            count_dsc+=1
            
        print("frame:", i)
        if all_dsc_aux !=  []:
            feature_vector = vb.gen_feature_vector(all_dsc_aux, kmeans)
            all_feature_vector.append(feature_vector)
        all_dsc_aux = []
        
    return all_feature_vector

def gen_data_set():
    listing = os.listdir(r'C:\Users\ADM\Desktop\MoSIFT\dict')
    count_frame = 0
    
    for video in listing:
        video = r"C:/Users/ADM/Desktop/MoSIFT/dict/"+video
        all_kp, all_dsc = gen_mosift_features(video)
        dfdata_dict = pd.DataFrame(all_dsc)
        dfdata_dict.to_csv('data_dict.csv',mode='a',index=False)



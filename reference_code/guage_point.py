# %%
from tensorflow.keras.models import *
import tensorflow as tf
import time
import os
import scipy.io as sio
import scipy
import numpy as np
import math
import cv2
import matplotlib.cm as cm
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import imutils
from more_itertools import sort_together
import warnings

start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def Find_Contours(path_img,path_img_fib,mask_name,fib_name,resize,thr_set,area_low,gaussian_blur,kernel_size,mask_x_l,mask_x_r,mask_y_u,mask_y_d,fib_x_l,fib_x_r,fib_y_u,fib_y_d,mask_pad_x_l,mask_pad_x_r,mask_pad_y_u,mask_pad_d,fib_pad_x_l,fib_pad_x_r,fib_pad_y_u,fib_pad_y_d,mode,pad_mode):
    
    #img = cv2.imread(mask_name + '.png')
    filter_edge = 20
    
    if mode == "Mask":
        os.chdir(path_img)
        img = cv2.imread(mask_name)# + '.png')
        print("\nInput img shape:",img.shape)
                
        input_img = img.copy()
        input_img2 = img.copy()
        input_img3 = img.copy()
        input_img4 = img.copy()
        img_gray = img.copy()
        img_gray = img_gray[mask_y_u:img_gray.shape[0] - mask_y_d , mask_x_l:img_gray.shape[1] - mask_x_r]
        if pad_mode == "on":
            img_gray = np.pad(img_gray,((mask_pad_y_d,mask_pad_y_u),(mask_pad_x_r,mask_pad_x_l)),'constant')
        print("Image gray shape:",img_gray.shape)
        img_gray = cv2.cvtColor(img_gray,cv2.COLOR_BGR2GRAY)
        thr_auto, img_binary_auto = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_final = cv2.drawContours(input_img, contours, -1, (0,0,255), 5)
        img_final_fib = np.zeros(img_final.shape, dtype = np.uint8)
        contours_fib = 0
        
        return img, input_img, input_img2, input_img3, input_img4, img_gray, img_final, img_final_fib, contours, contours_fib
        
    if mode == "FIB":
        os.chdir(path_img_fib)
        img = cv2.imread(fib_name)# + '.')
        print("\nInput img shape:",img.shape) 
        
        input_img = img.copy()
        input_img2 = img.copy()
        input_img3 = img.copy()
        img_gray = img.copy()
        img_gray = cv2.cvtColor(img_gray,cv2.COLOR_BGR2GRAY)
        
        if gaussian_blur == "on":
            img_gray = cv2.GaussianBlur(img_gray, kernel_size, 0)
        
        img_gray = img_gray[fib_y_u:img_gray.shape[0] - fib_y_d , fib_x_l:img_gray.shape[1] - fib_x_r]
        if pad_mode == "on":
            img_gray = np.pad(img_gray,((fib_pad_y_d,fib_pad_y_u),(fib_pad_x_r,fib_pad_x_l)),'constant')
        #img_gray = img_gray[0:img_gray.shape[0]-220,:img_gray.shape[1]-126]#img_gray.shape[1]-96]
        #img_gray = np.flipud(img_gray)
        
        img_gray_copy = img_gray.copy()
        print("Image gray shape:",img_gray.shape) 
        thr_auto, img_binary_auto = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr, binary = cv2.threshold(img_gray, thr_auto + thr_set, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
        img_binary_tmp = np.zeros(img_gray_copy.shape, dtype = np.uint8)
        List_contours = []
        List_cx = []
        List_cy = []
        for c in contours:             
            area = cv2.contourArea(c) 
            
            if area >= area_low and area <= 50000:   #3000
                m = cv2.moments(c)
                if m["m00"]!=0: 
                    #centroid of x
                    cx = int(m["m10"] / m["m00"])
                    #centroid of y
                    cy = int(m["m01"] / m["m00"])         
                else:
                    cx = -1
                    cy = -1 
                if cx >= filter_edge and cx <= np.size(img_gray_copy,1)-filter_edge and cy >= filter_edge and cy <= np.size(img_gray_copy,0)-filter_edge:
                    List_contours.append(c)
                    List_cx.append(cx)
                    List_cy.append(cy)
        
        sort_together([List_cy, List_cx])[1]
        sort_together([List_cy, List_contours])[1]
        List_cy.sort()
        List_cx = List_cx[1:-2]
        List_cy = List_cy[1:-2]
        List_contours = List_contours[1:-2]
        
        temp_cy = sort_together([List_cx, List_cy])[1]
        temp_contours = sort_together([List_cx, List_contours])[1]
        temp_cx = sorted(List_cx)
        temp_cx = temp_cx[:-2]
        temp_cy = temp_cy[:-2]
        temp_contours = temp_contours[:-2]
        
        img_final_fib = cv2.drawContours(img_binary_tmp, temp_contours, -1, (255,255,255), -1)
        
        img_final_cnt = img_final_fib.copy()
        print("\nimage fib binary shape:",img_final_cnt.shape)
        img_final_cnt1 = cv2.cvtColor(img_final_cnt,cv2.COLOR_GRAY2RGB)
        contours_fib = temp_contours
        print("img final cnt1 shape:",img_final_cnt1.shape)
        img_final = cv2.drawContours(img_final_cnt1, temp_contours, -1, (0,0,255), 10)
        img = img_final_cnt1
        input_img = img_final_cnt1
        input_img2 = cv2.cvtColor(img_final_cnt,cv2.COLOR_GRAY2RGB)
        input_img3 = cv2.cvtColor(img_final_cnt,cv2.COLOR_GRAY2RGB)
    
        return img, input_img, input_img2, input_img3, img_gray, img_final, img_final_fib, contours, contours_fib    
    
    if mode == "Mask+FIB":
        os.chdir(path_img)
        img = cv2.imread(mask_name)# + '.png') #[y:x:RGB]
        
        img = img[mask_y_u:img.shape[0] - mask_y_d , mask_x_l:img.shape[1] - mask_x_r]
        if pad_mode == "on":
            img = np.pad(img,((mask_pad_y_d,mask_pad_y_u),(mask_pad_x_r,mask_pad_x_l),(0,0)),'constant')
        
        #print("\nInput mask img shape:",img.shape)
        input_img = img.copy()

        img_gray = img.copy()
        #print("Image gray shape:",img_gray.shape)
        img_gray = cv2.cvtColor(img_gray,cv2.COLOR_BGR2GRAY)

        thr_auto, img_binary_auto = cv2.threshold(img_gray , 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr, binary = cv2.threshold(img_gray, thr_auto + thr_set, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        __, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_final = cv2.drawContours(input_img, contours, -1, (0,0,255), 10)
        img_final_fib = np.zeros(img_final.shape, dtype = np.uint8)
        contours_fib = 0 
         
        os.chdir(path_img_fib)
        img_1 = cv2.imread(fib_name)# + '.png')
         

        img_gray_1 = img_1.copy()
        img_gray_1 = cv2.cvtColor(img_gray_1,cv2.COLOR_BGR2GRAY)
        if pad_mode == "on":
            img_gray_1 = np.pad(img_gray_1,((fib_pad_y_d,fib_pad_y_u),(fib_pad_x_r,fib_pad_x_l)),'constant')
        if gaussian_blur == "on":
            img_gray_1 = cv2.GaussianBlur(img_gray_1, kernel_size, 0)
        
        img_gray_1 = img_gray_1[fib_y_u:img_gray_1.shape[0] - fib_y_d , fib_x_l:img_gray_1.shape[1] - fib_x_r]
        img_gray_1_copy = img_gray_1.copy()
        #print("Image gray shape:",img_gray_1.shape) 
        thr_auto1, img_binary_auto1 = cv2.threshold(img_gray_1 , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr1, binary1 = cv2.threshold(img_gray_1, thr_auto1 + thr_set, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        __, contours1, hierarchy = cv2.findContours(binary1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
        img_binary_tmp = np.zeros(img_gray_1_copy.shape, dtype = np.uint8)
        List_contours = []
        List_cx = []
        List_cy = []
        for c in contours1:             
            area = cv2.contourArea(c) 
            
            if area >= area_low and area <= 50000:   
                m = cv2.moments(c)
                if m["m00"]!=0: 
                    #centroid of x
                    cx = int(m["m10"] / m["m00"])
                    #centroid of y
                    cy = int(m["m01"] / m["m00"])         
                else:
                    cx = -1
                    cy = -1 
                if cx >= filter_edge and cx <= np.size(img_gray_1_copy,1)-filter_edge and cy >= filter_edge and cy <= np.size(img_gray_1_copy,0)-filter_edge:
                    List_contours.append(c)
                    List_cx.append(cx)
                    List_cy.append(cy)
                
        temp_cy = sort_together([List_cx, List_cy])[1] #根據cx大小排列
        temp_contours = sort_together([List_cx, List_contours])[1]
        temp_cx = sorted(List_cx)
        temp_cx = temp_cx[:-2]
        temp_cy = temp_cy[:-2]
        temp_contours = temp_contours[:-2]
        tmp_c = []

        img_final_fib = cv2.drawContours(img_binary_tmp, List_contours, -1, (255,255,255), -1)
        input_img_fib = img_final_fib.copy()

        contours_fib = List_contours#temp_contours

        return img, input_img, input_img_fib, img_gray, img_final, img_final_fib, contours, contours_fib

    
def Find_Vertex(contours,img,input_img2,input_img5,mode,vertex_num):
    vertex_x = []
    vertex_y = []
    img_final2 = img.copy()
    list_contours = []
    pts_array = []
    pts_array_x = []
    pts_array_y = []
    vertex_array = []
    vertex_array_idx = []
    vertex_pattern = []    
    for c in contours:
        list_contours.append(c)        
    
    #print(pts_array)
    for pattern in range(0,len(list_contours)):
        #print("\nCurrent pattern = %d"%pattern)
        pts_array.append([]*len(list_contours))
        pts_array_x.append([]*len(list_contours))
        pts_array_y.append([]*len(list_contours))
        vertex_array.append([]*len(list_contours))
        vertex_array_idx.append([]*len(list_contours))
        vertex_pattern.append([]*len(list_contours))
        for pts in range(0,len(list_contours[pattern])):
            pts_array[pattern].append(list_contours[pattern][pts][0])
            pts_array_x[pattern].append(list_contours[pattern][pts][0][0])
            pts_array_y[pattern].append(list_contours[pattern][pts][0][1])
        
        first_max_x_idx = np.where(pts_array_x[pattern] == np.max(pts_array_x[pattern]))[0][0]
        first_max_y_idx = np.where(pts_array_y[pattern] == np.max(pts_array_y[pattern]))[0][0]
        first_min_x_idx = np.where(pts_array_x[pattern] == np.min(pts_array_x[pattern]))[0][0]
        first_min_y_idx = np.where(pts_array_y[pattern] == np.min(pts_array_y[pattern]))[0][0]
        last_max_x_idx = np.where(pts_array_x[pattern] == np.max(pts_array_x[pattern]))[0][-1]
        last_max_y_idx = np.where(pts_array_y[pattern] == np.max(pts_array_y[pattern]))[0][-1]
        last_min_x_idx = np.where(pts_array_x[pattern] == np.min(pts_array_x[pattern]))[0][-1]
        last_min_y_idx = np.where(pts_array_y[pattern] == np.min(pts_array_y[pattern]))[0][-1]
        
        '''
        if vertex3[0] == vertex1[0] and vertex3[1] == vertex1[1]:
            vertex3 = tuple([pts_array_x[pattern][last_max_x_idx] ,pts_array_y[pattern][last_max_y_idx]])
        '''
        if vertex_num == 8:
            vertex1 = tuple([pts_array_x[pattern][first_min_y_idx] ,pts_array_y[pattern][last_min_y_idx]])
            vertex2 = tuple([pts_array_x[pattern][last_min_y_idx] ,pts_array_y[pattern][last_min_y_idx]])
            vertex3 = tuple([pts_array_x[pattern][first_max_y_idx] ,pts_array_y[pattern][first_max_y_idx]])
            vertex4 = tuple([pts_array_x[pattern][last_max_y_idx] ,pts_array_y[pattern][last_max_y_idx]])
            
            vertex_pattern[pattern].append(vertex1)
            vertex_pattern[pattern].append(vertex2)
            vertex_pattern[pattern].append(vertex3)
            vertex_pattern[pattern].append(vertex4) 

        else:   
            vertex1 = tuple([pts_array_x[pattern][first_min_y_idx] ,pts_array_y[pattern][first_min_y_idx]])
            #vertex1 = tuple([pts_array_x[pattern][first_min_y_idx] ,pts_array_y[pattern][first_min_y_idx]])
            vertex2 = tuple([pts_array_x[pattern][last_min_x_idx] ,pts_array_y[pattern][last_min_x_idx]])
            #vertex2 = tuple([pts_array_x[pattern][last_min_x_idx] ,pts_array_y[pattern][last_min_x_idx]])
            vertex3 = tuple([pts_array_x[pattern][last_max_y_idx] ,pts_array_y[pattern][last_max_y_idx]])
            #vertex3 = tuple([pts_array_x[pattern][last_max_y_idx] ,pts_array_y[pattern][first_max_y_idx]])
            
            #vertex3 = tuple([pts_array_x[idx][last_max_x_idx] ,pts_array_y[idx][last_max_y_idx]])
            vertex4 = tuple([pts_array_x[pattern][last_max_x_idx] ,pts_array_y[pattern][last_max_x_idx]])
            
            vertex_pattern[pattern].append(vertex1)
            vertex_pattern[pattern].append(vertex2)
            vertex_pattern[pattern].append(vertex3)
            vertex_pattern[pattern].append(vertex4) 
        #print(type(vertex_pattern[0][0]))
        
        if mode == "Mask":
            img_final2 = cv2.circle(input_img2,vertex1,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img2,vertex2,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img2,vertex3,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img2,vertex4,10,(0,0,255),-1)
        if mode == "FIB":
            img_final2 = cv2.circle(input_img2,vertex1,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img2,vertex2,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img2,vertex3,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img2,vertex4,10,(0,0,255),-1)
        if mode == "Mask+FIB":
            img_final2 = cv2.circle(input_img5,vertex1,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img5,vertex2,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img5,vertex3,10,(0,0,255),-1)
            img_final2 = cv2.circle(input_img5,vertex4,10,(0,0,255),-1)
        
    return pts_array, pts_array_x, pts_array_y, vertex_pattern, img_final2 #, vertex_array
        
def Find_Center_of_Mass(input_img, contours):
    center_of_mass_x = []
    center_of_mass_y = []
    center_of_mass = []
    for pattern in range(0, len(contours)):
        center_of_mass_x.append([])
        center_of_mass_y.append([])
        center_of_mass.append([])

        tmp_moments = cv2.moments(contours[pattern])
        tmp_cx = int(tmp_moments["m10"] / tmp_moments["m00"])
        tmp_cy = int(tmp_moments["m01"] / tmp_moments["m00"])
        Cm = (tmp_cx, tmp_cy)

        center_of_mass_x[pattern].append(tmp_cx)
        center_of_mass_y[pattern].append(tmp_cy)
        center_of_mass[pattern].append(Cm)

        img_final3 = cv2.circle(input_img, Cm, 10, (255, 0, 0), -1)

    return img_final3, center_of_mass

def Cm_Position(Cm_Mask,Cm_FIB,image):
    image_copy = image.copy() 
    image_copy = cv2.cvtColor(image_copy,cv2.COLOR_GRAY2RGB)
    
    for pattern in range(0,len(Cm_Mask)):
        img_final4 = cv2.circle(image_copy,Cm_Mask[pattern][0],10,(0,0,255),-1)
    for pattern1 in range(0,len(Cm_FIB)):
        img_final4 = cv2.circle(image_copy,Cm_FIB[pattern1][0],10,(255,0,0),-1)        
 
    return img_final4
    
def Axis_Extraction(pixel_size,vertex_pattern,input_img5,vertex_num):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    var_x1 = []
    var_x2 = []
    var_x3 = []
    var_y1 = []
    var_y2 = []
    var_y3 = []
    vector1 = []
    vector2 = []
    vector3 = []
    len_vector1_pixel = []
    len_vector2_pixel = []
    len_vector3_pixel = []
    long_axis_vector = []
    short_axis_vector = []
    short_axis_vector1 = []
    len_vector1 = np.zeros((len(vertex_pattern),1))
    len_vector2 = np.zeros((len(vertex_pattern),1))
    len_vector3 = np.zeros((len(vertex_pattern),1))
    long_axis = np.zeros((len(vertex_pattern),1))
    short_axis = np.zeros((len(vertex_pattern),1))
    short_axis1 = np.zeros((len(vertex_pattern),1))
    #print(vertex_pattern[0][1])
    for pattern in range(0,len(vertex_pattern)):
        x1.append([]*len(vertex_pattern))
        x2.append([]*len(vertex_pattern))
        x3.append([]*len(vertex_pattern))
        x4.append([]*len(vertex_pattern))
        y1.append([]*len(vertex_pattern))
        y2.append([]*len(vertex_pattern))
        y3.append([]*len(vertex_pattern))
        y4.append([]*len(vertex_pattern))
        var_x1.append([]*len(vertex_pattern))
        var_x2.append([]*len(vertex_pattern))
        var_x3.append([]*len(vertex_pattern))
        var_y1.append([]*len(vertex_pattern))
        var_y2.append([]*len(vertex_pattern))
        var_y3.append([]*len(vertex_pattern))
        vector1.append([]*len(vertex_pattern))
        vector2.append([]*len(vertex_pattern))
        vector3.append([]*len(vertex_pattern))
        len_vector1_pixel.append([]*len(vertex_pattern))
        len_vector2_pixel.append([]*len(vertex_pattern))
        len_vector3_pixel.append([]*len(vertex_pattern))
        long_axis_vector.append([]*len(vertex_pattern))
        short_axis_vector.append([]*len(vertex_pattern))
        short_axis_vector1.append([]*len(vertex_pattern))
        
        x1[pattern].append(vertex_pattern[pattern][0][0])
        x2[pattern].append(vertex_pattern[pattern][1][0])
        x3[pattern].append(vertex_pattern[pattern][3][0])
        x4[pattern].append(vertex_pattern[pattern][2][0])
        y1[pattern].append(vertex_pattern[pattern][0][1])
        y2[pattern].append(vertex_pattern[pattern][1][1])
        y3[pattern].append(vertex_pattern[pattern][3][1])
        y4[pattern].append(vertex_pattern[pattern][2][1])
        #備註:要加個判斷式讓水平或垂直的fin可以算正確長短軸尺寸
        if vertex_num == 8:
            x_tmp1 = np.array(x4[pattern]) - np.array(x1[pattern])
            x_tmp2 = np.array(x2[pattern]) - np.array(x1[pattern])
            x_tmp3 = np.array(x4[pattern]) - np.array(x1[pattern])
            y_tmp1 = np.array(y4[pattern]) - np.array(y1[pattern])
            y_tmp2 = np.array(y2[pattern]) - np.array(y1[pattern])
            y_tmp3 = np.array(y4[pattern]) - np.array(y1[pattern])
        else:    
            x_tmp1 = np.array(x4[pattern]) - np.array(x1[pattern])
            x_tmp2 = np.array(x2[pattern]) - np.array(x1[pattern])
            x_tmp3 = np.array(x3[pattern]) - np.array(x1[pattern])
            y_tmp1 = np.array(y3[pattern]) - np.array(y1[pattern])
            y_tmp2 = np.array(y2[pattern]) - np.array(y1[pattern])
            y_tmp3 = np.array(y3[pattern]) - np.array(y1[pattern])
            
        var_x1[pattern].append(x_tmp1)
        var_x2[pattern].append(x_tmp2)
        var_x3[pattern].append(x_tmp3)
        var_y1[pattern].append(y_tmp1)
        var_y2[pattern].append(y_tmp2)
        var_y3[pattern].append(y_tmp3)
        
        vector_tmp1 = var_x1[pattern][0][0] + var_y1[pattern][0][0] * 1j
        vector_tmp2 = var_x2[pattern][0][0] + var_y2[pattern][0][0] * 1j
        vector_tmp3 = var_x3[pattern][0][0] + var_y3[pattern][0][0] * 1j
        vector1[pattern].append(vector_tmp1)
        vector2[pattern].append(vector_tmp2)
        vector3[pattern].append(vector_tmp3)
        
        len_vector_tmp1 = np.real(np.sqrt(vector1[pattern][0] * np.conj(vector1[pattern][0]))) #+ 1
        len_vector_tmp2 = np.real(np.sqrt(vector2[pattern][0] * np.conj(vector2[pattern][0]))) #+ 1
        len_vector_tmp3 = np.real(np.sqrt(vector3[pattern][0] * np.conj(vector3[pattern][0]))) #+ 1
        len_vector1_pixel[pattern].append(len_vector_tmp1)
        len_vector2_pixel[pattern].append(len_vector_tmp2)
        len_vector3_pixel[pattern].append(len_vector_tmp3)
        len_vector1[pattern] = np.array(len_vector1_pixel[pattern][0]) * pixel_size
        len_vector2[pattern] = np.array(len_vector2_pixel[pattern][0]) * pixel_size
        len_vector3[pattern] = np.array(len_vector3_pixel[pattern][0]) * pixel_size
        
        
        if len_vector1[pattern] >= len_vector2[pattern]:
            long_axis[pattern] = len_vector1[pattern]
            long_axis_vector[pattern] = vector1[pattern]
            short_axis[pattern] = len_vector2[pattern]
            short_axis_vector[pattern] = vector2[pattern]
        else:
            long_axis[pattern] = len_vector2[pattern]
            long_axis_vector[pattern] = vector2[pattern]
            short_axis[pattern] = len_vector1[pattern]
            short_axis_vector[pattern] = vector1[pattern]
        if vertex_num == 8:
            img_final5 = cv2.arrowedLine(input_img5, (x1[pattern][0], y1[pattern][0]), (x2[pattern][0], y2[pattern][0]), (0, 200, 0), 5)
            img_final5 = cv2.arrowedLine(input_img5, (x1[pattern][0], y1[pattern][0]), (x4[pattern][0], y4[pattern][0]), (0, 200, 0), 5)
            img_final5 = cv2.circle(input_img5,vertex_pattern[pattern][0],5,(0,0,255),-1)
        else:
            img_final5 = cv2.arrowedLine(input_img5, (x1[pattern][0], y1[pattern][0]), (x2[pattern][0], y2[pattern][0]), (0, 200, 0), 5)
            img_final5 = cv2.arrowedLine(input_img5, (x1[pattern][0], y1[pattern][0]), (x3[pattern][0], y3[pattern][0]), (0, 200, 0), 5)
            img_final5 = cv2.circle(input_img5,vertex_pattern[pattern][0],5,(0,0,255),-1)
            
    print("\n",long_axis_vector[0])
    #print("\n",vector1[0])
    #print("\n",vector2[0])
    #print("\n",vector3[0])
    cv2.namedWindow('Vector',0)
    cv2.imshow('Vector',img_final5)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows()
    '''
    print("\n") 
    print("length vector1:",len_vector1_pixel)
    print("length vector2:",len_vector2_pixel)
    print("long axis is {:.2f} nm".format(long_axis))
    print("short_axis is {:.2f} nm".format(short_axis))
    '''
    return var_x1, var_x2, var_y1, var_y2, vector1, vector2, len_vector1_pixel, len_vector2_pixel, len_vector1, len_vector2, long_axis_vector, short_axis_vector, long_axis, short_axis, img_final5

def Get_Angle(pixel_size, vector1, vector2, short_axis_vector, len_vector1, len_vector2, len_vector3):
    array_vector1 = []
    array_vector2 = []
    array_vector3 = []
    len_product = []
    len_product_horizontal = []
    dot_product = []
    dot_product_horizontal = []
    angle_fin = np.zeros((len(vector1),1))
    angle_PB = np.zeros((len(vector1),1))
    angle_fin_correct = np.zeros((len(vector1),1))
    angle_PB_correct = np.zeros((len(vector1),1))
    array_vector_horizontal = [pixel_size,0]
    len_vector_horizontal = pixel_size
    #print("\nArray vector:",array_vector_horizontal[0])
    for pattern in range(0,len(vector1)):
        array_vector1.append([]*len(vector1))
        array_vector2.append([]*len(vector1))
        array_vector3.append([]*len(vector1))
        len_product.append([]*len(vector1))
        len_product_horizontal.append([]*len(vector1))
        dot_product.append([]*len(vector1))
        dot_product_horizontal.append([]*len(vector1))
        
        tmp1 = [np.round((np.real(vector1[pattern]) * pixel_size),3), (np.imag(vector1[pattern]) * pixel_size)]
        tmp2 = [np.round((np.real(vector2[pattern]) * pixel_size),3), (np.imag(vector2[pattern]) * pixel_size)]
        tmp3 = [np.round((np.real(short_axis_vector[pattern]) * pixel_size),3), np.round((np.imag(short_axis_vector[pattern]) * pixel_size),3)]
        len_product_tmp = (len_vector1[pattern][0] - pixel_size) * (len_vector2[pattern][0] - pixel_size) 
        len_product_horizontal_tmp = (len_vector3[pattern][0] - pixel_size) * len_vector_horizontal
        
        array_vector1[pattern].append(tmp1)
        array_vector2[pattern].append(tmp2)
        array_vector3[pattern].append(tmp3)
        len_product[pattern].append(len_product_tmp)
        len_product_horizontal[pattern].append(len_product_horizontal_tmp)
        
        dot_product_tmp = (array_vector1[pattern][0][0] * array_vector2[pattern][0][0] + array_vector1[pattern][0][1] * array_vector2[pattern][0][1])
        dot_product_horizontal_tmp = (array_vector3[pattern][0][0] * array_vector_horizontal[0] + array_vector3[pattern][0][1] * array_vector_horizontal[1])
        
        dot_product[pattern].append(np.round(dot_product_tmp,2))
        dot_product_horizontal[pattern].append(np.round(dot_product_horizontal_tmp,2))
                
        angle_tmp = np.round(np.degrees(np.arccos(np.round(dot_product[pattern][0],4) / len_product[pattern][0])),0)
        inverse_inner_product_tmp = np.round(dot_product_horizontal[pattern][0] / len_product_horizontal[pattern][0],2)
        if np.abs(inverse_inner_product_tmp) > 1:
            inverse_inner_product_tmp = 1
        angle_PB_tmp = np.round(np.degrees(np.arccos(inverse_inner_product_tmp)),2)
        #print("%d idx angle is %.2f"%(pattern,angle_PB_tmp))
        angle_fin[pattern] = angle_tmp
        angle_PB[pattern] = angle_PB_tmp
        
        
        
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    warnings.simplefilter('ignore', SyntaxWarning)
    
    '''
    print("\nlen horizontal vector:{} nm".format(len_vector_horizontal))
    #print("Multiply len vector1 and horizontal vector:",len_product_with_horizontal)
    print("Long-short axis inner product value is: {:.3f}".format(dot_product))
    print("PB inner product value is: {:.3f}".format(dot_product_with_horizontal))
    #print("Multiply len vector1 and horizontal vector:",np.round(len_product_with_horizontal,3))
    print("\nLong-short axis angle is {} degrees".format(angle))
    print("Nano Fin rotation angle is {} degrees".format(angle_PB))
    '''    
    return array_vector1, array_vector2, array_vector3, len_product, len_product_horizontal, dot_product, dot_product_horizontal, angle_fin, angle_PB


def Alignment_Mask_FIB(img_mask,img_fib,contours_mask,contours_FIB,center_of_mass_mask,center_of_mass_mass_FIB):
    #print("\nLen contour fib:",len(contours_FIB))
    #print("Len center of mask fib:",len(center_of_mass_FIB))
    #print("\nBinary Mask shape:",img_mask.shape)
    #print("Binary FIB shape:",img_fib.shape)
    
    #img_mask_alignment = img_mask
    img_mask = img_mask[:img_mask.shape[0]-0,:img_mask.shape[1]-0]
    img_mask = np.pad(img_mask,((0,0),(0,0)))
    img_mask_copy = np.zeros(img_mask.shape, dtype = np.uint8)
    #img_mask_alignment = cv2.cvtColor(img_mask_alignment,cv2.COLOR_GRAY2RGB)
    img_fib_copy = np.zeros(img_fib.shape, dtype = np.uint8)
    img_fib_alignment = img_fib_copy

    List_cx_in = []            
    List_cy_in = []
    List_area_in = []
    List_contours_in = []
    
    List_cx_out = []            
    List_cy_out = []
    List_area_out = []
    List_contours_out = []
    
    thr_auto_in, img_in_binary = cv2.threshold(img_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    __, contours_in , hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thr_auto_out, img_out_binary = cv2.threshold(img_fib,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    __, contours_out , hierarchy = cv2.findContours(img_fib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    cv2.namedWindow('Before FIB Alignment',0)
    cv2.imshow('Before FIB Alignment',img_fib)
    cv2.waitKey(0)
    '''
    for c1 in contours_in:             
        area_in = cv2.contourArea(c1)            
        m1 = cv2.moments(c1)  
        if m1["m00"]!=0: 
            #centroid of x
            cx_in = int(m1["m10"] / m1["m00"])
            #centroid of y
            cy_in = int(m1["m01"] / m1["m00"])         
        else:
            cx_in = -1
            cy_in = -1 
            
        # cv2.circle(img_mask_copy, (cx_in, cy_in), 10, (255,255,0), -1)    
        List_cx_in.append(cx_in)               
        List_cy_in.append(cy_in)    
        List_area_in.append(area_in)
        List_contours_in.append(c1)
    
    for c2 in contours_out:            
        area_out = cv2.contourArea(c2)            
        m2 = cv2.moments(c2)
        if m2["m00"]!=0: 
            #centroid of x
            cx_out = int(m2["m10"] / m2["m00"])
            #centroid of y
            cy_out = int(m2["m01"] / m2["m00"])         
        else:
            cx_out = -1
            cy_out = -1 
            
        #cv2.circle(img_out_copy, (cx_out, cy_out), 10, (0,255,0), -1)    
        List_cx_out.append(cx_out)               
        List_cy_out.append(cy_out)    
        List_area_out.append(area_out)
        List_contours_out.append(c2)
        
    for i in range(len(List_contours_out)):
        temp_diff = []
        temp_diff_x = []
        temp_diff_y = []
        for j in range(len(List_contours_in)):
            diff_x = List_cx_out[i]-List_cx_in[j]
            diff_y = List_cy_out[i]-List_cy_in[j]
            diff = (diff_x)**2 + (diff_y)**2
            temp_diff.append(diff)
            temp_diff_x.append(diff_x)
            temp_diff_y.append(diff_y) 
            
        index = np.where(temp_diff == np.amin(temp_diff)) 
        index = int(index[0][0])
        draw_diff_x = temp_diff_x[index]
        draw_diff_y = temp_diff_y[index]
        
        
        x, y, w, h = cv2.boundingRect(List_contours_out[i])
        x1, y1, w1, h1 = cv2.boundingRect(List_contours_in[index])
        draw_x = x - draw_diff_x
        draw_y = y - draw_diff_y
        
        img_temp_fib = img_fib[y:y+h,x:x+w]        
        img_fib_copy[draw_y:draw_y+h,draw_x:draw_x+w] = img_temp_fib
        
        img_temp_mask = img_mask[y1:y1+h1,x1:x1+w1]
        img_mask_copy[y1:y1+h1,x1:x1+w1] = img_temp_mask
        
    
    img_fib_alignment = img_fib_copy  
    img_mask_alignment = img_mask_copy
    '''
    cv2.namedWindow('After FIB Alignment',0)
    cv2.imshow('After FIB Alignment',img_temp_fib)
    cv2.waitKey(0)
    '''
    thr_auto_in, img_in_binary = cv2.threshold(img_mask_alignment,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    __, new_mask_contours , hierarchy = cv2.findContours(img_mask_alignment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thr_auto_out, img_out_binary = cv2.threshold(img_fib_alignment,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    __, new_fib_contours , hierarchy = cv2.findContours(img_fib_alignment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    img_mask_filt = np.zeros(img_mask.shape, dtype = np.uint8)
    img_fib_filt = np.zeros(img_fib.shape, dtype = np.uint8)
    
    list_contours_new_mask = []
    list_contours_new_fib = []
    
    for c3 in new_mask_contours:
        area = cv2.contourArea(c3)
        if area>=100:
            list_contours_new_mask.append(c3)
            
    for c4 in new_fib_contours:
        area = cv2.contourArea(c4)
        if area>=100:
            list_contours_new_fib.append(c4)
    
    img_mask_alignment_final = cv2.drawContours(img_mask_filt, list_contours_new_mask, -1, (255, 255, 255), -1)
    img_fib_alignment_final = cv2.drawContours(img_fib_filt, list_contours_new_fib, -1, (255, 255, 255), -1)
    
    return img_mask_alignment_final, img_fib_alignment_final, list_contours_new_mask, list_contours_new_fib

def draw_ellipse(input_img, short_axis, long_axis, center_of_mass, angle_PB, pixel_size):
    for pattern in range(0, len(short_axis)):
        short_axis_tmp = (short_axis[pattern][0])/pixel_size/2
        long_axis_tmp = (long_axis[pattern][0])/pixel_size/2
        # print(center_of_mass[pattern][0], (int(np.round(short_axis_tmp)), int(np.round(long_axis_tmp))), angle_PB[pattern][0])
        img_final4 = cv2.ellipse(input_img, center_of_mass[pattern][0], (int(np.round(short_axis_tmp)), int(np.round(long_axis_tmp))), angle_PB[pattern][0], 0, 360, (255, 0, 255), 5)
        
        
    return img_final4

def Guage_List_Generation_mask(img_test,img_test2,info_update,pixel_size):
    print("\n----------------------Skip----------------------")
    long_x_vector_sort = []
    long_y_vector_sort = []
    long_len_sort = []
    long_len_sort_cal = []
    short_x_vector_sort = []
    short_y_vector_sort = []
    short_len_sort = []
    short_len_sort_cal = []
    angle_sort = []
    Cm_sort = []
    cnt_sort = []
    guage_list_sort = []
    guage_list_sort1 = []
    guage_list_sort2 = []
    guage_r_l = []
    guage_u_b = []
    guage_long = np.zeros((len(info_update),1))
    guage_short = np.zeros((len(info_update),1))
    Cm_sort_x = np.zeros((len(info_update),1))
    Cm_sort_y = np.zeros((len(info_update),1))
    
    for i in range(0,len(info_update)):
        long_x_vector_sort.append(i*[])
        long_y_vector_sort.append(i*[])
        long_len_sort.append(i*[])
        long_len_sort_cal.append(i*[])
        short_x_vector_sort.append(i*[])
        short_y_vector_sort.append(i*[])
        short_len_sort.append(i*[])
        short_len_sort_cal.append(i*[])
        angle_sort.append(i*[])
        Cm_sort.append(i*[])
        cnt_sort.append(i*[])
        guage_list_sort.append(i*[])
        guage_list_sort1.append(i*[])
        guage_list_sort2.append(i*[])
        guage_r_l.append(i*[])
        guage_u_b.append(i*[])
                
        Cm_sort[i].append(info_update[i][0][0])
        long_x_vector_sort[i].append(np.real(info_update[i][3][0]))
        long_y_vector_sort[i].append(np.imag(info_update[i][3][0]))
        long_len_sort[i].append(np.round(info_update[i][1][0],4))
        long_len_sort_cal[i].append(np.round(info_update[i][1][0]-pixel_size,4))
        short_x_vector_sort[i].append(np.real(info_update[i][4][0]))
        short_y_vector_sort[i].append(np.imag(info_update[i][4][0]))
        short_len_sort[i].append(np.round(info_update[i][2][0],4))
        short_len_sort_cal[i].append(np.round(info_update[i][2][0]-pixel_size,4))
        angle_sort[i].append(info_update[i][5][0])
        cnt_sort[i].append(info_update[i][6][0])
        
        Cm_sort_x[i][0] = Cm_sort[i][0][0]
        Cm_sort_y[i][0] = Cm_sort[i][0][1]
    #angle_sort[295] = 180.0
    #angle_sort[493] = 0.0
    #print("\nAngle sort:",angle_sort)
    for j in range(0,len(info_update)):    
        i_right = 0 #distance to the right boundary
        i_up = 0 #distance to the up boundary
        i_left = 0 #distance to the left boundary
        i_bottom = 0 #distance to the bottom boundary
        
        i_right1 = 0 #distance to the right boundary
        i_up1 = 0 #distance to the up boundary
        i_left1 = 0 #distance to the left boundary
        i_bottom1 = 0 #distance to the bottom boundary
        
        i_right2 = 0 #distance to the right boundary
        i_up2 = 0 #distance to the up boundary
        i_left2 = 0 #distance to the left boundary
        i_bottom2 = 0 #distance to the bottom boundary
        
        i_right3 = 0 #distance to the right boundary
        i_up3 = 0 #distance to the up boundary
        i_left3 = 0 #distance to the left boundary
        i_bottom3 = 0 #distance to the bottom boundary
        if int(angle_sort[j][0]) == 0 and int(angle_sort[j][0]) != 90:
            #Right edge dection
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right+1] == 0 :
                    break
                else:
                    i_right+=1
            guage_list_sort[j].append(i_right)
            
            #Left edge dection    
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left-1] == 0 :
                    break
                else:
                    i_left+=1
            guage_list_sort[j].append(i_left)
            
            #Up edge dection
            while img_test[int(Cm_sort_y[j])-i_up,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])-i_up-1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_up+=1
            guage_list_sort[j].append(i_up)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+i_bottom,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])+i_bottom+1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_bottom+=1
            guage_list_sort[j].append(i_bottom)
               
        if int(angle_sort[j][0]) == 90 and int(angle_sort[j][0]) != 0:
            #Right edge dection
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right1] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right1+1] == 0 :
                    break
                else:
                    i_right1+=1
            guage_list_sort[j].append(i_right1)
            
            #Left edge dection    
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left1] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left1-1] == 0 :
                    break
                else:
                    i_left1+=1
            guage_list_sort[j].append(i_left1)
            
            #Up edge dection
            while img_test[int(Cm_sort_y[j])-i_up1,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])-i_up1-1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_up1+=1
            guage_list_sort[j].append(i_up1)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+i_bottom1,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])+i_bottom1+1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_bottom1+=1
            guage_list_sort[j].append(i_bottom1)
        
        if int(angle_sort[j][0]) > 90:
            #Right edge dection
            #while img_test[int(Cm_sort_y[j])-int(i_right2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j])),int(Cm_sort_x[j])+int(i_right2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])] == 1:
            #    if img_test[int(Cm_sort_y[j])-int(i_right2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j]))-1,int(Cm_sort_x[j])+int(i_right2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])+1] == 0 :
            while img_test[int(Cm_sort_y[j])-int(i_right2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_right2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_right2+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_right2+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :        
                    break
                else:
                    i_right2+=1
            guage_list_sort[j].append(i_right2)
            
            #Left edge dection    
            #while img_test[int(Cm_sort_y[j])+int(i_left2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j])),int(Cm_sort_x[j])-int(i_left2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])] == 1:
            #    if img_test[int(Cm_sort_y[j])+int(i_left2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j]))+1,int(Cm_sort_x[j])-int(i_left2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])-1] == 0 :
            while img_test[int(Cm_sort_y[j])+int(i_left2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_left2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_left2+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_left2+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_left2+=1
            guage_list_sort[j].append(i_left2)
            
            #Up edge dection
            #while img_test[int(Cm_sort_y[j])-int(i_up2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort[j])),int(Cm_sort_x[j])-int(i_up2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort[j])] == 1:
            #    if img_test[int(Cm_sort_y[j])-int(i_up2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort[j]))-1,int(Cm_sort_x[j])-int(i_up2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort[j])-1] == 0 :
            while img_test[int(Cm_sort_y[j])-int(i_up2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_up2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_up2+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_up2+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_up2+=1
            guage_list_sort[j].append(i_up2)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+int(i_bottom2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_bottom2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_bottom2+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_bottom2+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_bottom2+=1
            guage_list_sort[j].append(i_bottom2)
            
        if int(angle_sort[j][0]) < 90 and int(angle_sort[j][0]) != 0:
            #Right edge dection
            while img_test[int(Cm_sort_y[j])+int(i_right3*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_right3*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_right3+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_right3+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_right3+=1
            guage_list_sort[j].append(i_right3)
            
            #Left edge dection    
            while img_test[int(Cm_sort_y[j])-int(i_left3*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_left3*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_left3+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_left3+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_left3+=1
            guage_list_sort[j].append(i_left3)
            
            #Up edge dection
            while img_test[int(Cm_sort_y[j])-int(i_up3*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_up3*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_up3+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_up3+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_up3+=1
            guage_list_sort[j].append(i_up3)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+int(i_bottom3*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_bottom3*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_bottom3+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_bottom3+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_bottom3+=1
            guage_list_sort[j].append(i_bottom3)
    
    for pattern in range(0,len(info_update)):
        guage_r_l[pattern].append(int(np.array(guage_list_sort[pattern][0])+np.array(guage_list_sort[pattern][1])))
        guage_u_b[pattern].append(int(np.array(guage_list_sort[pattern][2])+np.array(guage_list_sort[pattern][3])))
        if guage_r_l[pattern] > guage_u_b[pattern]:
            guage_long[pattern] = guage_r_l[pattern]
            guage_short[pattern] = guage_u_b[pattern]
        else:
            guage_long[pattern] = guage_u_b[pattern]
            guage_short[pattern] = guage_r_l[pattern]
        
    iii = 0
    print("\nAngle sort:",angle_sort[iii][0])
    print("\nLong axis x sort:",long_x_vector_sort[iii][0])
    print("Long axis y sort:",long_y_vector_sort[iii][0])
    print("Long axis length:",long_len_sort[iii][0])
    print("\nShort axis x sort:",short_x_vector_sort[iii][0])
    print("Short axis y sort:",short_y_vector_sort[iii][0])
    print("Short axis length:",short_len_sort[iii][0])
    print("Cm sort:",Cm_sort[iii][0])
    
    return Cm_sort, long_x_vector_sort, long_y_vector_sort, long_len_sort, short_x_vector_sort, short_y_vector_sort, short_len_sort, angle_sort, guage_list_sort, guage_long, guage_short, guage_r_l, guage_u_b

def Guage_List_Generation_fib(img_test,img_test2,info_update,pixel_size,info_update_fib):        
    idx = 1
    print("\n----------------------Skip----------------------")
    long_x_vector_sort = []
    long_y_vector_sort = []
    long_len_sort = []
    long_len_sort_cal = []
    short_x_vector_sort = []
    short_y_vector_sort = []
    short_len_sort = []
    short_len_sort_cal = []
    angle_sort = []
    Cm_sort = []
    cnt_sort = []
    guage_list_sort = []
    guage_list_sort1 = []
    guage_list_sort2 = []
    guage_r_l = []
    guage_u_b = []
    guage_long = np.zeros((len(info_update),1))
    guage_short = np.zeros((len(info_update),1))
    Cm_sort_x = np.zeros((len(info_update),1))
    Cm_sort_y = np.zeros((len(info_update),1))
    
    for i in range(0,len(info_update)):#_fib
        long_x_vector_sort.append(i*[])
        long_y_vector_sort.append(i*[])
        long_len_sort.append(i*[])
        long_len_sort_cal.append(i*[])
        short_x_vector_sort.append(i*[])
        short_y_vector_sort.append(i*[])
        short_len_sort.append(i*[])
        short_len_sort_cal.append(i*[])
        angle_sort.append(i*[])
        Cm_sort.append(i*[])
        cnt_sort.append(i*[])
        guage_list_sort.append(i*[])
        guage_list_sort1.append(i*[])
        guage_list_sort2.append(i*[])
        guage_r_l.append(i*[])
        guage_u_b.append(i*[])
                
        Cm_sort[i].append(info_update_fib[i][0][0])
        long_x_vector_sort[i].append(np.real(info_update[i][3][0]))
        long_y_vector_sort[i].append(np.imag(info_update[i][3][0]))
        long_len_sort[i].append(np.round(info_update[i][1][0],4))
        long_len_sort_cal[i].append(np.round(info_update[i][1][0]-pixel_size,4))
        short_x_vector_sort[i].append(np.real(info_update[i][4][0]))
        short_y_vector_sort[i].append(np.imag(info_update[i][4][0]))
        short_len_sort[i].append(np.round(info_update[i][2][0],4))
        short_len_sort_cal[i].append(np.round(info_update[i][2][0]-pixel_size,4))
        angle_sort[i].append(info_update[i][5][0])
        cnt_sort[i].append(info_update_fib[i][1][0])
        
        Cm_sort_x[i][0] = Cm_sort[i][0][0]
        Cm_sort_y[i][0] = Cm_sort[i][0][1]
        
    for j in range(0,len(info_update)): #_fib   
        i_right = 0 #distance to the right boundary
        i_up = 0 #distance to the up boundary
        i_left = 0 #distance to the left boundary
        i_bottom = 0 #distance to the bottom boundary
        
        i_right1 = 0 #distance to the right boundary
        i_up1 = 0 #distance to the up boundary
        i_left1 = 0 #distance to the left boundary
        i_bottom1 = 0 #distance to the bottom boundary
        
        i_right2 = 0 #distance to the right boundary
        i_up2 = 0 #distance to the up boundary
        i_left2 = 0 #distance to the left boundary
        i_bottom2 = 0 #distance to the bottom boundary
        
        i_right3 = 0 #distance to the right boundary
        i_up3 = 0 #distance to the up boundary
        i_left3 = 0 #distance to the left boundary
        i_bottom3 = 0 #distance to the bottom boundary
        if int(angle_sort[j][0]) == 0 and int(angle_sort[j][0]) != 90:
            #Right edge dection
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right+1] == 0 :
                    break
                else:
                    i_right+=1
            guage_list_sort[j].append(i_right)
            
            #Left edge dection    
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left-1] == 0 :
                    break
                else:
                    i_left+=1
            guage_list_sort[j].append(i_left)
            
            #Up edge dection
            while img_test[int(Cm_sort_y[j])-i_up,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])-i_up-1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_up+=1
            guage_list_sort[j].append(i_up)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+i_bottom,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])+i_bottom+1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_bottom+=1
            guage_list_sort[j].append(i_bottom)
               
        if int(angle_sort[j][0]) == 90 and int(angle_sort[j][0]) != 0:
            #Right edge dection
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right1] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])+i_right1+1] == 0 :
                    break
                else:
                    i_right1+=1
            guage_list_sort[j].append(i_right1)
            
            #Left edge dection    
            while img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left1] == 1:
                if img_test[int(Cm_sort_y[j]),int(Cm_sort_x[j])-i_left1-1] == 0 :
                    break
                else:
                    i_left1+=1
            guage_list_sort[j].append(i_left1)
            
            #Up edge dection
            while img_test[int(Cm_sort_y[j])-i_up1,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])-i_up1-1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_up1+=1
            guage_list_sort[j].append(i_up1)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+i_bottom1,int(Cm_sort_x[j])] == 1:
                if img_test[int(Cm_sort_y[j])+i_bottom1+1,int(Cm_sort_x[j])] == 0 :
                    break
                else:
                    i_bottom1+=1
            guage_list_sort[j].append(i_bottom1)
        
        if int(angle_sort[j][0]) > 90:
            #Right edge dection
            #while img_test[int(Cm_sort_y[j])-int(i_right2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j])),int(Cm_sort_x[j])+int(i_right2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])] == 1:
            #    if img_test[int(Cm_sort_y[j])-int(i_right2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j]))-1,int(Cm_sort_x[j])+int(i_right2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])+1] == 0 :
            while img_test[int(Cm_sort_y[j])-int(i_right2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_right2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_right2+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_right2+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :        
                    break
                else:
                    i_right2+=1
            guage_list_sort[j].append(i_right2)
            
            #Left edge dection    
            #while img_test[int(Cm_sort_y[j])+int(i_left2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j])),int(Cm_sort_x[j])-int(i_left2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])] == 1:
            #    if img_test[int(Cm_sort_y[j])+int(i_left2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort[j]))+1,int(Cm_sort_x[j])-int(i_left2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort[j])-1] == 0 :
            while img_test[int(Cm_sort_y[j])+int(i_left2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_left2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_left2+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_left2+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_left2+=1
            guage_list_sort[j].append(i_left2)
            
            #Up edge dection
            #while img_test[int(Cm_sort_y[j])-int(i_up2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort[j])),int(Cm_sort_x[j])-int(i_up2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort[j])] == 1:
            #    if img_test[int(Cm_sort_y[j])-int(i_up2*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort[j]))-1,int(Cm_sort_x[j])-int(i_up2*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort[j])-1] == 0 :
            while img_test[int(Cm_sort_y[j])-int(i_up2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_up2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_up2+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_up2+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_up2+=1
            guage_list_sort[j].append(i_up2)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+int(i_bottom2*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_bottom2*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_bottom2+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_bottom2+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_bottom2+=1
            guage_list_sort[j].append(i_bottom2)
            
            
        
        if int(angle_sort[j][0]) < 90 and int(angle_sort[j][0]) != 0:
            #Right edge dection
            while img_test[int(Cm_sort_y[j])+int(i_right3*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_right3*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_right3+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_right3+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_right3+=1
            guage_list_sort[j].append(i_right3)
            
            #Left edge dection    
            while img_test[int(Cm_sort_y[j])-int(i_left3*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_left3*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_left3+1)*(np.abs(short_y_vector_sort[j])*pixel_size/short_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_left3+1)*np.abs(short_x_vector_sort[j])*pixel_size/short_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_left3+=1
            guage_list_sort[j].append(i_left3)
            
            #Up edge dection
            while img_test[int(Cm_sort_y[j])-int(i_up3*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int(i_up3*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])-int((i_up3+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])+int((i_up3+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_up3+=1
            guage_list_sort[j].append(i_up3)
            
            #Bottom edge dection
            while img_test[int(Cm_sort_y[j])+int(i_bottom3*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int(i_bottom3*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 1:
                if img_test[int(Cm_sort_y[j])+int((i_bottom3+1)*(np.abs(long_y_vector_sort[j])*pixel_size/long_len_sort_cal[j])),int(Cm_sort_x[j])-int((i_bottom3+1)*np.abs(long_x_vector_sort[j])*pixel_size/long_len_sort_cal[j])] == 0 :
                    break
                else:
                    i_bottom3+=1
            guage_list_sort[j].append(i_bottom3)
    
    for pattern in range(0,len(info_update)):#_fib
        guage_r_l[pattern].append(int(np.array(guage_list_sort[pattern][0])+np.array(guage_list_sort[pattern][1])))
        guage_u_b[pattern].append(int(np.array(guage_list_sort[pattern][2])+np.array(guage_list_sort[pattern][3])))
        if guage_r_l[pattern] > guage_u_b[pattern]:
            guage_long[pattern] = guage_r_l[pattern]
            guage_short[pattern] = guage_u_b[pattern]
        else:
            guage_long[pattern] = guage_u_b[pattern]
            guage_short[pattern] = guage_r_l[pattern]
        
    iii = 0
    print("\nAngle sort:",angle_sort[iii][0])
    print("\nLong axis x sort:",long_x_vector_sort[iii][0])
    print("Long axis y sort:",long_y_vector_sort[iii][0])
    print("Long axis length:",long_len_sort[iii][0])
    print("\nShort axis x sort:",short_x_vector_sort[iii][0])
    print("Short axis y sort:",short_y_vector_sort[iii][0])
    print("Short axis length:",short_len_sort[iii][0])
    print("Cm sort:",Cm_sort[iii][0])
    
    return Cm_sort, long_x_vector_sort, long_y_vector_sort, long_len_sort, short_x_vector_sort, short_y_vector_sort, short_len_sort, angle_sort, guage_list_sort, guage_long, guage_short, guage_r_l, guage_u_b

def Guage_Points_mask(img_int,Cm,angle,guage,long_x_vector,long_y_vector,short_x_vector,short_y_vector,long,short,pixel_size):
    guage_point_up = []
    guage_point_bottom = []
    guage_point_right = []
    guage_point_left = []
    img_copy = img_int.copy()
    print("\n")
    for i in range(0,len(Cm)):
        guage_point_up.append(i*[])
        guage_point_bottom.append(i*[])
        guage_point_right.append(i*[])
        guage_point_left.append(i*[])
    
        if int(angle[i][0]) == 0 and int(angle[i][0]) != 90:
            idx_r = guage[i][0]
            idx_l = guage[i][1]
            idx_u = guage[i][2]
            idx_b = guage[i][3]
            
            tmp = []
            tmp.append(Cm[i][0][0]+idx_r)
            tmp.append(Cm[i][0][1])
            #tmp.append(Cm[i][0][0]+idx_r)
            tmp = tuple(tmp)
            guage_point_right[i].append(tmp)
            
            tmp1 = []
            tmp1.append(Cm[i][0][0]-idx_l)
            tmp1.append(Cm[i][0][1])
            #tmp1.append(Cm[i][0][0]-idx_l)
            tmp1 = tuple(tmp1)
            guage_point_left[i].append(tmp1)
            
            tmp2 = []
            tmp2.append(Cm[i][0][0])
            tmp2.append(Cm[i][0][1]-idx_u)
            #tmp2.append(Cm[i][0][0])
            tmp2 = tuple(tmp2)
            guage_point_up[i].append(tmp2)
            
            tmp3 = []
            tmp3.append(Cm[i][0][0])
            tmp3.append(Cm[i][0][1]+idx_b)
            #tmp3.append(Cm[i][0][0])
            tmp3 = tuple(tmp3)
            guage_point_bottom[i].append(tmp3)

        
        if int(angle[i][0]) == 90 and int(angle[i][0]) != 0:
            idx_r1 = guage[i][2]
            idx_l1 = guage[i][3]
            idx_u1 = guage[i][1]
            idx_b1 = guage[i][0]
            
            tmp4 = []
            
            tmp4.append(Cm[i][0][0])
            tmp4.append(Cm[i][0][1]+idx_r1)
            #tmp.append(Cm[i][0][1]+idx_r)
            tmp4 = tuple(tmp4)
            guage_point_right[i].append(tmp4)
            
            tmp5 = []
            
            tmp5.append(Cm[i][0][0])
            tmp5.append(Cm[i][0][1]-idx_l1)
            #tmp1.append(Cm[i][0][1]-idx_l)
            tmp5 = tuple(tmp5)
            guage_point_left[i].append(tmp5)
            
            tmp6 = []
            
            tmp6.append(Cm[i][0][0]-idx_u1)
            tmp6.append(Cm[i][0][1])
            #tmp2.append(Cm[i][0][1])
            tmp6 = tuple(tmp6)
            guage_point_up[i].append(tmp6)
            
            tmp7 = []
            
            tmp7.append(Cm[i][0][0]+idx_b1)
            tmp7.append(Cm[i][0][1])
            #tmp3.append(Cm[i][0][1])
            tmp7 = tuple(tmp7)
            guage_point_bottom[i].append(tmp7)
        
        if int(angle[i][0]) > 90:
            idx_r2 = guage[i][0]
            idx_l2 = guage[i][1]
            idx_u2 = guage[i][2]
            idx_b2 = guage[i][3]
            
            tmp8 = []
            tmp8.append(int(Cm[i][0][0])+int(idx_r2*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp8.append(int(Cm[i][0][1])-int(idx_r2*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp.append(int(Cm[i][0][0])+int(idx_r*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp8 = tuple(tmp8)
            guage_point_right[i].append(tmp8)
            
            tmp9 = [] 
            tmp9.append(int(Cm[i][0][0])-int(idx_l2*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp9.append(int(Cm[i][0][1])+int(idx_l2*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp1.append(int(Cm[i][0][0])-int(idx_l*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp9 = tuple(tmp9)
            guage_point_left[i].append(tmp9)
            
            tmp10 = []
            tmp10.append(int(Cm[i][0][0])-int(idx_u2*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp10.append(int(Cm[i][0][1])-int(idx_u2*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp2.append(int(Cm[i][0][0])-int(idx_u*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp10 = tuple(tmp10)
            guage_point_up[i].append(tmp10)
            
            tmp11 = []
            tmp11.append(Cm[i][0][0]+int(idx_b2*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp11.append(Cm[i][0][1]+int(idx_b2*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp3.append(Cm[i][0][0]+int(idx_b*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp11 = tuple(tmp11)
            guage_point_bottom[i].append(tmp11)
            
        if int(angle[i][0]) < 90 and int(angle[i][0]) != 0:
            idx_r = guage[i][0]
            idx_l = guage[i][1]
            idx_u = guage[i][2]
            idx_b = guage[i][3]
            
            tmp = []
            tmp.append(int(Cm[i][0][0])-int(idx_r*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp.append(int(Cm[i][0][1])-int(idx_r*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp.append(int(Cm[i][0][0])-int(idx_r*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp = tuple(tmp)
            guage_point_right[i].append(tmp)
            
            tmp1 = []  
            tmp1.append(int(Cm[i][0][0])+int(idx_l*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp1.append(int(Cm[i][0][1])+int(idx_l*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp1.append(int(Cm[i][0][0])+int(idx_l*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp1 = tuple(tmp1)
            guage_point_left[i].append(tmp1)
            
            tmp2 = []
            tmp2.append(int(Cm[i][0][0])+int(idx_u*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp2.append(int(Cm[i][0][1])-int(idx_u*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp2.append(int(Cm[i][0][0])+int(idx_u*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp2 = tuple(tmp2)
            guage_point_up[i].append(tmp2)
            
            tmp3 = []
            tmp3.append(Cm[i][0][0]-int(idx_b*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp3.append(Cm[i][0][1]+int(idx_b*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp3.append(Cm[i][0][0]-int(idx_b*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp3 = tuple(tmp3)
            guage_point_bottom[i].append(tmp3)
        
        #print("Current index:",i)
    
    #idx = 35
    #print(Cm[5][0])
    for idx in range(0,len(guage)):    
        img_ff = cv2.circle(img_copy,guage_point_right[idx][0],5,(0,0,255),-1)
        img_ff = cv2.circle(img_copy,guage_point_left[idx][0],5,(0,0,255),-1)
        img_ff = cv2.circle(img_copy,guage_point_up[idx][0],5,(0,0,255),-1)
        img_ff = cv2.circle(img_copy,guage_point_bottom[idx][0],5,(0,0,255),-1)
     
    '''
    #print(Cm[0][0])
    #print(guage)        
    print("\nR:",guage_point_right)
    print("\nL:",guage_point_left)
    print("\nU:",guage_point_up)
    print("\nB:",guage_point_bottom)
    '''
    return img_ff, guage_point_right, guage_point_left, guage_point_up, guage_point_bottom

def Guage_Points_fib(img_int,Cm,angle,guage,long_x_vector,long_y_vector,short_x_vector,short_y_vector,long,short,pixel_size):
    guage_point_up = []
    guage_point_bottom = []
    guage_point_right = []
    guage_point_left = []
    img_copy = img_int.copy()
    print("\n")
    for i in range(0,len(Cm)):
        guage_point_up.append(i*[])
        guage_point_bottom.append(i*[])
        guage_point_right.append(i*[])
        guage_point_left.append(i*[])
    
        if int(angle[i][0]) == 0 and int(angle[i][0]) != 90:
            idx_r = guage[i][0]
            idx_l = guage[i][1]
            idx_u = guage[i][2]
            idx_b = guage[i][3]
            
            tmp = []
            tmp.append(Cm[i][0][0]+idx_r)
            tmp.append(Cm[i][0][1])
            #tmp.append(Cm[i][0][0]+idx_r)
            tmp = tuple(tmp)
            guage_point_right[i].append(tmp)
            
            tmp1 = []
            tmp1.append(Cm[i][0][0]-idx_l)
            tmp1.append(Cm[i][0][1])
            #tmp1.append(Cm[i][0][0]-idx_l)
            tmp1 = tuple(tmp1)
            guage_point_left[i].append(tmp1)
            
            tmp2 = []
            tmp2.append(Cm[i][0][0])
            tmp2.append(Cm[i][0][1]-idx_u)
            #tmp2.append(Cm[i][0][0])
            tmp2 = tuple(tmp2)
            guage_point_up[i].append(tmp2)
            
            tmp3 = []
            tmp3.append(Cm[i][0][0])
            tmp3.append(Cm[i][0][1]+idx_b)
            #tmp3.append(Cm[i][0][0])
            tmp3 = tuple(tmp3)
            guage_point_bottom[i].append(tmp3)

        
        if int(angle[i][0]) == 90 and int(angle[i][0]) != 0:
            idx_r1 = guage[i][2]
            idx_l1 = guage[i][3]
            idx_u1 = guage[i][1]
            idx_b1 = guage[i][0]
            
            tmp4 = []
            
            tmp4.append(Cm[i][0][0])
            tmp4.append(Cm[i][0][1]+idx_r1)
            #tmp.append(Cm[i][0][1]+idx_r)
            tmp4 = tuple(tmp4)
            guage_point_right[i].append(tmp4)
            
            tmp5 = []
            
            tmp5.append(Cm[i][0][0])
            tmp5.append(Cm[i][0][1]-idx_l1)
            #tmp1.append(Cm[i][0][1]-idx_l)
            tmp5 = tuple(tmp5)
            guage_point_left[i].append(tmp5)
            
            tmp6 = []
            
            tmp6.append(Cm[i][0][0]-idx_u1)
            tmp6.append(Cm[i][0][1])
            #tmp2.append(Cm[i][0][1])
            tmp6 = tuple(tmp6)
            guage_point_up[i].append(tmp6)
            
            tmp7 = []
            
            tmp7.append(Cm[i][0][0]+idx_b1)
            tmp7.append(Cm[i][0][1])
            #tmp3.append(Cm[i][0][1])
            tmp7 = tuple(tmp7)
            guage_point_bottom[i].append(tmp7)
        
        if int(angle[i][0]) > 90:
            idx_r2 = guage[i][0]
            idx_l2 = guage[i][1]
            idx_u2 = guage[i][2]
            idx_b2 = guage[i][3]
            
            tmp8 = []
            tmp8.append(int(Cm[i][0][0])+int(idx_r2*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp8.append(int(Cm[i][0][1])-int(idx_r2*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp.append(int(Cm[i][0][0])+int(idx_r*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp8 = tuple(tmp8)
            guage_point_right[i].append(tmp8)
            
            tmp9 = [] 
            tmp9.append(int(Cm[i][0][0])-int(idx_l2*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp9.append(int(Cm[i][0][1])+int(idx_l2*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp1.append(int(Cm[i][0][0])-int(idx_l*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp9 = tuple(tmp9)
            guage_point_left[i].append(tmp9)
            
            tmp10 = []
            tmp10.append(int(Cm[i][0][0])-int(idx_u2*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp10.append(int(Cm[i][0][1])-int(idx_u2*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp2.append(int(Cm[i][0][0])-int(idx_u*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp10 = tuple(tmp10)
            guage_point_up[i].append(tmp10)
            
            tmp11 = []
            tmp11.append(Cm[i][0][0]+int(idx_b2*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp11.append(Cm[i][0][1]+int(idx_b2*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp3.append(Cm[i][0][0]+int(idx_b*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp11 = tuple(tmp11)
            guage_point_bottom[i].append(tmp11)
            
        if int(angle[i][0]) < 90 and int(angle[i][0]) != 0:
            idx_r = guage[i][0]
            idx_l = guage[i][1]
            idx_u = guage[i][2]
            idx_b = guage[i][3]
            
            tmp = []
            tmp.append(int(Cm[i][0][0])-int(idx_r*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp.append(int(Cm[i][0][1])-int(idx_r*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp.append(int(Cm[i][0][0])-int(idx_r*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp = tuple(tmp)
            guage_point_right[i].append(tmp)
            
            tmp1 = []  
            tmp1.append(int(Cm[i][0][0])+int(idx_l*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp1.append(int(Cm[i][0][1])+int(idx_l*(np.abs(short_y_vector[i])*pixel_size/short[i])))
            #tmp1.append(int(Cm[i][0][0])+int(idx_l*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp1 = tuple(tmp1)
            guage_point_left[i].append(tmp1)
            
            tmp2 = []
            tmp2.append(int(Cm[i][0][0])+int(idx_u*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp2.append(int(Cm[i][0][1])-int(idx_u*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp2.append(int(Cm[i][0][0])+int(idx_u*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp2 = tuple(tmp2)
            guage_point_up[i].append(tmp2)
            
            tmp3 = []
            tmp3.append(Cm[i][0][0]-int(idx_b*np.abs(long_x_vector[i])*pixel_size/long[i]))
            tmp3.append(Cm[i][0][1]+int(idx_b*(np.abs(long_y_vector[i])*pixel_size/long[i])))
            #tmp3.append(Cm[i][0][0]-int(idx_b*np.abs(short_x_vector[i])*pixel_size/short[i]))
            tmp3 = tuple(tmp3)
            guage_point_bottom[i].append(tmp3)
        
    for idx in range(0,len(guage)):    
        img_ff = cv2.circle(img_copy,guage_point_right[idx][0],5,(0,0,255),-1)
        img_ff = cv2.circle(img_copy,guage_point_left[idx][0],5,(0,0,255),-1)
        img_ff = cv2.circle(img_copy,guage_point_up[idx][0],5,(0,0,255),-1)
        img_ff = cv2.circle(img_copy,guage_point_bottom[idx][0],5,(0,0,255),-1)
      
    '''
    #print(Cm[0][0])
    #print(guage)        
    print("\nR:",guage_point_right)
    print("\nL:",guage_point_left)
    print("\nU:",guage_point_up)
    print("\nB:",guage_point_bottom)
    '''
    return img_ff, guage_point_right, guage_point_left, guage_point_up, guage_point_bottom
    
def Zip_and_Sort_Information(position,size1,size2,axis1,axis2,angle,cnts):
    sort_array = []
    for i in range(0,len(position)):
        sort_array.append(i*[])
    
    for j in range(0,len(sort_array)):
        sort_array[j].append(position[j])
        sort_array[j].append(size1[j])
        sort_array[j].append(size2[j])
        sort_array[j].append(axis1[j])
        sort_array[j].append(axis2[j])
        sort_array[j].append(angle[j])
        sort_array[j].append(cnts[j])
        #sort_array[j].append(axis2_len[j])
    
    sort_array_update = sorted(sort_array, key = lambda sort_array: sort_array[0][0][0])    
    sort_array_update = sorted(sort_array_update, key = lambda sort_array_update: sort_array_update[0][0][1])

    return sort_array_update
    
def Zip_and_Sort_Information_Mask_fib(center_of_mass,long_axis,short_axis,long_axis_vector, short_axis_vector, angle_PB, contours_mask_alignment, center_of_mass_FIB, contours_fib_alignment):
    sort_array_mask = []
    sort_array_fib = []
    for i in range(0,len(center_of_mass)):
        sort_array_mask.append(i*[])
    
    for j in range(0,len(sort_array_mask)):
        sort_array_mask[j].append(center_of_mass[j])
        sort_array_mask[j].append(long_axis[j])
        sort_array_mask[j].append(short_axis[j])
        sort_array_mask[j].append(long_axis_vector[j])
        sort_array_mask[j].append(short_axis_vector[j])
        sort_array_mask[j].append(angle_PB[j])
        sort_array_mask[j].append(contours_mask_alignment[j])
        #sort_array[j].append(axis2_len[j])
    
    sort_array_update_mask = sorted(sort_array_mask, key = lambda sort_array_mask: sort_array_mask[0][0][0])    
    sort_array_update_mask = sorted(sort_array_update_mask, key = lambda sort_array_update_mask: sort_array_update_mask[0][0][1])
    
    for m in range(0,len(center_of_mass_FIB)):
        sort_array_fib.append(m*[])
    
    for n in range(0,len(sort_array_fib)):
        sort_array_fib[n].append(center_of_mass_FIB[n])
        sort_array_fib[n].append(contours_fib_alignment[n])
        #sort_array[j].append(axis2_len[j])
    
    sort_array_update_fib = sorted(sort_array_fib, key = lambda sort_array_fib: sort_array_fib[0][0][0])    
    sort_array_update_fib = sorted(sort_array_update_fib, key = lambda sort_array_update_fib: sort_array_update_fib[0][0][1])
    '''
    for a in range(0,len(sort_update_mask)):
        
        
    for b in range(0,len(sort_update_fib)):
    '''

    return sort_array_update_mask, sort_array_update_fib

def Fib_insert_information(info_fib,long_fib,short_fib):
    long = []
    short = []
    for i in range(0,len(info_fib)):
        long.append(i*[])
        short.append(i*[])
        info_fib[i].append(long_fib[i])
        info_fib[i].append(short_fib[i])
        
    return info_fib

def Put_text(input_img, position, information, information2, information3):
    print("\nCenter of mass:",position[0][0][0])
    print("Long axis:",information[0][0])
    print("Short axis:",information2[0][0])
    print("Angle PB:",information3[0][0])
    
    for pattern in range(0,len(information)):
        #print(pattern)
        #cv2.putText(input_img, "(" + str(np.round(information[pattern][0],2))+" nm" + " , " +str(np.round(information2[pattern][0],2))+" nm"+")" ,(position[pattern][0][0]-65, position[pattern][0][1]), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 175, 0), 2, cv2.LINE_AA)
        #cv2.putText(input_img, "(" + str(np.round(information[pattern][0],2))+ " , " +str(np.round(information2[pattern][0],2))+")" ,(position[pattern][0][0]-65, position[pattern][0][1]), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 175, 0), 2, cv2.LINE_AA)
        cv2.putText(input_img, str(np.round(information[pattern][0],2)) ,(position[pattern][0][0]-15, position[pattern][0][1]), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 175, 0), 2, cv2.LINE_AA)
        #cv2.putText(input_img, str(int(information3[pattern][0]))+" deg",(position[pattern][0][0]-15, position[pattern][0][1]+15), cv2.FONT_HERSHEY_PLAIN,0.5, (0, 175, 0), 1, cv2.LINE_AA)
    
    return input_img

def Alignment_Matrix(Cm_mask, Cm_fib, info, info2, grid_idx, grid_idx2, pixel_size, pitch):
    grid_unit_pix = int(pitch / pixel_size)
    grid_list = []
    Cm_cal_x = []
    Cm_cal_y = []
    tmp_cal_x = []
    tmp_cal_x1 = []
    tmp_cal_y = []
    tmp_cal_y1 = []
    Cm_m_x = []
    Cm_m_y = []
    Cm_f_x = []
    Cm_f_y = []
    Cm_fib_restort = []
    Cm_fib_restort_final = []
    Cm_mask_restort = []
    Cm_mask_restort_final = []
    long = []
    short = []
    long_fib = []
    short_fib = []
    angle = []
    grid_matrix_x = np.zeros((grid_idx2,grid_idx))
    grid_matrix_y = np.zeros((grid_idx2,grid_idx))
    target_x_array = np.zeros((grid_idx*grid_idx2,1))
    target_y_array = np.zeros((grid_idx*grid_idx2,1))
    long_matrix = np.zeros((grid_idx2,grid_idx))
    short_matrix = np.zeros((grid_idx2,grid_idx))
    long_matrix_fib = np.zeros((grid_idx2,grid_idx))
    short_matrix_fib = np.zeros((grid_idx2,grid_idx))
    angle_matrix = np.zeros((grid_idx2,grid_idx))
    
    Cx_init = Cm_mask[0][0][0]
    Cy_init = Cm_mask[0][0][1]
    print("\nCx_init:",Cx_init)
    print("\nCy_init:",Cy_init)
    for i in range(0,grid_idx2):
        for j in range(0,grid_idx):
            tmp_x = Cx_init + j * grid_unit_pix
            tmp_y = Cy_init + i * grid_unit_pix
            grid_matrix_x[i][j] = tmp_x
            grid_matrix_y[i][j] = tmp_y
    
    for a in range(0,grid_idx):
        tmp = Cx_init + a * grid_unit_pix
        tmp_cal_x.append(tmp)
        tmp2 = Cy_init + a * grid_unit_pix
        tmp_cal_y.append(tmp2)
    tmp_cal_y_new = []
    while len(tmp_cal_x) < grid_idx * grid_idx:
        tmp_cal_x.extend(tmp_cal_x)
    for c in tmp_cal_y:
        tmp_cal_y_new.extend([c] * grid_idx)
    '''    
    target_x_list = tmp_cal_x[:grid_idx*grid_idx]
    target_y_list = tmp_cal_y_new
    #print("\n",tmp_cal_y)    
    #print("\nTarget x pos:",target_x_list)
    #print("\nTarget y pos:",target_y_list)
    target_x_array = np.array(target_x_list)
    target_y_array = np.array(target_y_list)
    target_x_array = np.reshape(target_x_array,(target_x_array.shape[0],1))
    target_y_array = np.reshape(target_y_array,(target_y_array.shape[0],1))
    #print("\nTarget x array shape:",target_x_array.shape)
    '''
    for k in range(0,len(info)):
        long.append(info[k][1][0])
        short.append(info[k][2][0])
        long_fib.append(info2[k][1][0])
        short_fib.append(info2[k][2][0])
        angle.append(info[k][5][0])
        Cm_m_x.append(info[k][0][0][0])
        Cm_m_y.append(info[k][0][0][1])
        Cm_f_x.append(info[k][0][0][0])
        Cm_f_y.append(info[k][0][0][1])
    
    #print("\nCm_m_x",Cm_m_x)
    #print("\nCm_m_y",Cm_m_y)
    print("\nFIB pts",info2[0][0])
    
       
    long = np.array(long)
    short = np.array(short)
    long_fib = np.array(long_fib)
    short_fib = np.array(short_fib)
    angle = np.array(angle)
    long = np.reshape(long,(long.shape[0],1)) 
    short = np.reshape(short,(short.shape[0],1))
    long_fib = np.reshape(long_fib,(long_fib.shape[0],1)) 
    short_fib = np.reshape(short_fib,(short_fib.shape[0],1))
    angle = np.reshape(angle,(angle.shape[0],1))
    Cm_m_x = np.array(Cm_m_x)
    Cm_m_x = np.reshape(Cm_m_x,(Cm_m_x.shape[0],1)) 
    Cm_m_y = np.array(Cm_m_y)
    Cm_m_y = np.reshape(Cm_m_y,(Cm_m_y.shape[0],1)) 
    Cm_f_x = np.array(Cm_f_x)
    Cm_f_x = np.reshape(Cm_f_x,(Cm_f_x.shape[0],1)) 
    Cm_f_y = np.array(Cm_f_y)
    Cm_f_y = np.reshape(Cm_f_y,(Cm_f_y.shape[0],1)) 
    Cm_fib_1 = Cm_fib
   
    for m in range(0,grid_idx2):
        for n in range(0,grid_idx):
            long_matrix[m][n] = long[grid_idx*m + n][0]
            long_matrix_fib[m][n] = long_fib[grid_idx*m + n][0]
            short_matrix[m][n] = short[grid_idx*m + n][0]
            short_matrix_fib[m][n] = short_fib[grid_idx*m + n][0]
            angle_matrix[m][n] = angle[grid_idx*m + n][0]
         
    for m_ in range(0,grid_idx):
        Cm_tmp_fib = Cm_fib[grid_idx * m_ : grid_idx * (m_ + 1)]
        Cm_tmp_fib_update = sorted(Cm_tmp_fib, key = lambda Cm_tmp_fib: Cm_tmp_fib[0][0])
        Cm_tmp_mask = Cm_mask[grid_idx * m : grid_idx * (m + 1)]
        Cm_tmp_mask_update = sorted(Cm_tmp_mask, key = lambda Cm_mask_fib: Cm_mask_fib[0][0])
    print("\nCm tmp fib",Cm_tmp_fib_update)
    '''    
    for o in range(0,grid_idx):
        Cm_tmp_fib_update = np.array(Cm_tmp_fib_update)
        Cm_fib_restort.append(tuple(list(Cm_tmp_fib_update[o][0])))
        Cm_tmp_mask_update = np.array(Cm_tmp_mask_update)
        Cm_mask_restort.append(tuple(list(Cm_tmp_mask_update[o][0])))
    
    tmp = []
    tmp1 = []
    for o in range(0,len(Cm_fib)):
        tmp.append(o*[])
        tmp[o].append(Cm_fib_restort[o])
    for p in range(0,len(Cm_mask)):
        tmp1.append(p*[])
        tmp1[p].append(Cm_mask_restort[p])
    
    Cm_fib_restort_final = tmp 
    Cm_mask_restort_final = tmp1        
    #print("\n",Cm_fib_restort_final)
    '''
    return grid_matrix_x, grid_matrix_y, Cm_m_x, Cm_m_y, Cm_f_x, Cm_f_y, long_matrix, short_matrix, long_matrix_fib, short_matrix_fib, angle_matrix#, Cm_fib_restort_final, Cm_mask_restort_final
    
#return grid_matrix_x, grid_matrix_y
#def Alignment_Matrix_Coloring(img_single,img_l1,img_l2):

def Analysis_Size_Distriburion(mask_size,exp_size,min_,max_,delta):
    mask_dis = []
    exp_dis = []
    step = int((max_ - min_) / delta)
    '''
    # 用 while loop 產生每個區間
    i = 0
    while i < step:
        start = min_ + i * delta
        end = start + delta - 1
        if end > max_:
            end = max_
        exp_dis.append(list(range(start, end + 1)))
        i += 1
    '''
    # 計算總共要幾個區間
    step = int((max_ - min_) / delta)
    exp_dis = [[] for _ in range(step)]  # 先建好空的 list
    lengths = np.zeros(step, dtype=int)  # 每個區間的長度
    # 對每個數值判斷區間
    for num in exp_size.flatten():  # flatten 把 2D 轉成 1D
        if min_ <= num <= max_:
            index = int((num - min_) // delta)  # 轉成 int
            if index >= step:
                index = step - 1  # 避免超出範圍
            exp_dis[index].append(num)
            lengths[index] += 1  # 記錄長度
    
    # 將 2D 轉成 1D
    data_flat = exp_size.flatten()
    
    # 畫直方圖，bins 對應每個區間
    bins = np.arange(min_, max_ + delta, delta)
    '''
    fig = plt.figure()        
    plt.figure(figsize=(12, 10))
    plt.hist(data_flat, bins=bins, edgecolor='black', linewidth=3, rwidth=1.5)
    plt.xlabel("CD Range(nm)",fontsize=30)
    plt.ylabel("Counts",fontsize=30)
    plt.title("Size Distribution of Training Datasets",fontsize=32)
    plt.xticks(np.arange(min_, max_+1, delta),fontsize=22)
    plt.yticks(fontsize=28)
    #plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim((0,140))
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.tick_params(axis='both', which='both', width=5)  
    plt.show() 
    '''
    return mask_dis, exp_dis, lengths

def Plot_results_map(title,X_diameter, Y_diameter, info, figuresize, colormap,label_info):
    fig = plt.figure()
    plt.figure(figsize = figuresize)
    im = plt.imshow(info, colormap, aspect = 'auto', extent = [0,X_diameter,0,Y_diameter])
    plt.title(title, fontsize = '22', y=1.025)
    
    cb = plt.colorbar(label=label_info, orientation="vertical", shrink = 1.0, pad = 0.1)
    im.figure.axes[0].tick_params(axis="both",labelsize=20)
    im.figure.axes[1].tick_params(axis="both",labelsize=20)
    cb.set_label(label=label_info,size=20)
    #plt.clim(0,-0.15)
    
    plt.xlabel('x',fontsize = '20')
    plt.ylabel('y', fontsize = '20')
    plt.xticks(fontsize = '18')
    plt.yticks(fontsize = '18')
    #plt.colorbar(label=label_info, orientation="vertical")
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.tick_params(axis='both', which='both', width=3)  
    plt.show() 
#%%Import Information 
path_img = r'D:\Master\2025_12_17'#r"D:\Master\2025_10_30\P457\Mask"
path_img_fib = r'D:\Master\2025_12_17'#r"D:\Master\2025_10_30\P457\DOE"#"D:\Master\2025_09_03\Model record\Dataset_4k\PR"
path_save = r'D:\Master\2025_12_17'#r"D:\Master\2025_10_30\P457\Lens" #Save Mask and FIB image alignment
path_save_alignment = path_save#r"D:\Master\2024_12_13\Mask11 ADI AEI\EB\Alignment" #Save Cm, guage, vertex

mask_name = 'PR_P457_hologram_LD_PR_L1_alignment.bmp'#"P457_CGH_IPC_Lens+DOE_Mask_LU" + ".png"#"ROE_target_single_LU_1"#"Mask_alignment_ROE_target_single_LU"#"L1_opc_LD"
fib_name = 'P457_hologram_LD.tif'#"Mask24_P457_CGH_IPC_new_lens+DOE_model_LU" + ".tif"#"ROE_PB +Prop_L1_ori_LU"#"ROE_PB+Prop_L2_opc_LD"
fib_etch = "P457_hologram_LD" + ".tif"
name_txt_guage = "Model_predict_DOE_P457_Unet_DL_101x101_RU_mask_alignment" + "1450_guage.txt"

color_etch = "off" #on, off ; Mean want to do etch SEM fig coloring
Layer_const = "even" #odd,even
gaussian_blur = "on" #on, off ; Blur the contours to be smooth
pad_mode = "on" #on, off ; padding image with large size  
mode = "Mask+FIB" # Mask, FIB, Mask+FIB ; pro cess  Mask, FIB, or Mask align w

etch = False #Mean want to analyze etch bias with FIB 
sort_mode = "Mask+FIB" #Mask, Mask+FIB ; Sort the each pattern in order
save = True#Save Fig
ruler = False
show = True #Show Fig
resize = False 
sort = True 

vertex_number = 4
bias_slope = 1#0.91
bias_intercept = 1#-6.48 #nm ; e.g. +5 >>> + 5 nm
area_low = 10 #FIB image contour lower limit after binarization
kernel_size = (21,21) #Gaussian blur kernel size
pixel_size = 3.4668#2.0825#3.4668
grid_idx = 17#30 #Want to analyze with map, then define this parameter in x direction
grid_idx2 = 29#26 #Want to analyze with map, then define this parameter in y direction
pitch = 457 #Want to analyze with map, then define this parameter in terms of lattice pitch
thr_set = 0 #Binary image threshold

#Cut image parameters
mask_x_l = 0
mask_x_r = 0#75#15
mask_y_u = 0
mask_y_d = 0
fib_x_l = 0  
fib_x_r = 0
fib_y_u = 0
fib_y_d = 220#220 
crop = 0#285
#Pad image parameters
mask_pad_x_l = 0#+75#63
mask_pad_x_r = 0#75+75#15#63
mask_pad_y_u = 0#15
mask_pad_y_d = 0#15
fib_pad_x_l = 0
fib_pad_x_r = 0
fib_pad_y_u = 0
fib_pad_y_d = 0
#%% Main Process
#Find_Contours(path_img,mask_name,resize)
if __name__ == "__main__":
    print("\n------------------------------------Start All Process----------------------------------")
    if mode == "Mask":
        start = time.time()
        img, input_img, input_img2, input_img3, input_img4, img_gray, img_final, img_final_fib, contours, contours_fib = Find_Contours(path_img,path_img_fib,mask_name,fib_name,resize,thr_set,area_low,gaussian_blur,kernel_size, \
                                                                                                                             mask_x_l,mask_x_r,mask_y_u,mask_y_d,fib_x_l,fib_x_r,fib_y_u,fib_y_d,mask_pad_x_l, \
                                                                                                                             mask_pad_x_r,mask_pad_y_u,mask_pad_d,fib_pad_x_l,fib_pad_x_r,fib_pad_y_u,fib_pad_y_d,mode,pad_mode)
        input_img5 = img.copy()
        input_img8 = img.copy()
        #input_img9 = img.copy()
        end = time.time()
        print("\nFind mask contours costs time = ",end - start," s")
    if mode == "FIB":
        start = time.time()
        img, input_img, input_img2, input_img3, img_gray, img_final, img_final_fib, contours, contours_fib = Find_Contours(path_img,path_img_fib,mask_name,fib_name,resize,thr_set,area_low,gaussian_blur,kernel_size, \
                                                                                                                           mask_x_l,mask_x_r,mask_y_u,mask_y_d,fib_x_l,fib_x_r,fib_y_u,fib_y_d,mask_pad_x_l, \
                                                                                                                           mask_pad_x_r,mask_pad_y_u,mask_pad_y_d,fib_pad_x_l,fib_pad_x_r,fib_pad_y_u,fib_pad_y_d,mode,pad_mode)
        input_img5 = img.copy()
        end = time.time()
        print("\nFind fib contours costs time = ",end - start," s")
    if mode == "Mask+FIB":
        img, input_img, input_img_fib, img_gray, img_final, img_final_fib, contours, contours_fib = Find_Contours(path_img,path_img_fib,mask_name,fib_name,resize,thr_set,area_low,gaussian_blur,kernel_size, \
                                                                                                                  mask_x_l,mask_x_r,mask_y_u,mask_y_d,fib_x_l,fib_x_r,fib_y_u,fib_y_d,mask_pad_x_l, 
                                                                                                                  mask_pad_x_r,mask_pad_y_u,mask_pad_y_d,fib_pad_x_l,fib_pad_x_r,fib_pad_y_u,fib_pad_y_d,mode,pad_mode)   
        if etch == True:
            _, _, input_img_fib_etch, _, _, img_final_fib_etch, _, contours_fib_etch = Find_Contours(path_img,path_img_fib,mask_name,fib_etch,resize,thr_set,area_low,gaussian_blur,kernel_size, \
                                                                                                     mask_x_l,mask_x_r,mask_y_u,mask_y_d,fib_x_l,fib_x_r,fib_y_u,fib_y_d,mask_pad_x_l, 
                                                                                                     mask_pad_x_r,mask_pad_y_u,mask_pad_y_d,fib_pad_x_l,fib_pad_x_r,fib_pad_y_u,fib_pad_y_d,mode,pad_mode)
            img_final3_FIB_etch, center_of_mass_FIB_etch = Find_Center_of_Mass(input_img_fib_etch, contours_fib_etch)
            
        start = time.time()
        img_final3, center_of_mass = Find_Center_of_Mass(img_final, contours)
        img_final3_FIB, center_of_mass_FIB = Find_Center_of_Mass(input_img_fib, contours_fib)
        
        if color_etch == "on":
            print("\nEtching Layer Start Coloring...")
            tmp = []
            tmp1 = []
            tmp2 = []
            idx = 30
            for a in range(0,len(contours_fib)):
                tmp.append(a*[])
                tmp1.append(a*[])
                tmp2.append(a*[])
            for b in range(0,len(contours_fib)):
                tmp[b].append(center_of_mass_FIB[b])
                tmp[b].append(contours_fib[b])
                tmp1[b].append(center_of_mass_FIB[b][0][0])
                tmp2[b].append(center_of_mass_FIB[b][0][1])
            
            sort_tmp = sorted(tmp, key = lambda tmp: tmp[0][0][0])    
            sort_tmp = sorted(sort_tmp, key = lambda sort_tmp: sort_tmp[0][0][1])
            sort_tmp1 = []
            
            print("\nMin x:",np.min(tmp1))
            print("\nMax x:",np.max(tmp1))
            print("\nMin y:",np.min(tmp2))
            print("\nMax y:",np.max(tmp2))
            print("\n......")
            
            delta_grid = int((np.max(tmp2) - np.min(tmp2)) / idx)
            print("\nDelta grid:",delta_grid)
            tmp_list = []
            sort_tmp_update = []
            sort_tmp_update2 = []
            idx_list = []
            contour_arrange = []
            contour_arrange_L1 = []
            contour_arrange_L2 = []
            center_of_mass_fib_arrange = []
            center_of_mass_fib_arrange_L1 = []
            for j in range(0,idx+1):
                tmp_val = np.min(tmp2) + j * delta_grid
                idx_list.append(tmp_val)
            
            for c in range(0,len(idx_list)):
                tmp_list.append(c*[])
                for d in range(0,len(sort_tmp)):
                    if idx_list[c] <= sort_tmp[d][0][0][1] < idx_list[c] + delta_grid:
                        tmp_list[c].append(sort_tmp[d])
                        tmp_list[c] = sorted(tmp_list[c], key = lambda tmp_list: tmp_list[0][0][0]) 
                
                if Layer_const == "odd":
                    for e in range(0,len(tmp_list[c])):
                        sort_tmp_update.append(tmp_list[c][e])
                        if c % 2 == 0:
                            if e % 2 == 0:
                                sort_tmp_update2.append(tmp_list[c][e])
                        if c % 2 == 1:
                            if e % 2 == 1: 
                                sort_tmp_update2.append(tmp_list[c][e])
                if Layer_const == "even":
                    for e in range(0,len(tmp_list[c])):
                        sort_tmp_update.append(tmp_list[c][e])
                        if c % 2 == 0:
                            if e % 2 == 1:
                                sort_tmp_update2.append(tmp_list[c][e])
                        if c % 2 == 1:
                            if e % 2 == 0: 
                                sort_tmp_update2.append(tmp_list[c][e])    
                    #if c % 2 == 1:
                    #   if e % 2 == 0:
                    #       sort_tmp_update2.append(tmp_list[c][e])
                    
            for z in range(0,len(sort_tmp_update)):
                contour_arrange.append(sort_tmp_update[z][1])
                center_of_mass_fib_arrange.append(sort_tmp_update[z][0])
            for z1 in range(0,len(sort_tmp_update2)):
                contour_arrange_L1.append(sort_tmp_update2[z1][1])
                center_of_mass_fib_arrange_L1.append(sort_tmp_update[z1][0])
                
            img_uu = np.zeros((img_final.shape[0],img_final.shape[1]), dtype = np.uint8)
            img_uu_1 = np.zeros((img_final.shape[0],img_final.shape[1]), dtype = np.uint8)
            img_uu_2 = np.zeros((img_final.shape[0],img_final.shape[1]), dtype = np.uint8)
            img_uu = cv2.drawContours(img_uu, contour_arrange, -1, (255,255,255), -1)
            img_uu_1 = cv2.drawContours(img_uu_1, contour_arrange_L1, -1, (255,255,255), -1)
            
            img_final_fib = img_uu_1
            contours_fib = contour_arrange_L1
            center_of_mass_FIB = center_of_mass_fib_arrange_L1
            
            
            cv2.namedWindow('Re-sort fib: ',0)
            cv2.imshow('Re-sort fib: ',img_uu)
            cv2.waitKey(0)
            
            cv2.namedWindow('Re-sort fib L1: ',0)
            cv2.imshow('Re-sort fib L1: ',img_uu_1)
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            
            print("\nRe-arrange contour len:",len(contour_arrange))
            print("\nRe-arrange L1 contour len:",len(contour_arrange_L1)) 
            
        if color_etch == "off":
            img_final_fib = img_final_fib
            contours_fib = contours_fib
            center_of_mass_FIB = center_of_mass_FIB
            
        img_Cm =  Cm_Position(center_of_mass,center_of_mass_FIB,img_final_fib)
        end = time.time()
        print("\nFind Cm costs time = ",end - start," s")
        
        start = time.time()
        img_mask_alignment, img_fib_alignment, contours_mask_alignment, contours_fib_alignment = Alignment_Mask_FIB(img_gray,img_final_fib,contours,contours_fib,center_of_mass,center_of_mass_FIB)
        
        end = time.time()
        print("\nMask and FIB image alignment costs time = ",end - start," s")

        img_mask_alignment1 = img_mask_alignment.copy()
        img_mask_alignment2 = img_mask_alignment.copy()
        img_mask_alignment3 = img_mask_alignment.copy()
        
        img_fib_alignment1 = img_fib_alignment.copy()
        img_fib_alignment2 = img_fib_alignment.copy()
        img_fib_alignment3 = img_fib_alignment.copy()
        
        img_mask_alignment_RGB = cv2.cvtColor(img_mask_alignment, cv2.COLOR_GRAY2RGB)
        img_mask_alignment_RGB1 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB2 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB3 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB4 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB5 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB6 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB7 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB8 = img_mask_alignment_RGB.copy()
        img_mask_alignment_RGB9 = img_mask_alignment_RGB.copy()
        
        img_fib_alignment_RGB = cv2.cvtColor(img_fib_alignment, cv2.COLOR_GRAY2RGB)
        img_fib_alignment_RGB1 = img_fib_alignment_RGB.copy()
        img_fib_alignment_RGB2 = img_fib_alignment_RGB.copy()
        img_fib_alignment_RGB3 = img_fib_alignment_RGB.copy()
        
        
        start = time.time()
        img_mask_alignment_RGB1, center_of_mass = Find_Center_of_Mass(img_mask_alignment_RGB1, contours_mask_alignment)
        img_fib_alignment_RGB1, center_of_mass_FIB = Find_Center_of_Mass(img_fib_alignment_RGB1, contours_fib_alignment)
        
        if etch == True:
            _, img_fib_etch_alignment, _, contours_fib_etch_alignment = Alignment_Mask_FIB(img_mask_alignment,img_final_fib_etch,contours_mask_alignment,contours_fib_etch,center_of_mass,center_of_mass_FIB_etch)
            img_fib_etch_alignment1 = img_fib_etch_alignment.copy()
            img_fib_etch_alignment2 = img_fib_etch_alignment.copy()
            img_fib_etch_alignment3 = img_fib_etch_alignment.copy()
            img_fib_etch_alignment_RGB = cv2.cvtColor(img_fib_etch_alignment, cv2.COLOR_GRAY2RGB)
        
            #if etch == True:
            img_fib_etch_alignment_RGB1 = img_fib_etch_alignment_RGB.copy()
            img_fib_etch_alignment_RGB2 = img_fib_etch_alignment_RGB.copy()
            img_fib_etch_alignment_RGB3 = img_fib_etch_alignment_RGB.copy()
            img_fib_etch_alignment_RGB1, center_of_mass_FIB_etch = Find_Center_of_Mass(img_fib_etch_alignment_RGB1, contours_fib_etch_alignment)
        end = time.time()
        print("\nFind new Cm costs time = ",end - start," s")
        
        start = time.time()
        points_array_list, points_array_x_list, points_array_y_list, vertex_pattern_list, img_final_vertex = Find_Vertex(contours_mask_alignment,img_mask_alignment_RGB2,img_mask_alignment_RGB3,img_mask_alignment_RGB4,mode,vertex_number)
    
        end = time.time()
        print("\nFind mask vertex costs time = ",end - start," s")
        
        start = time.time()
        var_x1, var_x2, var_y1, var_y2, vector1, vector2, len_vector1_pixel, len_vector2_pixel, len_vector1, len_vector2, long_axis_vector, short_axis_vector, long_axis, short_axis, img_final_axis = Axis_Extraction(pixel_size,vertex_pattern_list,img_mask_alignment_RGB5,vertex_number)
        end = time.time()
        print("\nAxis extraction costs time = ",end - start," s")
        
        start = time.time()
        array_vector1, array_vector2, array_vector3, len_product, len_product_horizontal, dot_product, dot_product_horizontal, angle_fin, angle_PB = Get_Angle(pixel_size, vector1, vector2, short_axis_vector, len_vector1, len_vector2, short_axis)
        print("\nLength calculation and angle extraction costs time = ",end - start," s")
        img_final_ellipse = draw_ellipse(img_mask_alignment_RGB6, short_axis, long_axis, center_of_mass, angle_PB, pixel_size)
            
    else:
        if mode == "Mask":
            start = time.time()
            points_array_list, points_array_x_list, points_array_y_list, vertex_pattern_list, img_final2 = Find_Vertex(contours,img,input_img2,input_img2,mode,vertex_number)
            end = time.time()
            print("\nFind mask vertex costs time = ",end - start," s")
            
            start = time.time()
            img_final3, center_of_mass = Find_Center_of_Mass(input_img3,contours)
            end = time.time()
            print("\nFind Cm costs time = ",end - start," s")
        
            start = time.time()
            var_x1, var_x2, var_y1, var_y2, vector1, vector2, len_vector1_pixel, len_vector2_pixel, len_vector1, len_vector2, long_axis_vector, short_axis_vector, long_axis, short_axis, img_final5 = Axis_Extraction(pixel_size,vertex_pattern_list,input_img5,vertex_number)
            end = time.time()
            print("\nAxis extraction costs time = ",end - start," s")
            
            start = time.time()
            len_vector_horizontal = 1
            array_vector1, array_vector2, array_vector3, len_product, len_product_horizontal, dot_product, dot_product_horizontal, angle_fin, angle_PB = Get_Angle(pixel_size, vector1, vector2, short_axis_vector, len_vector1, len_vector2, short_axis)
            print("\nLength calculation and angle extraction costs time = ",end - start," s")
            input_img4 = input_img5
            img_final4 = draw_ellipse(input_img4, short_axis, long_axis, center_of_mass, angle_PB, pixel_size)
            
        if mode == "FIB":
            start = time.time()
            points_array_list, points_array_x_list, points_array_y_list, vertex_pattern_list, img_final2 = Find_Vertex(contours_fib,img,input_img2,input_img2,mode,vertex_number)
            end = time.time()
            print("\nFind fib vertex costs time = ",end - start," s")
            
            start = time.time()
            img_final3, center_of_mass = Find_Center_of_Mass(input_img3,vertex_pattern_list,contours)
            end = time.time()
            print("\nFind Cm costs time = ",end - start," s")
        
            start = time.time()
            var_x1, var_x2, var_y1, var_y2, vector1, vector2, len_vector1_pixel, len_vector2_pixel, len_vector1, len_vector2, long_axis_vector, short_axis_vector, long_axis, short_axis, img_final5 = Axis_Extraction(pixel_size,vertex_pattern_list,input_img5,vertex_number)
            end = time.time()
            print("\nAxis extraction costs time = ",end - start," s")
            
            start = time.time()
            len_vector_horizontal = 1
            array_vector1, array_vector2, array_vector3, len_product, len_product_horizontal, dot_product, dot_product_horizontal, angle_fin, angle_PB, angle_fin_correct, angle_PB_correct = Get_Angle(pixel_size, vector1, vector2, short_axis_vector, len_vector1, len_vector2, short_axis)
            print("\nLength calculation and angle extraction costs time = ",end - start," s")
            input_img4 = input_img5
            img_final4 = draw_ellipse(input_img4, short_axis, long_axis, center_of_mass, angle_PB_correct, pixel_size)
            

    if sort == True:
        if sort_mode == "Mask":
            sort_update = Zip_and_Sort_Information(center_of_mass,long_axis,short_axis,long_axis_vector, short_axis_vector, angle_PB, contours)
            input_img9 = img.copy()
            img9_gray = cv2.cvtColor(input_img9,cv2.COLOR_BGR2GRAY)
            print("\nImage 9 shape:",img9_gray.shape)
            img9_gray = img9_gray / 255
            Cm_sort, long_x_vector_sort, long_y_vector_sort, long_len_sort, short_x_vector_sort, short_y_vector_sort, short_len_sort, angle_sort, guage_list_sort, guage_long, guage_short, guage_r_l, guage_u_b = Guage_List_Generation_mask(img9_gray,input_img9,sort_update,pixel_size)
            img_ff, guage_point_right, guage_point_left, guage_point_up, guage_point_bottom = Guage_Points_mask(img,Cm_sort,angle_sort,guage_list_sort, long_x_vector_sort, long_y_vector_sort, short_x_vector_sort, short_y_vector_sort, long_len_sort, short_len_sort,pixel_size)
        
        if sort_mode == "Mask+FIB":
            sort_update_mask, sort_update_fib = Zip_and_Sort_Information_Mask_fib(center_of_mass,long_axis,short_axis,long_axis_vector, short_axis_vector, angle_PB, contours_mask_alignment, center_of_mass_FIB, contours_fib_alignment)
            img_mask_alignment1 = img_mask_alignment1 / 255
            
            Cm_sort_mask, long_x_vector_sort_mask, long_y_vector_sort_mask, long_len_sort_mask, short_x_vector_sort_mask, short_y_vector_sort_mask, short_len_sort_mask, angle_sort_mask, guage_list_sort_mask, guage_long_mask, guage_short_mask, guage_r_l_mask, guage_u_b_mask = Guage_List_Generation_mask(img_mask_alignment1,img_mask_alignment2,sort_update_mask,pixel_size)
            img_ff_mask, guage_point_right_mask, guage_point_left_mask, guage_point_up_mask, guage_point_bottom_mask = Guage_Points_mask(img_mask_alignment_RGB7,Cm_sort_mask,angle_sort_mask,guage_list_sort_mask, long_x_vector_sort_mask, long_y_vector_sort_mask, short_x_vector_sort_mask, short_y_vector_sort_mask, long_len_sort_mask, short_len_sort_mask,pixel_size)

            img_fib_alignment1 = img_fib_alignment1 / 255
            Cm_sort_fib, long_x_vector_sort_fib, long_y_vector_sort_fib, long_len_sort_fib, short_x_vector_sort_fib, short_y_vector_sort_fib, short_len_sort_fib, angle_sort_fib, guage_list_sort_fib, guage_long_fib, guage_short_fib, guage_r_l_fib, guage_u_b_fib = Guage_List_Generation_fib(
                img_fib_alignment1, img_fib_alignment2, sort_update_mask, pixel_size, sort_update_fib)
            img_ff_fib, guage_point_right_fib, guage_point_left_fib, guage_point_up_fib, guage_point_bottom_fib = Guage_Points_mask(
                img_fib_alignment_RGB2, Cm_sort_fib, angle_sort_fib, guage_list_sort_fib, long_x_vector_sort_fib, long_y_vector_sort_fib,
                short_x_vector_sort_fib, short_y_vector_sort_fib, long_len_sort_fib, short_len_sort_fib, pixel_size)
            
            if etch == True:
                sort_update_mask_, sort_update_fib_etch = Zip_and_Sort_Information_Mask_fib(center_of_mass,long_axis,short_axis,long_axis_vector, short_axis_vector, angle_PB, contours_mask_alignment, center_of_mass_FIB_etch, contours_fib_etch_alignment)
                img_fib_etch_alignment1 = img_fib_etch_alignment1 / 255
                Cm_sort_fib_etch, long_x_vector_sort_fib_etch, long_y_vector_sort_fib_etch, long_len_sort_fib_etch, short_x_vector_sort_fib_etch, short_y_vector_sort_fib_etch, short_len_sort_fib_etch, angle_sort_fib_etch, guage_list_sort_fib_etch, guage_long_fib_etch, guage_short_fib_etch, guage_r_l_fib_etch, guage_u_b_fib_etch = Guage_List_Generation_fib(
                    img_fib_etch_alignment1, img_fib_etch_alignment2, sort_update_mask, pixel_size, sort_update_fib_etch)
                
                img_ff_fib_etch, guage_point_right_fib_etch, guage_point_left_fib_etch, guage_point_up_fib_etch, guage_point_bottom_fib_etch = Guage_Points_mask(
                    img_fib_etch_alignment_RGB2, Cm_sort_fib_etch, angle_sort_fib_etch, guage_list_sort_fib_etch, long_x_vector_sort_fib_etch, long_y_vector_sort_fib_etch,
                    short_x_vector_sort_fib_etch, short_y_vector_sort_fib_etch, long_len_sort_fib_etch, short_len_sort_fib_etch, pixel_size)
                
    
            long_final_mask = np.round(pixel_size * guage_long_mask,2)  
            short_final_mask = np.round(pixel_size * guage_short_mask,2)
            long_final_fib = np.round(pixel_size * guage_long_fib,2)
            short_final_fib = np.round(pixel_size * guage_short_fib,2)
            if etch == True:
                long_final_fib_etch = np.round(pixel_size * guage_long_fib_etch,2)
                short_final_fib_etch = np.round(pixel_size * guage_short_fib_etch,2)    
            print("len long final mask:",len(long_final_mask))
            #delta_long = np.zeros((len(long_final_mask),1))
            delta_long = long_final_fib - long_final_mask
            #delta_short = np.zeros((len(short_final_mask),1))
            delta_short = short_final_fib - short_final_mask
            delta_long = np.reshape(delta_long,(1,len(long_final_mask)))
            delta_short = np.reshape(delta_short,(1,len(short_final_mask)))
            #Map_calculation(sort_update_mask,sort_update_fib,long_final_mask,short_final_mask,long_final_fib,short_final_fib,pixel_size,pitch)
            guage_long_fib1 = list(guage_long_fib)
            guage_long_mask1 = list(guage_long_mask)
            guage_short_fib1 = list(guage_short_fib)
            guage_short_mask1 = list(guage_short_mask)
            long_final_mask1 = list(bias_slope * np.round(pixel_size * guage_long_mask,2) + bias_intercept)  
            short_final_mask1 = list(bias_slope * np.round(pixel_size * guage_short_mask,2) + bias_intercept)
            long_final_fib1 = list(np.round(pixel_size * guage_long_fib,2))
            short_final_fib1 = list(np.round(pixel_size * guage_short_fib,2))
            
            sort_update_fib_final = Fib_insert_information(Cm_sort_fib,long_final_fib,short_final_fib)
            print("\nLen contour fib alignment:",len(contours_fib_alignment))
            
            delta_long_p = np.round(delta_long*100 / (np.reshape(long_final_mask,(1,len(long_final_mask)))),2)
            delta_short_p = np.round(delta_short*100 / (np.reshape(short_final_mask,(1,len(short_final_mask)))),2)
            CD_Mask_long = long_final_mask
            CD_Mask_short = short_final_mask
            CD_EXP_long = long_final_fib
            CD_EXP_short = short_final_fib
            if etch == True:
                CD_EXP_long_etch = long_final_fib_etch
                CD_EXP_short_etch = short_final_fib_etch
#%%
            '''
            # 轉成 numpy array
            data_array = np.array(guage_list_sort_fib)
            
            # 存成 txt 檔案，用空格分隔
            np.savetxt(name_txt_guage, data_array, fmt="%d", delimiter=" ")
            '''
            M_dis, E_dis, PR_counts = Analysis_Size_Distriburion(CD_Mask_long,CD_EXP_long,100,np.max(CD_EXP_long),20)
            Eff_area = pitch * pitch * len(CD_EXP_long)
            tmp = []
            for a in range(0,len(CD_EXP_long)):
                tmp.append(CD_EXP_long[a]**2)
                exp_pattern_area = np.sum(tmp)
            #print(len(CD_EXP_long))
            print("\nMin CD:",np.min(CD_EXP_long))
            print("\nMax CD:",np.max(CD_EXP_long))
            print("\nEff area:",Eff_area)
            print("Layout area:",np.round(exp_pattern_area,1))
            print("Layout Desnity:",np.round(np.round(exp_pattern_area,1) / Eff_area,3))
    
        if ruler == True:
            input_img = Put_text(input_img, center_of_mass, long_axis, short_axis, angle_PB)
        else:
            print("\n-------------------------------------No Ruler-----------------------------------")
    '''
            grid_matrix_x, grid_matrix_y, Cm_m_x_grid, Cm_m_y_grid, Cm_f_x_grid, Cm_f_y_grid, long_matrix_target, short_matrix_target, \
            long_matrix_fib, short_matrix_fib, angle_matrix_target = Alignment_Matrix(Cm_sort_mask, Cm_sort_fib, sort_update_mask, \
                                                                                      sort_update_fib_final, grid_idx, grid_idx2, pixel_size, pitch)
            #, Cm_fib_restort_final, Cm_mask_restort_final
            #grid_matrix_x, grid_matrix_y = Alignment_Matrix(Cm_sort_mask, Cm_sort_fib, sort_update_mask, sort_update_fib_final, grid_idx, grid_idx2, pixel_size, pitch)
            #delta_CD_long_map = np.zeros((long_matrix_target.shape[0],long_matrix_target.shape[1]))
            #delta_CD_short_map = np.zeros((short_matrix_target.shape[0],short_matrix_target.shape[1]))
            #long_matrix_target = 0.957 * np.round(long_matrix_target, 2) - 19
            
            delta_CD_long_map = (long_matrix_fib - np.round(long_matrix_target,2)) / long_matrix_target
            #delta_CD_short_map = (short_matrix_fib - short_matrix_target)# / short_matrix_target
                   
            Plot_results_map("Mask8 Experimantal ΔCD / CD",grid_idx, grid_idx2, delta_CD_long_map[:][:]*100, (10,10), cm.jet,"%")
            print("\nMean Delta CD Over CD (%):",np.round(np.mean(delta_CD_long_map*100),3))
            print("\nMax Delta CD Over CD (%):",np.max(np.round(delta_CD_long_map*100,3)))
            print("\nMin Delta CD Over CD (%):",np.min(np.round(delta_CD_long_map*100,3)))
            #max_delta = np.unravel_index(np.argmax(delta_CD_long_map, axis = None),delta_CD_long_map.shape)
            #print("\nMax delta:","("+str(max_delta[1]+1)+","+str(grid_idx2-max_delta[0])+")")
            
            max_delta_index = np.argpartition(np.abs(delta_CD_long_map.flatten()),-10)[-10:]
            max_indices = np.unravel_index(max_delta_index, delta_CD_long_map.shape)
            
            print("\nMax top 10 indices:")
            for i in range(10):
                print("("+str(max_indices[1][i]+1)+","+str(grid_idx2-max_indices[0][i])+")")
    
            #Plot_results_map("Short Axis Delta CD Over CD",grid_idx, grid_idx2, delta_CD_short_map,(10,10), cm.jet,"nm")
    '''     
        
    
    #%% Plot image with cv2
    if show == True:
        start = time.time()
        print("\n-----------------------------------Show image-----------------------------------")
        print("\nPlot input mask...")
        if mode == "Mask":
            cv2.namedWindow('Input Mask: '+ str(mask_name),0)
            cv2.imshow('Input Mask: '+ str(mask_name),img_gray)
            cv2.waitKey(0)
        if mode == "FIB":
            cv2.namedWindow('Input Mask Ori: '+ str(fib_name),0)
            cv2.imshow('Input Mask Ori: '+ str(fib_name),img_gray)
            cv2.waitKey(0) 
            
            cv2.namedWindow('Input Mask Binary: '+ str(fib_name),0)
            cv2.imshow('Input Mask Binary: '+ str(fib_name),img_final_fib)
            cv2.waitKey(0)    
        
        if mode == "Mask+FIB":
            cv2.namedWindow('Input Mask: '+ str(mask_name),0)
            cv2.imshow('Input Mask: '+ str(mask_name),img_gray)
            cv2.waitKey(0)
            
            cv2.namedWindow('Input FIB Binary: '+ str(fib_name),0)
            cv2.imshow('Input FIB Binary: '+ str(fib_name),img_final_fib)
            cv2.waitKey(0)
        
        print("\nPlot Contour...")
        if mode == "Mask":
            cv2.namedWindow('Contour Mask',0)
            cv2.imshow('Contour Mask',img_final)
            cv2.waitKey(0)
        if mode == "FIB":
            cv2.namedWindow('Contour FIB',0)
            cv2.imshow('Contour FIB',img_final)
            cv2.waitKey(0)
        
        if mode == "Mask+FIB":
            cv2.namedWindow('Contour Mask',0)
            cv2.imshow('Contour Mask',img_final)
            cv2.waitKey(0)
        '''
            cv2.namedWindow('Contour FIB',0)
            cv2.imshow('Contour FIB',input_img3)
            cv2.waitKey(0)
        '''
        print("\nPlot Vertex...")
        if mode == "Mask+FIB":
            cv2.namedWindow('Vertex Mask',0)
            cv2.imshow('Vertex Mask',img_final_vertex)
            cv2.waitKey(0)
            '''
            cv2.namedWindow('Vertex FIB',0)
            cv2.imshow('Vertex FIB',img_final2_FIB)
            cv2.waitKey(0)  
            '''
        else:
            cv2.namedWindow('Vertex',0)
            cv2.imshow('Vertex',img_final2)
            cv2.waitKey(0)
        
        
        print("\nPlot Center of Mass...")
        if mode == "Mask+FIB":
            cv2.namedWindow('Center of Mass Mask',0)
            cv2.imshow('Center of Mass Mask',img_final3)
            cv2.waitKey(0)
            
            cv2.namedWindow('Center of Mass FIB',0)
            cv2.imshow('Center of Mass FIB',img_final3_FIB)
            cv2.waitKey(0)
        else:
            cv2.namedWindow('Center of Mass',0)
            cv2.imshow('Center of Mass',img_final3)
            cv2.waitKey(0)
           
        if mode == "Mask+FIB":
            print("\nPlot Before Alignment...")
            cv2.namedWindow('Cm Before Alignment',0)
            cv2.imshow('Cm Before Alignment',img_Cm)
            cv2.waitKey(0)
            
            print("\nPlot After Alignment...")
            cv2.namedWindow('Mask Alignment',0)
            cv2.imshow('Mask Alignment',img_mask_alignment)
            cv2.waitKey(0)
            
            cv2.namedWindow('FIB Alignment',0)
            cv2.imshow('FIB Alignment',img_fib_alignment)
            cv2.waitKey(0)
            
            print("\nPlot Mask Guage Point...")
            cv2.namedWindow('Mask Guage Point',0)
            cv2.imshow('Mask Guage Point',img_ff_mask)
            cv2.waitKey(0)
            
            print("\nPlot FIB Guage Point...")
            cv2.namedWindow('FIB Guage Point',0)
            cv2.imshow('FIB Guage Point',img_ff_fib)
            cv2.waitKey(0)
        
        if mode == "Mask":
            print("\nPlot PB to Ellispe Contour...")
            cv2.namedWindow('Target Contour',0)
            cv2.imshow('Target Contour',img_final4)
            cv2.waitKey(0)
            
            print("\nPlot PB Vector Information...")
            cv2.namedWindow('Vector',0)
            cv2.imshow('Vector',img_final5)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
           
        cv2.destroyAllWindows()
        print("\n--------------------------------Finish Processing-------------------------------")
        end = time.time()
        print("\nPlot figures costs time = ",end - start," s")
        #print("\n")
    else:
        start = time.time()
        print("\n--------------------------------Finish Processing-------------------------------")
        end = time.time()
        print("\nPlot figures costs time = ",end - start," s")
       
    if save == True:
        start = time.time()
        if mode == "Mask":
            print("\n------------------------------------Save image----------------------------------")
            #img_gray = img_gray[14:,:img_gray.shape[1]-20]
            #img_gray = np.pad(img_gray,((7,7),(10,10)))
            cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_contour.bmp', img_final)
            print("\n------------------------Successfully saved contour image!-----------------------")
            
            cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_vertex.bmp', img_final2)
            print("\n------------------------Successfully saved vertex image!------------------------")
            cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_Cm.bmp', img_final3)
            print("\n--------------------Successfully saved center of mass image!--------------------")
            cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_target_contour.bmp', img_final4)
            print("\n--------------------Successfully saved target contour image!--------------------")
            cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_vector.bmp', img_final5)
            print("\n--------------------Successfully saved extracted vector image!--------------------")
            cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_guage_points.bmp', img_ff)
            print("\n----------------------Successfully saved guage points image!----------------------")
            #print('save image')
            
        if mode == "FIB":
            print("\n------------------------------------Save image----------------------------------")
            cv2.imwrite(path_save + '/'+str(fib_name)+'_binary.bmp', img_final_fib)
            print("\n------------------------Successfully saved binary image!------------------------")
            cv2.imwrite(path_save_alignment + '/'+str(fib_name)+'_contour.bmp', img_final)
            print("\n------------------Successfully saved binary with contour image!-----------------")
            cv2.imwrite(path_save_alignment + '/'+str(fib_name)+'_vertex.bmp', img_final2)
            print("\n------------------Successfully saved binary with vertex image!------------------")
            cv2.imwrite(path_save_alignment + '/'+str(fib_name)+'_Cm.bmp', img_final3)
            print("\n------------------Successfully saved binary center of mass image!-------------------")
        if mode == "Mask+FIB":
            print("\n------------------------------------Save image----------------------------------")
            #cv2.imwrite(path_save_alignment + '/'+str(fib_name)+'_Cm_Info.bmp', img_Cm)
            img_mask_alignment = cv2.cvtColor(img_mask_alignment, cv2.COLOR_GRAY2BGR)
            #img_mask_alignment = Put_text(img_mask_alignment, center_of_mass, long_axis, short_axis, angle_PB)
            #print("\n-------------------Successfully saved center of mass image!---------------------")
            cv2.imwrite(path_save + '/'+str(mask_name[:-4])+'_alignment.bmp', img_mask_alignment[:img_mask_alignment.shape[0],:img_mask_alignment.shape[1]])#275;-160
            print("\n-----------------------------Save mask alignment image--------------------------")
            img_fib_alignment = cv2.cvtColor(img_fib_alignment, cv2.COLOR_GRAY2BGR)
            #img_fib_alignment = Put_text(img_fib_alignment, center_of_mass_FIB, long_final_fib, short_final_fib, angle_PB)
            img_fib_alignment = cv2.imwrite(path_save + '/'+'PR_'+str(fib_name[:-4])+'_alignment.bmp', img_fib_alignment[:img_fib_alignment.shape[0],:img_fib_alignment.shape[1]])#275;-160
            print("\n-----------------------------Save FIB alignment image---------------------------")
            #cv2.imwrite(path_save_alignment + '/'+str(mask_name)+'_mask_guage_point.bmp', img_ff_mask)
            print("\n-----------------------------Save mask guage point image--------------------------")
            #cv2.imwrite(path_save_alignment + '/'+str(fib_name)+'_fib_guage_point.bmp', img_ff_fib)
            print("\n-----------------------------Save fib guage point image--------------------------")
        
        end = time.time()
        print("\nSave figures costs time = ",end - start," s")
        print("\n----------------------------------Finish All Process--------------------------------")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def abs_sobelx(image, sx_thresh=(20,200)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    return sxbinary

def dir_sobel(image, thresh=(0,np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx=cv2.Sobel(gray, cv2.CV_64F,1,0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel_x = np.absolute(sobelx)
    abs_sobel_y = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    dir = np.absolute(dir)
    # 5) Create a binary mask where direction thresholds are met
    scaled_dir = np.zeros_like(dir)
    scaled_dir[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return scaled_dir


def s_channel(image,smin=90, smax=255,hmin=170, hmax=255,lmin=130, lmax=255, vmin=100, vmax=255):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    v=hsv[:,:,2]
    s_binary=np.zeros_like(s)
    h_binary = np.zeros_like(h)
    l_binary = np.zeros_like(l)
    v_binary=np.zeros_like(v)
    s_binary[(s>smin) & (s<=smax)] = 1
    h_binary[(h>hmin) & (h<=hmax)] = 1
    l_binary[(l>lmin) & (l<=lmax)] = 1
    v_binary[(v>vmin) & (v<=vmax)] = 1
    R = image[:, :, 0]
    G = image[:, :, 1]
    r_g_condition = np.zeros_like(R)
    r_g_condition [(R > 140) & (G > 140)]=1
    combined = np.zeros_like(s_binary)
    # combined[((sxbinary==1) & (dir_binary==1)) | (s_binary==1)]=1
    combined[(l_binary == 1) | (s_binary == 1)] = 1
    return h_binary , l_binary , s_binary,r_g_condition, v_binary
    #return combined


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    #src = np.array([[91.4291, 718.468], [1021.46, 718.468], [711.63, 473.525], [515.621, 473.525]], np.float32)
    #dst = np.array([[271.63, 718.468], [908.028,718.468], [906.028, 0], [271.617, 0]], np.float32)
    src = np.array([[51.1679,312.231], [476.41,312.231], [325.833,192.721], [265.863, 192.721]], np.float32)
    dst = np.array([[120.746, 312.641], [433.027,312.641], [433.027, 0], [120.746, 0]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def stack(image):
    sxbinary=abs_sobelx(image)
    dir_binary=dir_sobel(image)
    s_binary=s_channel(image)
    combined=np.zeros_like(sxbinary)
    #combined[((sxbinary==1) & (dir_binary==1)) | (s_binary==1)]=1
    combined[(l_binary == 1) &  (s_binary == 1)] = 1
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary,s_binary)) * 255
    return combined

#img=mpimg.imread('/home/arnav08/PycharmProjects/Project 4/signs_vehicles_xygrad.jpg')
img=mpimg.imread('/home/arnav08/PycharmProjects/Project 4/Frames/Original/frame1039.jpg')
h, l ,s, rg ,v =s_channel(img)
#im =s_channel(img)
#plt.imshow(im)
#plt.show()

#warped_im=warp(im)
f, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, figsize=(10, 9))
f.tight_layout()

#ax1.imshow(,cmap='gray')
#ax1.set_title('Original Image', fontsize=40)
#ax2.imshow(warped_im,cmap='gray')
#ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.show()

ax1.imshow(h,cmap='gray')
ax1.set_title('H image', fontsize=40)
ax2.imshow(l,cmap='gray')
ax2.set_title('L image', fontsize=40)
ax3.imshow(s,cmap='gray')
ax3.set_title('S image', fontsize=40)
ax4.imshow(rg,cmap='gray')
ax4.set_title('RG image', fontsize=40)
ax5.imshow(v,cmap='gray')
ax5.set_title('V image', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
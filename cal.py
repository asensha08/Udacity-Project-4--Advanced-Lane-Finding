
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def roi(image):
    mask = np.zeros_like(image)
    height=image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = np.array([[0, height - 1], [width / 2, int(0.5 * height)], [width - 1, height - 1]],
                                           dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(image, mask)
    #masked_edges = cv2.bitwise_and(image, mask)
    return thresholded

def abs_sobelx(image, xmin=20,xmax=100):
    sobelx= cv2.Sobel(image,cv2.CV_64F,1,0)
    abs_sobel= np.absolute(sobelx)
    max_sobel=np.max(abs_sobel)
    scaled_sobel= np.uint8(255*abs_sobel/max_sobel)
    binary_sobel=np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel>=xmin) | (scaled_sobel<=xmax)]=1

    return binary_sobel




def lane_pipeline(image, s_thresh=(100,255), sx_thresh=(100,255)):
     hls=cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
     gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     l_channel=hls[:,:,1]
     s_channel=hls[:,:,2]

    #Sobel in the x direction
     sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0)
     abs_sobelx=np.absolute(sobel_x)
     scaled_sobel=np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    #Threshold x gradient
     sxbinary=np.zeros_like(scaled_sobel)
     sxbinary[(scaled_sobel>sx_thresh[0]) & (scaled_sobel<=sx_thresh[1])]=1

    #Threshold color channel
     s_binary=np.zeros_like(s_channel)
     s_binary[(s_channel>s_thresh[0]) & (s_channel<=s_thresh[1])]=1

     #color_binary = np.dstack((np.zeros_like(s_binary), sxbinary, s_binary)) * 255
     combined_binary = np.zeros_like(sxbinary)
     combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

     return combined_binary

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.array([[91.4291, 718.468], [1021.46, 718.468], [711.63, 473.525], [515.621, 473.525]], np.float32)
    dst = np.array([[271.63, 718.468], [908.028,718.468], [906.028, 0], [271.617, 0]], np.float32)
    #src = np.array([[51.1679,312.231], [476.41,312.231], [325.833,192.721], [265.863, 192.721]], np.float32)
    #dst = np.array([[120.746, 312.641], [433.027,312.641], [433.027, 0], [120.746, 0]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped




#img=mpimg.imread('/home/arnav08/PycharmProjects/Project 4/signs_vehicles_xygrad.jpg')
img=mpimg.imread('/home/arnav08/PycharmProjects/Project 4/test5.jpg')
im=lane_pipeline(img)
# result=lane_pipeline(img)
#warped_im=warp(im)
plt.imshow(im,cmap='gray')
plt.show()
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()

# ax1.imshow(im,cmap='gray')
# ax1.set_title('Original Image', fontsize=40)
# ax2.imshow(warped_im,cmap='gray')
# ax2.set_title('Pipeline Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()




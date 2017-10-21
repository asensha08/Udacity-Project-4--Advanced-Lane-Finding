import cv2
# vidcap = cv2.VideoCapture("/home/arnav08/PycharmProjects/Project 4/project_video.mp4")
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#     success,image = vidcap.read()
#     cv2.imwrite("/home/arnav08/PycharmProjects/Project 4/Frames/Original/frame%d.jpg" % count, image)     # save frame as JPEG file
#     if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#         break
#     count += 1

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
image=mpimg.imread('/home/arnav08/PycharmProjects/Project 4/Frames/Original/frame611.jpg')
plt.imshow(image)
plt.show()


def abs_x_threshold(image, s_thresh=(110, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel > s_thresh[0]) & (scaled_sobel <= s_thresh[1])] = 1

    return binary

def dir_sobel(image, thresh=(0.7,1.3)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx=cv2.Sobel(gray, cv2.CV_64F,1,0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel_x = np.absolute(sobelx)
    abs_sobel_y = np.absolute(sobely)
    dirsob = np.arctan2(abs_sobel_y, abs_sobel_x)
    abs_dir = np.absolute(dirsob)
    # 5) Create a binary mask where direction thresholds are met
    binary = np.zeros_like(abs_dir)
    binary[(abs_dir>=thresh[0]) & (abs_dir<=thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary

sx_binary = abs_x_threshold(image)
dir_binary=dir_sobel(image)
combined_binary=np.zeros_like(sx_binary)
combined_binary[((sx_binary==1) | (dir_binary==1))]=1

plt.imshow(combined_binary)
plt.show()
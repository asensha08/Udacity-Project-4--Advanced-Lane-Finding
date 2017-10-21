def lane_pipeline(image):
    binary_warped=visualize(image)
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    #plt.show()

    midpoint=np.int(histogram.shape[0]/2)
    leftx_base= np.argmax(histogram[:midpoint])
    rightx_base=np.argmax(histogram[midpoint:])+midpoint

    out_img=np.dstack((binary_warped,binary_warped,binary_warped))*255

    non_zeros=binary_warped.nonzero()
    non_zeros_y=non_zeros[0]
    non_zeros_x=non_zeros[1]

    leftx_current=leftx_base
    rightx_current=rightx_base

    left_lane_inds = []
    right_lane_inds = []

    min_pix=50


    num_windows=10
    window_height= np.int(binary_warped.shape[0]/num_windows)
    window_half_width=100

    for window in range(num_windows):
        y_min=binary_warped.shape[0]- (window+1)*window_height
        y_max=binary_warped.shape[0]- (window*window_height)
        xleft_min = leftx_current - window_half_width
        xleft_max=leftx_current + window_half_width
        xright_min=rightx_current - window_half_width
        xright_max=rightx_current + window_half_width
    
        cv2.rectangle(out_img,(xleft_max,y_max),(xleft_min,y_min),(0,255,0),2)
        cv2.rectangle(out_img,(xright_max,y_max),(xright_min,y_min),(0,255,0),2)
    
        good_left_inds=((non_zeros_y>=y_min) & (non_zeros_y<=y_max) & (non_zeros_x>=xleft_min) &
                    (non_zeros_x<=xleft_max)).nonzero()[0]
        good_right_inds=((non_zeros_y>=y_min) & (non_zeros_y<=y_max) & (non_zeros_x>=xright_min) &
                    (non_zeros_x<=xright_max)).nonzero()[0]
    
    
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        if len(good_left_inds)>min_pix:
            leftx_current=np.int(np.mean(non_zeros_x[good_left_inds]))
        if len(good_right_inds)>min_pix:
            rightx_current=np.int(np.mean(non_zeros_x[good_right_inds]))
        

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = non_zeros_x[left_lane_inds]
    lefty = non_zeros_y[left_lane_inds]
    rightx = non_zeros_x[right_lane_inds]
    righty = non_zeros_y[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
# Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #plt.imshow(result)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#src = np.array([[217,684], [1038,684], [714,453], [616,453]], np.float32)
#dst = np.array([[359, image.shape[0]], [966,image.shape[0]], [966, 0], [359, 0]], np.float32)
    src = np.array([[278,677], [998,677], [739,486], [552,486]], np.float32)
    dst = np.array([[359, image.shape[0]], [966,image.shape[0]], [966, 0], [359, 0]], np.float32)
    M_inv = cv2.getPerspectiveTransform(dst,src)
# Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    
    return result

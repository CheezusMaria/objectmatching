
import numpy as np
import cv2
import matplotlib.pyplot as plt
small_image=cv2.imread('fablab1.jpg',0)
big_image=cv2.imread('fablab2.jpg',0)

    orb=cv2.ORB_create()
    an1,target1=orb.detectAndCompute(small_image,None)
    an2,target2=orb.detectAndCompute(big_image,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=bf.match(target1,target2)
    matches=sorted(matches,key= lambda x:x.distance)
    last_image=cv2.drawMatches(small_image,an1,big_image,an2,matches[:50],None,flags=2)
    plt.imshow(last_image)
    plt.show()
import numpy as np
import cv2, PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import apriltag


image = cv2.imread("IMG_6906.JPEG")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = apriltag.Detector()
detections = detector.detect(gray)
print(detections)
# fx, fy, cx, cy
K = [929.7, 932.749, 616.4179214005839, 335.4107412307333]

result = detector.detection_pose(detections[0], K , tag_size=1, z_sign=1)
T = result[0]

print(T)
R = T[0:3,0:3]
t = T[0:3,3]

print("R: ", R)
print("t: ", t)

apriltag._draw_pose(image, K, 1, T, z_sign=1)


# Amount of detected fids
NR_DETECTIONS = len(detections)
print("# detected fiducials", NR_DETECTIONS)


#print(detections[0].corners)
# Iterate over detections
for i in range(NR_DETECTIONS):

	# We want to draw four lines
	for j in range(4):
		print(j)
		start_point = tuple(detections[i].corners[j-1, :].astype(int))
		end_point = tuple(detections[i].corners[j, :].astype(int))
		cv2.line(image, start_point,
			end_point, (0, 255, 0),
			thickness=10)


imS = cv2.resize(image, (540, 540))  
cv2.imshow("frame",imS)
cv2.waitKey(0)
cv2.destroyAllWindows()
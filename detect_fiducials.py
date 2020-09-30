import numpy as np
import cv2, PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import apriltag


image = cv2.imread("IMG_6906.JPEG")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = apriltag.Detector()
result = detector.detect(gray)


print(result)



# Amount of detected fids
NR_DETECTIONS = len(result)
print("# detected fiducials", NR_DETECTIONS)


print(result[0].corners)
# Iterate over detections
for i in range(NR_DETECTIONS):

	# We want to draw four lines
	for j in range(4):
		print(j)
		start_point = tuple(result[i].corners[j-1, :].astype(int))
		end_point = tuple(result[i].corners[j, :].astype(int))
		cv2.line(image, start_point,
			end_point, (0, 255, 0),
			thickness=10)


imS = cv2.resize(image, (540, 540))  
cv2.imshow("frame",imS)
cv2.waitKey(0)
cv2.destroyAllWindows()
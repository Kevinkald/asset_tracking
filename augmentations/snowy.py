import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
import xml.etree.ElementTree as ET

folder="../images/train_snowy/"

seq = iaa.Sequential([
	iaa.FastSnowyLandscape(lightness_threshold=(150,200), lightness_multiplier=(1.0, 4.0))
])

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    # Accessing each jpg image
    if img is not None:
        # Create image augmentation

        filename_split = filename.split(".jpg")
        xml_filename_old = filename_split[0] + ".xml"

        new_filename = filename_split[0] + "_cutout.jpg"
        xml_filename = filename_split[0] + "_cutout.xml"

        # copy and edit old xml file
        tree = ET.parse(folder + xml_filename_old)
        root = tree.getroot()
        k = root.find('filename')
        k.text = new_filename
        l = root.find('path')
        l.text = new_filename


        # augment image
        image_aug = seq(image=img)
        cv2.imshow("image", image_aug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

   

    	# save augmented image + edited xml file
    	#cv2.imwrite(new_filename, image_aug)
    	#tree.write(xml_filename)
    	




import os


rainy_occluded_count = 0
sunny_occluded_count = 0

occluded_images = os.fsencode('../images/test/occlusion/')
rainy_images = os.fsencode('../images/test/rainy/')
sunny_images = os.fsencode('../images/test/sunny/')

for file in os.listdir(occluded_images):
	filename = os.fsdecode(file)
	if filename.endswith(".jpg"):
		# rainy
		for file_rainy in os.listdir(rainy_images):
			filename_rainy = os.fsdecode(file_rainy)
			if filename_rainy.endswith(".jpg"):
				if (filename_rainy == filename):
					rainy_occluded_count = rainy_occluded_count + 1
		# sunny
		for file_sunny in os.listdir(sunny_images):
			filename_sunny = os.fsdecode(file_sunny)
			if filename_sunny.endswith(".jpg"):
				if (filename_sunny == filename):
					sunny_occluded_count = sunny_occluded_count + 1
					print(filename)


print("rainy_occluded_count: ",rainy_occluded_count)
print("sunny_occluded_count: ",sunny_occluded_count)
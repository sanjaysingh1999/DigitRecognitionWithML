import glob
import cv2,csv

label="3"

# header=["label"]

# for i in range(784):
# 	header.append("pixel_"+str(i))

file = open("csv/dataset.csv","a")
writer = csv.writer(file)

# writer.writerow(header)

dir_List = glob.glob("orig_images/"+label+"/*.png")

for img_path in dir_List:
	img = cv2.imread(img_path,0)
	img= cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
	# cv2.imshow("window show",img)

	# cv2.waitKey()

data = []
data.append(label)

rows,columns = img.shape

for i in range(rows):
	for j in range(columns):
		value = img[i][j]

		if value>100:
			value=1
		else:
			value=0

		data.append(value)	

writer.writerow(data)

	
import cv2
import os

image_folder = 'D:/2019.01.19'
#image_folder = 'D:/test'
video_name = 'D:/test/video_second_set.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".tiff")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 50, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
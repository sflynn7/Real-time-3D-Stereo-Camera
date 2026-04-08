import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

pipe = pipeline(task="depth-estimation", model="xingyang1/Distill-Any-Depth-Small-hf")

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

img_counter = 0

def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    image = Image.fromarray(frame)
    depth = pipe(image)["depth"]
    cv2.imshow("test", np.hstack([np.array(image), to_rgb(np.array(depth))]))

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

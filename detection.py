from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")
img = cv2.imread("man_with_PPE.jpg")  # Your training image
results = model(img)

# Draw results
annotated = results[0].plot()
cv2.imshow("Detected", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

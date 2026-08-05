import cv2

video = cv2.VideoCapture(0)
ret, frame = video.read()

if ret:
    cv2.imshow("Test Capture", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Webcam not detected")

video.release()
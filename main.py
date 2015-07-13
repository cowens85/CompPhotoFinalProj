import numpy as np
import scipy as sp
import scipy.signal
import cv2

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)

def drawAroundBanana():
    imagePath = "images/source/banana.jpg"
    cascPath = "cascade/banana_classifier.xml"

    cascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Detect object in the image
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {0} objects!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("objects found", image)
    cv2.waitKey(0)
    cv2.imwrite("images/output/found_banana_1.jpg", image)
drawAroundBanana()
exit(1)
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
    cap = cv2.VideoCapture("video/banana_still_2.m4v")
    fps = 60
    #capSize = gray.shape # this is the size of my source video
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('video/output_banana_through_still_2.mov',fourcc,fps,size,False)
    background = None
    prevObj = None
    objDict = {}
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        if background is None:
            print "Set background"
            background = frame

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascPath = "cascade/banana_classifier.xml"

        cascade = cv2.CascadeClassifier(cascPath)
        # Detect object in the image
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=2,
            minSize=(200, 200),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print "Found {0} objects!".format(len(faces))

        newFrame = background.copy()
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y-150), (x+w, y+h+50), (0, 255, 0), 2)
            newFrame[y-150:y+h+50, x:x+w] = frame[y-150:y+h+50, x:x+w]

        # Display the resulting frame
        cv2.imshow('frame',newFrame)
        if(len(faces) > 0):
            vout.write(newFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    print background.shape
    cap.release()
    vout.release()
    cv2.destroyAllWindows()


drawAroundBanana()
exit(1)
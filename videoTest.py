import numpy as np
import scipy as sp
import scipy.signal
import stitcher
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
    cap = cv2.VideoCapture("video/banan.m4v")
    fps = 60
    #capSize = gray.shape # this is the size of my source video
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('video/output_banana_through_still_2.mov',fourcc,fps,size,False)
    background = None
    prevObj = None
    objDict = {}
    leftFrames = []
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
            scaleFactor=2.6,
            minNeighbors=1,
            minSize=(200, 200),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print "Found {0} objects!".format(len(faces))

        newFrame = background.copy()

       # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x-30, y-150), (x+60+w, y+h+50), (0, 255, 0), 2)
            newFrame[y-150:y+h+50, x:x+w] = frame[y-150:y+h+50, x:x+w]
            break




        # Display the resulting frame
        cv2.imshow('frame', newFrame)
        if len(faces) > 0:
            vout.write(newFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # When everything done, release the capture
    print background.shape
    cap.release()
    vout.release()
    cv2.destroyAllWindows()



def countBananaFrames():
    highestValue = (0,0,0)
    scale = 1.1
    minNeighbor = 0
    while(scale < 3.0):
        for minNeighbor in range(0, 5):
            cap = cv2.VideoCapture("video/banan.m4v")
            counter = 0
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if frame is None:
                    break

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascPath = "cascade/banana_classifier.xml"

                cascade = cv2.CascadeClassifier(cascPath)
                # Detect object in the image

                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=minNeighbor,
                    minSize=(200, 200),
                    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                )
                if len(faces) == 1:
                    counter+=1

            if(highestValue[0] < counter):
                highestValue = (counter, scale, minNeighbor)
            print "Did: " + str(scale) + " " + str(minNeighbor) + " Result: " + str(counter)

            cap.release()
            cv2.destroyAllWindows()
        scale += .1
        print highestValue


def findAverageCenter():
    highestValue = (0,0,0)
    scale = 2.6
    minNeighbor = 1

    cap = cv2.VideoCapture("video/banan.m4v")
    counter = 0
    x = 0
    y = 0
    w = 0
    h = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascPath = "cascade/banana_classifier.xml"

        cascade = cv2.CascadeClassifier(cascPath)
        # Detect object in the image

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=minNeighbor,
            minSize=(200, 200),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if len(faces) == 1:
            counter+=1
            x += faces[0][0]
            y += faces[0][1]
            w += faces[0][2]
            h += faces[0][3]


    print str((x/counter, y/counter, w/counter, h/counter))


    cap.release()
    cv2.destroyAllWindows()

def drawAroundCenterPoint():
    cap = cv2.VideoCapture("video/banan.m4v")
    fps = 60
    #capSize = gray.shape # this is the size of my source video
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('video/output_banana_through_still_2.mov',fourcc,fps,size,False)
    background = None
    prevObj = None
    objDict = {}
    leftFrames = []

    rightFrames =[]
    counter = 0
    while(True):
        counter += 1
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
            scaleFactor=2.6,
            minNeighbors=1,
            minSize=(200, 200),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print "Found {0} objects!".format(len(faces))

        newFrame = background.copy()

       # Draw a rectangle around the faces
        x = 251
        y = 378
        w = 541
        h = 270
        #cv2.rectangle(frame, (x-20, y-20), (x+20+w, y+20+h), (0, 255, 0), 2)

        leftFrame = (frame[0:frame.shape[0], 0:x-20])
        rightFrame = (frame[0:frame.shape[0], x+20+w:frame.shape[1]])

        leftFrames.append(leftFrames)
        rightFrames.append(rightFrame)
        # Display the resulting frame
        cv2.imshow('frame', leftFrame)
        frameCounter ="000"
        if counter < 10:
            frameCounter = "00"+str(counter)
        elif counter < 100:
            frameCounter = "0" + str(counter)

        cv2.imwrite("images/temp/leftFrames/left_" +frameCounter+ ".png", leftFrame)
        #if len(faces) > 0:
         #   vout.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # When everything done, release the capture
    print background.shape
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

#drawAroundCenterPoint()
stitcher.AlignImagesRansac("images/temp/leftFrames", "images/temp/leftFrames/left_000.png", "images/temp/leftOutput/")
exit(1)
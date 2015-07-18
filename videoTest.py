import numpy as np
import scipy as sp
import scipy.signal
import shutil
import os
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

def drawAroundBanana(videoPath, classifierPath, scale, minNeighbor):
    cap = cv2.VideoCapture(videoPath)
    fps = 60
    #capSize = gray.shape # this is the size of my source video
    size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('video/output_banana_through_still_2.mov',fourcc,fps,size,False)
    background = None
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


        cascade = cv2.CascadeClassifier(classifierPath)
        # Detect object in the image

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=minNeighbor,
            minSize=(50, 50),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print "Found {0} objects!".format(len(faces))

        newFrame = background.copy()

       # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            newFrame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]





        # Display the resulting frame
        cv2.imshow('frame', frame)
        if len(faces) > 0:
            vout.write(newFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # When everything done, release the capture
    print background.shape
    cap.release()
    vout.release()
    cv2.destroyAllWindows()



def countObjectFrames(videoPath, classifierPath, maxScale, maxNeighbors):
    highestValue = (0,0,0)
    scale = 1.1
    while(scale <= maxScale):
        for minNeighbor in range(1, maxNeighbors+1):
            cap = cv2.VideoCapture(videoPath)
            counter = 0
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if frame is None:
                    break

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade = cv2.CascadeClassifier(classifierPath)
                # Detect object in the image

                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=minNeighbor,
                    minSize=(50, 50),
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
    return (highestValue[1], highestValue[2])


def findAverageCenter(videoPath,classifierPath, scale, minNeighbor):
    highestValue = (0,0,0)

    cap = cv2.VideoCapture(videoPath)
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

        cascade = cv2.CascadeClassifier(classifierPath)
        # Detect object in the image

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=minNeighbor,
            minSize=(50, 50),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if len(faces) == 1:
            counter+=1
            x += faces[0][0]
            y += faces[0][1]
            w += faces[0][2]
            h += faces[0][3]


    data = (x/counter, y/counter, w/counter, h/counter)
    print str(data)

    cap.release()
    cv2.destroyAllWindows()

    return data


def drawAroundCenterPoint(videoPath, x, y, w, h):
    cap = cv2.VideoCapture(videoPath)
    vout = cv2.VideoWriter()
    background = None
    leftFrames = []
    rightFrames =[]
    expansionDelta = 100
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


       # Draw a rectangle around the faces
        startPoint = (x-expansionDelta, y-expansionDelta)
        endPoint = (min(frame.shape[1],x+expansionDelta+w), min(frame.shape[0],y+expansionDelta+h))
        #cv2.rectangle(frame, startPoint, endPoint, (0, 255, 0), 2)

        # determine the max size of the 'cut frames' - this is the maximum rectangle we can make left and
        # right of the banana.  This way we get the same size image to create the panorama.
        maxWidth = min(startPoint[0], frame.shape[1] - endPoint[0])


        leftFrame = (frame[0:frame.shape[0], 0:maxWidth])
        rightFrame = (frame[0:frame.shape[0], frame.shape[1] - maxWidth:frame.shape[1]])

        leftFrames.append(leftFrames)
        rightFrames.append(rightFrame)
        # Display the resulting frame
        cv2.imshow('frame', leftFrame)
        cv2.imshow('frame2', rightFrame)
        frameCounter = "000"
        if counter < 10:
            frameCounter = "00"+str(counter)
        elif counter < 100:
            frameCounter = "0" + str(counter)
        elif counter >= 100:
            frameCounter = str(counter)

        if(counter % 5 == 0):
            cv2.imwrite("images/temp/allFrames/left_" +frameCounter+ ".png", frame)
            #cv2.imwrite("images/temp/allFrames/right_" + frameCounter + ".png", rightFrame)
        #if len(faces) > 0:
         #   vout.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # When everything done, release the capture
    print background.shape
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

def drawAroundObjectWithExpansion(videoPath, classifierPath, scale, minNeighbor, xExpansion, yExpansion):
    cap = cv2.VideoCapture(videoPath)
    background = None
    counter = 0
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


        cascade = cv2.CascadeClassifier(classifierPath)
        # Detect object in the image

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=minNeighbor,
            minSize=(50, 50),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print "Found {0} objects!".format(len(faces))

        newFrame = background.copy()

        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]

        startPoint = (x-xExpansion, y-yExpansion)
        endPoint = (x+xExpansion+w, y+yExpansion+h)

         # determine the max size of the 'cut frames' - this is the maximum rectangle we can make left and
        # right of the banana.  This way we get the same size image to create the panorama.
        maxWidth = min(startPoint[0], frame.shape[1] - endPoint[0])


        leftFrame = (frame[0:frame.shape[0], 0:maxWidth])
        rightFrame = (frame[0:frame.shape[0], frame.shape[1] - maxWidth:frame.shape[1]])

        # # Display the resulting frame
        # cv2.imshow('frame', leftFrame)
        # cv2.imshow('frame2', rightFrame)
        frameCounter = "000"
        if counter < 10:
            frameCounter = "00"+str(counter)
        elif counter < 100:
            frameCounter = "0" + str(counter)
        elif counter >= 100:
            frameCounter = str(counter)

        if(counter % 5 == 0):
            cv2.imwrite("images/temp/allFrames/left_" +frameCounter+ ".png", leftFrame)
            cv2.imwrite("images/temp/allFrames/right_" + frameCounter + ".png", rightFrame)

        frame = frame[0:frame.shape[0], 0:frame.shape[1] - maxWidth]
        #frame = frame[0:frame.shape[0], maxWidth:frame.shape[1] - maxWidth]
        cv2.imshow("frame", frame)
        cv2.imwrite("images/temp/testPanoMerge/full_" + frameCounter + ".png", frame)
        #if len(faces) > 0:
         #   vout.write(frame)
        counter += 1

    # When everything done, release the capture
    print background.shape
    cap.release()
    cv2.destroyAllWindows()


def makeVideo(imageDir, videoOutDir):
    fps = 60
    #capSize = gray.shape # this is the size of my source video
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    background = None
    dirList = os.listdir(imageDir)
    size = None
    for imagePath in dirList:
        print imagePath
        frame = cv2.imread(imageDir +"/" + imagePath)
        frame = frame[0:850, 0:2000]
        print frame.shape
        if size is None:
            size = (2000,850)
            success = vout.open(videoOutDir,fourcc,fps,size,False)

        vout.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    vout.release()
    cv2.destroyAllWindows()


videoPath = "video/cousin_kitchen.m4v"
classifierPath = "cascade/haar.xml"
#result = countObjectFrames(videoPath, classifierPath, 2.0, 5)
#rect = findAverageCenter(videoPath, classifierPath, result[0], result[1])
#drawAroundBanana(videoPath, result[0], result[1])
#drawAroundBanana(videoPath,classifierPath, 1.7, 5)
#drawAroundObjectWithExpansion(videoPath,classifierPath, 1.7, 5, 80, 150)

#drawAroundCenterPoint(videoPath, rect[0], rect[1], rect[2], rect[3])
#drawAroundCenterPoint(videoPath, 504, 467, 447, 223)
# for i in range(0, 224):
#     frameCounter = "000"
#     if i < 10:
#         frameCounter = "00"+str(i)
#     elif i < 100:
#         frameCounter = "0" + str(i)
#     elif i >= 100:
#         frameCounter = str(i)
#     #move in the new file
#     shutil.copyfile("images/temp/testPanoMerge/full_" + frameCounter + ".png", "images/temp/test1/full_" + frameCounter + ".png")
#     stitcher.AlignImagesRansac("images/temp/test1", "images/temp/test1/89.JPG", "images/temp/testPanoMergeOutput/",None, True)
#     try:
#         os.rename("images/temp/testPanoMergeOutput/0.JPG", "images/temp/output/" + frameCounter + ".JPG")
#     except:
#         print 'errar'
#     os.remove("images/temp/test1/full_" + frameCounter + ".png")

makeVideo("images/temp/output/", "video/outputMovie.mov")

exit(1)




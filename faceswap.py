#!/usr/bin/python3.5.3

import dlib
import cv2
import numpy as np
import sys

#dlib face detector and predictor initialisation
trainedModel_Path = "/home/vanv/faceswap/trainedModel/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(trainedModel_Path)


# take a bounding predicted by dlib and convert it
# to the format (x, y, w, h)
def rectToBoundingBox(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	return (x, y, w, h)


#convert a shape (value returned from dlib's detector) #into a numpy array for easier use
def getNpArrFromShape(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
 
	# looping over the part of the shape to create the list of coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the numpy array
	return coords


#get landmarks of a face from an image file
def getLandmarksFromImg(imgFilename):
	#reading the image
	img = cv2.imread(imgFilename, cv2.IMREAD_COLOR)

	#getting the shape from dlib's detector
	lm = detector(img, 1)

	lmPoints = []

	#loop over the shape
	for (i, mark) in enumerate(lm):
		# Predict with the pre-trained dlib's predictor
		#and convert into list of coordinates
		shape = predictor(img, mark)
		shape = getNpArrFromShape(shape)
	 
		# TEMPORARY ::::
		# loop over the coordinates (landmarks)
		# and draw them on the image 
		for (x, y) in shape:
			cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
			lmPoints.append((x,y))


	return img, lmPoints


def getDelaunayTriangles(img, faceLandmarks):

	#using Subdiv2D and delaunays triangulate to triangulate the face
	size = img.shape
	rect = (0,0, size[1], size[0])
	subdiv = cv2.Subdiv2D(rect)
	
	for landmark in faceLandmarks :
		subdiv.insert(landmark)

	# We are getting a list of points of delaunays' triangles
	# eg : aTriangle = ( (x1,y1), (x2,y2), (x3,y3) )
	triangleList = subdiv.getTriangleList()
	
	# We are recreating a list of delaunays' triangles but this time  
	# we want the indexes of landmarks and not coordinates of points
	delaunayTri = []
	for tri in triangleList:
		pt = []

		pt.append((tri[0], tri[1]))
		pt.append((tri[2], tri[3]))
		pt.append((tri[4], tri[5]))

		#searching for corespondance between landmarks idx and delaunays triangles coordinates
		triIdxs = []
		for j in range(0, 3):
			for k in range(0, len(faceLandmarks)):
				if (abs(pt[j][0] - faceLandmarks[k][0]) < 1.0 and abs(pt[j][1] - faceLandmarks[k][1]) < 1.0):
					triIdxs.append(k)
		if len(triIdxs) == 3:
			delaunayTri.append((triIdxs[0], triIdxs[1], triIdxs[2]))
	
	#Drawing delaunay triangles (TEMPORARY)
	for t in delaunayTri :
		cv2.line(img1, faceLandmarks[t[0]], faceLandmarks[t[1]], (255,255,255), 1)
		cv2.line(img1, faceLandmarks[t[1]], faceLandmarks[t[2]], (255,255,255), 1)
		cv2.line(img1, faceLandmarks[t[2]], faceLandmarks[t[0]], (255,255,255), 1)

		
	return delaunayTri


#From two triangles (one of each images) calculate the affine transform
#and output the target image with the source triangle
def applyAffineTransform(srcImg, srcTri, targTri, sizeTargRect) :
	#get the affine transform from the two triangles
	mat = cv2.getAffineTransform(np.float32(srcTri), np.float32(targTri))
	
	#apply the affine transform to the source image
	targ = cv2.warpAffine( srcImg, mat, (sizeTargRect[0], sizeTargRect[1]), borderMode=cv2.BORDER_REFLECT_101 )

	return targ


#Warp and blend a triangular region of the source face into the target face
def warpTriangle(img1, img2, tri1, tri2):

	
	#Get the bounding rectangle of the triangles
	# note :: rect[0] : x, rect[1] : y, rect[2] : w, rect[3] : h
	rect1 = cv2.boundingRect(np.float32([tri1])) 
	rect2 = cv2.boundingRect(np.float32([tri2])) 

	#Get coordinates of the triangles corresponding to the bounding rectangle
	tri1Rect = []
	tri2Rect = []
	
	for i in range(0,3):
		tri1Rect.append( ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])) ) 
		tri2Rect.append( ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])) ) 
		
	#Creating the mask in the shape of the triangle (in B&W)
	mask = np.zeros( (rect2[3], rect2[2], 3), dtype = np.float32 )
	cv2.fillConvexPoly(mask, np.int32(tri2Rect), (1.0,1.0,1.0))
	
	#get the rectangle part of the image
	img1Rect = img1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]

	sizeTargRect = (rect2[2], rect2[3])

	#apply the affine transform
	img2Rect = applyAffineTransform(img1Rect, tri1Rect, tri2Rect, sizeTargRect)

	#And applying the mask to the bounding rectangle image
	img2Rect = img2Rect * mask

	#Copy the actual region of the face we want (triangle) to the output image
	img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * ( (1.0, 1.0, 1.0) - mask )

	img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] + img2Rect 



#### MAIN ########

#Get faces landmarks
img1, lmPoints1 = getLandmarksFromImg(sys.argv[1])
img2, lmPoints2 = getLandmarksFromImg(sys.argv[2])
img1Warped = np.copy(img2)

#since we will be only swapping the face we only triangulate the face contour landmarks
#(got the face contour landmarks indixes by displaying them on the img)
facePoints1 = [lmPoints1[i] for i in range(0,27)]
delaunayTri1 = getDelaunayTriangles(img1, lmPoints1)

facePoints2 = [lmPoints2[i] for i in range(0,27)]
delaunayTri2 = getDelaunayTriangles(img2, facePoints2)



for i in range(0, len(delaunayTri2)):
	tri1 = [ facePoints1[delaunayTri1[i][0]], facePoints1[delaunayTri1[i][1]], facePoints1[delaunayTri1[i][2]] ]
	tri2 = [ facePoints2[delaunayTri2[i][0]], facePoints2[delaunayTri2[i][1]], facePoints2[delaunayTri2[i][2]] ]

	warpTriangle(img1, img1Warped, tri1, tri2)



# outup the image+landmarks into a file
cv2.imwrite("Output.jpg", img1)



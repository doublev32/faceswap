#!/usr/bin/python3.5.3

from PIL import Image
import dlib
import cv2
import sys
import numpy as np
import math
import scipy.spatial as spatial

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


	return lmPoints


def GetBilinearPixel(imArr, posX, posY, out):

	#Get integer and fractional parts of numbers
	modXi = int(posX)
	modYi = int(posY)
	modXf = posX - modXi
	modYf = posY - modYi

	#Get pixels in four corners
	for chan in range(imArr.shape[2]):
		bl = imArr[modYi, modXi, chan]
		br = imArr[modYi, modXi+1, chan]
		tl = imArr[modYi+1, modXi, chan]
		tr = imArr[modYi+1, modXi+1, chan]
	
		#Calculate interpolation
		b = modXf * br + (1. - modXf) * bl
		t = modXf * tr + (1. - modXf) * tl
		pxf = modYf * t + (1. - modYf) * b
		out[chan] = int(pxf+0.5) #Do fast rounding to integer

	return None #Helps with profiling view

def WarpProcessing(inIm, inArr, 
		outArr, 
		inTriangle, 
		triAffines, shape):

	#Ensure images are 3D arrays
	px = np.empty((inArr.shape[2],), dtype=np.int32)
	homogCoord = np.ones((3,), dtype=np.float32)

	#Calculate ROI in target image
	xmin = shape[:,0].min()
	xmax = shape[:,0].max()
	ymin = shape[:,1].min()
	ymax = shape[:,1].max()
	xmini = int(xmin)
	xmaxi = int(xmax+1.)
	ymini = int(ymin)
	ymaxi = int(ymax+1.)
	#print xmin, xmax, ymin, ymax

	#Synthesis shape norm image		
	for i in range(xmini, xmaxi):
		for j in range(ymini, ymaxi):
			homogCoord[0] = i
			homogCoord[1] = j

			#Determine which tesselation triangle contains each pixel in the shape norm image
			if i < 0 or i >= outArr.shape[1]: continue
			if j < 0 or j >= outArr.shape[0]: continue

			#Determine which triangle the destination pixel occupies
			tri = inTriangle[i,j]
			if tri == -1: 
				continue
				
			#Calculate position in the input image
			affine = triAffines[tri]
			outImgCoord = np.dot(affine, homogCoord)

			#Check destination pixel is within the image
			if outImgCoord[0] < 0 or outImgCoord[0] >= inArr.shape[1]:
				for chan in range(px.shape[0]): outArr[j,i,chan] = 0
				continue
			if outImgCoord[1] < 0 or outImgCoord[1] >= inArr.shape[0]:
				for chan in range(px.shape[0]): outArr[j,i,chan] = 0
				continue

			#Nearest neighbour
			#outImgL[i,j] = inImgL[int(round(inImgCoord[0])),int(round(inImgCoord[1]))]

			#Copy pixel from source to destination by bilinear sampling
			#print i,j,outImgCoord[0:2],im.size
			GetBilinearPixel(inArr, outImgCoord[0], outImgCoord[1], px)
			for chan in range(px.shape[0]):
				outArr[j,i,chan] = px[chan]
			#print outImgL[i,j]

	return None

def PiecewiseAffineTransform(srcIm, srcPoints, dstIm, dstPoints):

	#Convert input to correct types
	srcArr = np.asarray(srcIm, dtype=np.float32)
	dstPoints = np.array(dstPoints)
	srcPoints = np.array(srcPoints)

	#Split input shape into mesh
	tess = spatial.Delaunay(dstPoints)

	#Calculate ROI in target image
	xmin, xmax = dstPoints[:,0].min(), dstPoints[:,0].max()
	ymin, ymax = dstPoints[:,1].min(), dstPoints[:,1].max()
	#print xmin, xmax, ymin, ymax

	#Determine which tesselation triangle contains each pixel in the shape norm image
	inTessTriangle = np.ones(dstIm.size, dtype=np.int) * -1
	for i in range(int(xmin), int(xmax+1.)):
		for j in range(int(ymin), int(ymax+1.)):
			if i < 0 or i >= inTessTriangle.shape[0]: continue
			if j < 0 or j >= inTessTriangle.shape[1]: continue
			normSpaceCoord = (float(i),float(j))
			simp = tess.find_simplex([normSpaceCoord])
			inTessTriangle[i,j] = simp

	#Find affine mapping from input positions to mean shape
	triAffines = []
	for i, tri in enumerate(tess.vertices):
		meanVertPos = np.hstack((srcPoints[tri], np.ones((3,1)))).transpose()
		shapeVertPos = np.hstack((dstPoints[tri,:], np.ones((3,1)))).transpose()

		affine = np.dot(meanVertPos, np.linalg.inv(shapeVertPos)) 
		triAffines.append(affine)

	#Prepare arrays, check they are 3D	
	targetArr = np.copy(np.asarray(dstIm, dtype=np.uint8))
	srcArr = srcArr.reshape(srcArr.shape[0], srcArr.shape[1], len(srcIm.mode))
	targetArr = targetArr.reshape(targetArr.shape[0], targetArr.shape[1], len(dstIm.mode))

	#Calculate pixel colours
	WarpProcessing(srcIm, srcArr, targetArr, inTessTriangle, triAffines, dstPoints)
	
	#Convert single channel images to 2D
	if targetArr.shape[2] == 1:
		targetArr = targetArr.reshape((targetArr.shape[0],targetArr.shape[1]))
	dstIm.paste(Image.fromarray(targetArr))

if __name__ == "__main__":
	#Load images
	srcIm = Image.open(sys.argv[1])
	dstIm = Image.open(sys.argv[2])
	
	#Get faces landmarks
	lmPoints1 = getLandmarksFromImg(sys.argv[1])
	lmPoints2 = getLandmarksFromImg(sys.argv[2])

	#Perform transform
	PiecewiseAffineTransform(srcIm, lmPoints1, dstIm, lmPoints2)

	#Save and visualize result
	dstIm.save("Output.jpg")
	dstIm.show()




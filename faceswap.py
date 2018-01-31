#!/usr/bin/python3.5.3

from PIL import Image
import dlib
import cv2
import sys
import numpy as np
import math
import os
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
def getLandmarksFromImg(img):

	#getting the shape from dlib's detector
	lm = detector(img, 1)

	lmPoints = []

	#loop over the shape
	for (i, mark) in enumerate(lm):
		# Predict with the pre-trained dlib's predictor
		#and convert into list of coordinates
		shape = predictor(img, mark)
		shape = getNpArrFromShape(shape)

		# loop over the coordinates (landmarks)
		# and add them to the result list
		for (x, y) in shape:
			#cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
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

def WarpProcessing(inIm, inArr,  outArr,  inTriangle,  triAffines, shape):

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
	return Image.fromarray(targetArr)


def getDelaunayTriangles(rect, faceLandmarks):

	#using Subdiv2D and delaunays triangulate to triangulate the face
	subdiv = cv2.Subdiv2D(rect)
	
	for landmark in faceLandmarks :
		subdiv.insert(landmark)

	# We are getting a list of points of delaunays' triangles
	# eg : aTriangle = ( (x1,y1), (x2,y2), (x3,y3) )
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
	# note :: r[0] : x, r[1] : y, r[2] : w, r[3] : h
	r1 = cv2.boundingRect(np.float32([tri1])) 
	r2 = cv2.boundingRect(np.float32([tri2])) 

	#Get coordinates of the triangles corresponding to the bounding rectangle
	tri1Rect = []
	tri2Rect = []
	
	for i in range(0,3):
		tri1Rect.append( ((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])) ) 
		tri2Rect.append( ((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])) ) 
		
	#Creating the mask in the shape of the triangle (in B&W)
	mask = np.zeros( (r2[3], r2[2], 3), dtype = np.float32 )
	cv2.fillConvexPoly(mask, np.int32(tri2Rect), (1.0,1.0,1.0))
	
	#get the rectangle part of the image
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

	sizeTargRect = (r2[2], r2[3])

	#apply the affine transform
	img2Rect = applyAffineTransform(img1Rect, tri1Rect, tri2Rect, sizeTargRect)

	#And applying the mask to the bounding rectangle image
	img2Rect = img2Rect * mask

	#Copy the actual region of the face we want (triangle) to the output image
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect



def createMaskFromImg(lmPoints, img):
	#Reduce mask by around 3 pixels (Erosion Opencv)
	
	# Calculate Mask
	hull = []

	hullIndex = cv2.convexHull(np.array(lmPoints), returnPoints = False)
		  
	for i in range(0, len(hullIndex)):
		hull.append(lmPoints[int(hullIndex[i])])

	hull8U = []
	for i in range(0, len(hull)):
		hull8U.append((hull[i][0], hull[i][1]))

	mask = np.zeros(img.shape, dtype = img.dtype)  

	cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

	return hull, mask


def seamlessCloneImgs( swappedFacesImg, cvDstIm, lmPointsDst, imgIsCv=True ):

	#convert the after warp image in opencv image type
	if not imgIsCv:
		swappedFacesImg = np.array(swappedFacesImg)
		swappedFacesImg = swappedFacesImg[:,:,::-1].copy()
		#cv2.imwrite("swappedFacesImg.jpg", swappedFacesImg)

	hull, mask = createMaskFromImg(lmPointsDst, cvDstIm)
	kernel = np.ones((5,5),np.uint8)
	mask = cv2.erode(mask, kernel, iterations=3)
	#cv2.imwrite("mask.jpg", mask)

	#get center of face
	r = cv2.boundingRect(np.float32([hull]))    
	center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

	# Clone seamlessly.
	output = cv2.seamlessClone(np.uint8(swappedFacesImg), cvDstIm, mask, center, cv2.NORMAL_CLONE)
	#cv2.imwrite("Output.jpg", output)

	return output

	
def imageResize(image, height):
	dim = None
	(h, w) = image.shape[:2]

	# calculate the ratio of the height and construct the
	# dimensions
	r = height / float(h)
	dim = (int(w * r), height)

	# resize the image
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	# return the resized image
	return resized

def faceSwapWithId(srcImg, srcLmPoints, dstImg, dstLmPoints):
 
	srcImgWarped = np.copy(dstImg);  

	# Find convex hull
	hull1 = []
	hull2 = []

	hullIndex = cv2.convexHull(np.array(dstLmPoints), returnPoints = False)
	  
	for i in range(0, len(hullIndex)):
		hull1.append(srcLmPoints[int(hullIndex[i])])
		hull2.append(dstLmPoints[int(hullIndex[i])])
    
    
	# Find delanauy traingulation for convex hull points
	sizeDstImg = dstImg.shape
	rect = (0, 0, sizeDstImg[1], sizeDstImg[0])

	dt = getDelaunayTriangles(rect, hull2)

	if len(dt) == 0:
		quit()


	# Apply affine transformation to Delaunay triangles
	for i in range(0, len(dt)):
		t1 = []
		t2 = []

		#get points for img1, img2 corresponding to the triangles
		for j in range(0, 3):
			t1.append(hull1[dt[i][j]])
			t2.append(hull2[dt[i][j]])

		warpTriangle(srcImg, srcImgWarped, t1, t2)

	return srcImgWarped



def faceSwap(srcImgPath, dstImgPath) : 
	#Load images
	srcIm = Image.open(srcImgPath)
	dstIm = Image.open(dstImgPath)

	#reading the images (with opencv)
	cvImg1 = cv2.imread(srcImgPath, cv2.IMREAD_COLOR)
	cvImg2 = cv2.imread(dstImgPath, cv2.IMREAD_COLOR)
	
	#Get faces landmarks
	lmPoints1 = getLandmarksFromImg(cvImg1)
	lmPoints2 = getLandmarksFromImg(cvImg2)

	#Perform swapping (Identity not preserved)
	output_NoID = PiecewiseAffineTransform(srcIm, lmPoints1, dstIm, lmPoints2)
	output_NoID_seamLessCloning = seamlessCloneImgs( output_NoID, cvImg2, lmPoints2, imgIsCv=False)

	#Perform swapping (Identity preserved)
	output_ID = faceSwapWithId(cvImg1, lmPoints1, cvImg2, lmPoints2)
	newLmAfterSwap = getLandmarksFromImg(output_ID)
	output_ID_seamLessCloning = seamlessCloneImgs( output_ID, cvImg2, lmPoints2 )

	#Create a combined image with: source image + destination image + identity not preserved +identity preserved
	(h, w) = cvImg2.shape[:2]
	cvImg1 = imageResize(cvImg1, h)
	combinedImg = np.concatenate((cvImg1, cvImg2, output_NoID_seamLessCloning, output_ID_seamLessCloning), axis=1)
	cv2.imwrite("Result.jpg", combinedImg)
	
	return combinedImg, h



if __name__ == "__main__":
	
	
	folderPath = sys.argv[1]

	imgsPath = []
	for path, subdir, files in os.walk(folderPath):
		for f in files :
			imgsPath.append(os.path.join(path, f))

	imgCount = 0
	heights = []
	for f in imgsPath:
		for g in imgsPath:
			if ( f != g ) :
				print (f + " + " + g)
				img, height = faceSwap(f,g)
				heights.append(height)
				cv2.imwrite("resultImages/result"+str(imgCount)+".jpg", img)
				imgCount += 1


	htmlStr = """
<html>
	<body>
"""
	
	for i in range(0,len(heights)):
		htmlStr += """
		<img src="resultImages/result"""+str(i)+""".jpg" />
		"""

	htmlStr += """
	</body>
</html>
"""
	HtmlFile = open("resultHtml.html", "w")
	HtmlFile.write(htmlStr)
	HtmlFile.close()


import cv2
import numpy as np
import math
from objloader_simple import *

kamera = cv2.VideoCapture(1)  # indeks 1 bo irium, droidcam jest 0
wzorzec = cv2.imread('wzorzec3.jpg', 0)
klatka2 = cv2.imread('klatka3.jpg')  # do testuf
model = OBJ('illidan10D.obj', swapyz=True)
orb = cv2.ORB_create()
kp_wzorzec, des_wzorzec = orb.detectAndCompute(wzorzec, None)
# brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

drawRender = True
drawMatches = False
drawKeypoints = False
drawRectangle = False

while True:
    denton = cv2.imread('denton2.jpg')
    ok, klatka = kamera.read()
    #klatka = klatka2
    kp_klatka, des_klatka = orb.detectAndCompute(cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY), None)
    if des_klatka is not None:
        #matches = bf.match(des_klatka, des_wzorzec)
        matches = bf.match(des_wzorzec, des_klatka)  # nie dziala (?!)
        if len(matches) > 40:
            print('matches len:', len(matches))
            #matches = sorted(matches, key=lambda x: x.distance)
            punkty_klatka = np.empty(0).astype(np.int32)
            punkty_wzorzec = np.empty(0).astype(np.int32)
            #for m in matches:  # indeksy bf.match(queryIdx, trainIdx)
            #    punkty_wzorzec = np.append(punkty_wzorzec, kp_wzorzec[m.queryId].pt)
            #    punkty_klatka = np.append(punkty_klatka, kp_klatka[m.trainIdx].pt)
            #punkty_klatka = punkty_klatka.reshape(-1, 1, 2)
            #punkty_wzorzec = punkty_wzorzec.reshape(-1, 1, 2)

            srcPts = np.float32([kp_wzorzec[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dstPts = np.float32([kp_klatka[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            #homografia, x = cv2.findHomography(punkty_wzorzec, punkty_klatka, cv2.RANSAC, 5.0)
            homografia, x = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

            # prostokat wzorca
            wierzcholki = np.float32([[0, 0], [0, wzorzec.shape[0]-1],[wzorzec.shape[1]-1, wzorzec.shape[0]-1],[wzorzec.shape[1]-1, 0]]).reshape(-1,1,2)
            hT, wT = wzorzec.shape
            wierzcholki = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
            # perspektywa
            dst = cv2.perspectiveTransform(wierzcholki, homografia)
            if drawRectangle:
                klatka = cv2.polylines(klatka, [np.int32(dst)], True, (0, 255, 0), 3)


            # projekcja 3d
            camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
            rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homografia * (-1))
            col_1 = rot_and_transl[:, 0]
            col_2 = rot_and_transl[:, 1]
            col_3 = rot_and_transl[:, 2]
            l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
            rot_1 = col_1 / l
            rot_2 = col_2 / l
            translation = col_3 / l
            c = rot_1 + rot_2
            p = np.cross(rot_1, rot_2)
            d = np.cross(c, p)
            rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
            rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
            rot_3 = np.cross(rot_1, rot_2)
            projection = np.stack((rot_1, rot_2, rot_3, translation)).T
            projection = np.dot(camera_parameters, projection)

            if drawRender:
                # render modelu 3d
                vertices = model.vertices
                scale_matrix = np.eye(3) * 3
                h, w = wzorzec.shape

                for face in model.faces:
                    face_vertices = face[0]
                    points = np.array([vertices[vertex - 1] for vertex in face_vertices])
                    #points = np.dot(points, scale_matrix)
                    points = points * 3
                    points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
                    dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
                    imgpts = np.int32(dst)
                    cv2.fillConvexPoly(denton, imgpts, (0, 255, 0))
                    #cv2.fillConvexPoly(klatka, imgpts, (0, 255, 0))



            ############################################################


            img2 = denton
            if drawMatches:
                #img2 = cv2.drawMatches(wzorzec, kp_wzorzec, klatka, kp_klatka, matches, None, flags=2)
                img2 = cv2.drawMatches(klatka, kp_klatka, wzorzec, kp_wzorzec, matches, None, flags=2)

            cv2.imshow('a', img2)
            cv2.waitKey(1)
        else: #jesli nie wykrylo
            img2 = klatka
            if drawKeypoints:
                img2 = cv2.drawKeypoints(klatka, kp_klatka, None, flags=0)
            cv2.imshow('a', img2)
            cv2.waitKey(1)

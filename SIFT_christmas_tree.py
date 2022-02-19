import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

single = cv2.imread("./data/single_christmas_tree.jpeg", 0)
single = cv2.resize(single, (1920, 1080), interpolation = cv2.INTER_AREA)
single = cv2.blur(single, (5, 5)) 

# pip install opencv-contrib-python
sift = cv2.SIFT_create()

key_points1, descriptors1 = sift.detectAndCompute(single, None)

matcher = cv2.BFMatcher()

while cam.isOpened():
    ret, image = cam.read()
    # image = cv2.blur(image, (8, 8)) 
    try:
        key_points2, descriptors2 = sift.detectAndCompute(image, None)

        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        print(len(matches))
        # Ищем лучшие точки
        best = []

        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                best.append([m1])

        print(len(best))

        if len(best) > 30:
            src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best])
            src_pts = src_pts.reshape(-1, 1, 2)

            dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best])
            dst_pts = dst_pts.reshape(-1, 1, 2)

            M, mast = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = single.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print(f"Not enough matches - {len(best)}")

        image = cv2.drawMatchesKnn(single, key_points1, image, key_points2, best, None)
    except:
        print("error")

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    cv2.imshow("Camera", image)


cam.release()
cv2.destroyAllWindows()
# print(matches[0][0].distance)
# print(matches[0][0].distance - matches[0][1].distance)
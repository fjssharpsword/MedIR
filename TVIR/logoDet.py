import traceback
import cv2
import numpy as np

def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]


def detectFeatures(img, train_features):
    train_kps, train_descs, shape = train_features
    # get features from input image
    kps, descs, _ = getFeatures(img)
    # check if keypoints are extracted
    if not kps:
        return None
    # now we need to find matching keypoints in two sets of descriptors (from sample image, and from current image)
    # knnMatch uses k-nearest neighbors algorithm for that
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_descs, descs, k=2)

    good = []
    # apply ratio test to matches of each keypoint
    # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
    # otherwise, all KPs will be almost equally far
    try:
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])

        # stop if we didn't find enough matching keypoints
        if len(good) < 0.1 * len(train_kps):
            return None

        # estimate a transformation matrix which maps keypoints from train image coordinates to sample image
        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if m is not None:
            # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
            scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1),
                                                                (shape[1] - 1, shape[0] - 1),
                                                                (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
            rect = cv2.minAreaRect(scene_points)
            # check resulting rect ratio knowing we have almost square train image
            if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
                return rect
    except:
        pass
    return None

def canny_detection(gray_scale_image=None):
    """
    Run openCV Canny detection on a provided gray scale image. Return the polygons of canny contours and bounding
    rectangles.
    https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    """

    multiplier = 2
    contour_accuracy = 3
    min_threshold = 100
    max_threshold = int(min_threshold * multiplier)

    canny_output = cv2.Canny(gray_scale_image, min_threshold, max_threshold)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    bound_rect = [None] * len(contours)

    for index, contour in enumerate(contours):
        contours_poly[index] = cv2.approxPolyDP(contour, contour_accuracy, True)
        bound_rect[index] = cv2.boundingRect(contours_poly[index])

    return contours_poly, bound_rect

def grayscale_blur(image):
    """
    Convert image to gray and blur it.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.blur(image_gray, (3, 3))

    return image_gray

#https://ai-facets.org/robust-logo-detection-with-opencv/
#https://github.com/luiszugasti/IconMatch/tree/main
def main():
    # get train features
    root = '/data/pycode/MedIR/TVIR/imgs/'
    imsrc = cv2.imread(root+"big.png")
    imobj = cv2.imread(root+"small.png")

    imsrc = grayscale_blur(imsrc)
    imobj = grayscale_blur(imobj)

    src_contours, src_bound = canny_detection(imsrc)
    obj_contours, obj_bound = canny_detection(imobj)

    print(obj_bound)
    """
    train_features = getFeatures(imobj)
    # detect features on test image
    region = detectFeatures(imsrc, train_features)
    if region is not None:
        # draw rotated bounding box
        box = cv2.boxPoints(region)
        box = np.int0(box)
        cv2.drawContours(imsrc, [box], 0, (0, 255, 0), 2)
    # display the image
    cv2.imwrite(root+'cv_match.jpg', imsrc)
    """



if __name__ == "__main__":
    main()
from PIL import Image
import cv2
import sys
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import imutils
import colorsys
import matplotlib.pyplot as plt

def CaptureFocus():
    root = '/data/pycode/MedIR/TVIR/imgs/'
    image = cv2.imread(root+"focus_icon.png", 0)
    
    retval, img_global = cv2.threshold(image,130,255,cv2.THRESH_BINARY)

    img_ada_mean=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)
    img_ada_gaussian=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)

    imgs=[image,img_global,img_ada_mean,img_ada_gaussian]
    titles=['Original Image','Global Thresholding(130)','Adaptive Mean','Adaptive Guassian']

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(imgs[i],'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.savefig('/data/pycode/MedIR/TVIR/imgs/focus_rec.png', dpi=300, bbox_inches='tight')


def StripeDetector():

    root = '/data/pycode/MedIR/TVIR/imgs/'
    image = cv2.imread(root+"stripe_abnormal.png")

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H = cv2.split(img_hsv)[0]
        
    # 傅里叶变换
    f = np.fft.fft2(H)
    r, c = f.shape
    fshift = np.fft.fftshift(f)
    # f_img = 20 * np.log(np.abs(f))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    matrix_mean = np.mean(magnitude_spectrum)
    # 计算阈值
    matrix_std = np.std(magnitude_spectrum)
    # 最大值
    matrix_max = magnitude_spectrum.max()
    # 计算阈值(均值加3倍标准差 和 最大值/2 中大的值为阈值)
    T = max(matrix_mean + 3 * matrix_std, matrix_max / 2)
    # 将小于T的变为0
    # magnitude_spectrum[magnitude_spectrum < T] = 0
    # 统计大于T的点数
    magnitude_points = (magnitude_spectrum >= T)
    target_array = magnitude_spectrum[magnitude_points]
    magnitude_sum = target_array.size
    streak_rate = magnitude_sum / (c * r)
    # print("条纹率", streak_rate)
    if streak_rate > 0.0005:
        print("图片条纹") 
    else:
        print("图片正常")

def Solidcolor():
    root = '/data/pycode/MedIR/TVIR/imgs/'
    image = cv2.imread(root+"bg_red.png")
    
    #https://www.jb51.net/article/256496.htm
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #turn to HSV
    #target color range
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    ratio = mask[mask==0].shape[0]/mask.shape[0]*mask.shape[1]
    if ratio>0.90: 
        print('Red')

def LuminanceAssert():
    root = '/data/pycode/MedIR/TVIR/imgs/'
    img = cv2.imread(root+'luminance.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    a = 0
    ma = 0
    #np.full 构造一个数组，用指定值填充其元素
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i-128-da) * hist[i])
    m = abs(ma / size)
    # 亮度系数
    k = abs(da) / m
    print(k)
    if k[0] > 1:
        # 过亮
        if da > 0:
            print("过亮","da:", da)
        else:
            print("过暗","da:",da)
    else:
        print("亮度正常","da:", da)

def ShapeDetector():
    #loading images
    root = '/data/pycode/MedIR/TVIR/imgs/'
    img = cv2.imread(root+'bar2.png')

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) > 4:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, 'Circle', (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(root+"bar2_shape.png", img)

def TextRecognition():
    root = '/data/pycode/MedIR/TVIR/imgs/'
    ocr=PaddleOCR(use_angle_cls=True, lang="en")

    result=ocr.ocr(root+"ocr.png",cls=True)
    for line in result:
        print(line)
    
    image = Image.open(root+"ocr.png").convert('RGB')
    boxes = [detection[0] for line in result for detection in line] # Nested loop added
    txts = [detection[1][0] for line in result for detection in line] # Nested loop added
    scores = [detection[1][1] for line in result for detection in line] # Nested loop added
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/data/pycode/MedIR/TVIR/docs/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(root+'ocr_ret.jpg')

def SubImageSearch():

    root = '/data/pycode/MedIR/TVIR/imgs/'
    imsrc = cv2.imread(root+"big.png")
    imobj = cv2.imread(root+"small.png")

    result = cv2.matchTemplate(imsrc, imobj, cv2.TM_SQDIFF_NORMED)
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    theight, twidth = imobj.shape[:2]
    cv2.rectangle(imsrc, min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
    cv2.imwrite(root+'match.jpg', imsrc)

def main():
    #SubImageSearch()
    #TextRecognition()
    #ShapeDetector()
    #LuminanceAssert()
    #Solidcolor()
    #StripeDetector()
    CaptureFocus()

if __name__ == "__main__":
    main()
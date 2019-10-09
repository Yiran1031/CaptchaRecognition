import glob
import os
import sys
import cv2
import numpy as np

def load_image(filename,num):
    src = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(gray,(5,5),0)

    rows = image_blur.shape[0]
    circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, rows/3,
                               param1=100, param2=26,
                               minRadius=7, maxRadius=25)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)
    record = np.zeros(gray.shape)
    for i in range(0,len(gray)):
        for j in range(0,len(gray[i])):
            for k in circles[0,:]:
                distance = np.sqrt((j-k[0])**2+(i-k[1])**2)
                if distance <= k[2]and record[i][j] == 0:
                    gray[i][j] = 255-gray[i][j]
                    record[i][j] = 1
    cv2.imshow("detected circles", src)
    cv2.imshow("gray", gray)
    cv2.imshow("GaussianBlur",image_blur)
    image_median = cv2.medianBlur(gray, 5)
    cv2.imshow("medianBlur",image_median)
    cv2.waitKey(0)
    cv2.imwrite('result/'+str(num)+'_blurImage.png',image_median)
    #cv2.imwrite('result/'+str(num)+'_grayImage.png',gray)
    return 0
def main():
    IMAGE_PATH = 'set'
    filenames = glob.glob(os.path.join(IMAGE_PATH,"*.jpg"))
    print ("name of image:")
    filename = filenames[0]
    image = cv2.imread(filename,cv2.IMREAD_COLOR)
    cv2.imshow("origin image",image)
    i = 0
    while (True):
        key = cv2.waitKey(0)
        if key == ord("i"):
            if i in range(len(filenames)):
                cv2.destroyAllWindows()
                print (filenames[i])
                current = load_image(filenames[i],i)
                i = i+1
            else:
                print("all image has been read")
                exit()

        if key == ord("q"):
            exit()

if __name__ == "__main__":
    main()

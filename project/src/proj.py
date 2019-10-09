import cv2
import numpy as np
import sys
from keras.models import load_model

def load_image():
    if len(sys.argv)>1:
        image = cv2.imread(sys.argv[1])
    else:
        image = cv2.imread(DEFAULT_IMAGE)
    return image

def to_gray(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image_gray

def max_index(sumlist,i,j):
    maxIndex = i+1
    for index in range(i+1,j):
        if sumlist[index] >= sumlist[maxIndex]:
            maxIndex = index
    return maxIndex

def min_index(sumlist,i,j):
    minIndex = i+1
    for index in range(i+1,j):
        if sumlist[index] <= sumlist[minIndex]:
            minIndex = index
    return minIndex

def segmentation(bounds, image):
    segments = []
    for i, j in zip(bounds[0::2], bounds[1::2]):
        segments.append(image[:,i:j])
    return segments

def process(img):
    img = cv2.resize(img, (20,20))
    img = img.reshape(-1, 20, 20, 1)
    img = np.array(img,dtype="float")/255.0
    return img

def main():
    image = load_image()
    image_gray = to_gray(image)
    # print(image_gray.shape)

    thresh = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    thresh[thresh == 0] = 1
    thresh[thresh == 255] = 0
    img = thresh.astype(np.float)

    img_col_sum = np.sum(img,axis=0).tolist()
    img_col_sum = [int(x) for x in img_col_sum]


    p = []
    bounds = []
    WIDTH = 23
    DIFF = 3
    THRES = 2

    for index, ele in enumerate(img_col_sum):
        if ele <= THRES:
            p.append(index)
    # print("p is:")
    # print(p)

    for i in range(len(p)-1):
        left = p[i]
        right = p[i+1]
        if right-left >= WIDTH/3 and right-left <= WIDTH/2:
            bounds.append(left-5)
            bounds.append(right+5)
            # print("1")
        if right-left >= WIDTH-1 and right-left <= 2*WIDTH+1:
            bounds.append(left)
            bounds.append(right)
            # print("2")
        if right-left > 2*WIDTH+1:
            maxIndex = max_index(img_col_sum,left,right)
            minIndex = min_index(img_col_sum,left,right)
            # print("min value is: {}".format(img_col_sum[minIndex]))
            # print("minIndex is: {}".format(minIndex))
            # print("maxIndex is: {}".format(maxIndex))
            if abs(maxIndex-left-WIDTH)<DIFF:
                bounds.append(left)
                bounds.append(maxIndex)
                bounds.append(maxIndex)
                bounds.append(right)
                # print("3")
            elif abs(minIndex-left-WIDTH)<DIFF:
                bounds.append(left)
                bounds.append(minIndex)
                bounds.append(minIndex)
                bounds.append(right)
                # print("4")
            else:
                bounds.append(left)
                bounds.append(int((left+right)/2))
                bounds.append(int((left+right)/2))
                bounds.append(right)
                # print("5")

    print("\n-------\nbounds is:")
    print(bounds)
    print('------\n')

    segments = segmentation(bounds, image_gray)

    categories='123456789ABCDEFGHJKLMNPQRSTUVWXYZ'

    model = load_model('./model/captcha_model.hdf5')
    text = []
    for k in range(len(segments)):
        img = process(segments[k])
        prediction = model.predict(img)
        text.append(categories[np.argmax(prediction)])

    result = ''.join(text)
    print('\n---------\nThe captcha image is : {}'.format(result))


    cv2.imshow('image', image_gray)
    for i in range(len(segments)):
        cv2.imshow('seg {}'.format(i), segments[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite('l.png',segments[0])
    # cv2.imwrite('z.png',segments[1])
    # cv2.imwrite('k.png',segments[2])
    # cv2.imwrite('t.png',segments[3])
    # #cv2.imwrite('m.png',segments[4])
if __name__ == '__main__':
    main()

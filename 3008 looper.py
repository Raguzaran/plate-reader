from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import mathics as m3
#import imutils
#import os
import io
#import glob
#from getpass import getpass

#MySQL Database
from mysql.connector import connect
'''
try:
    with connect(
        host="localhost",
        user="####",
        password="####",
        database = "####"
    ) as connection:
        print(connection)

except Error as e:
    print(e)
'''
def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

#3 FIlters being examined
def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

#Letters being examined

#letterlist = ['2', '2(1)', '3', '3(1)', '5', '6', '7', '7(1)', '8', '8(1)', 'E(2)', 'E(3)', 'F', 'G', 'H', 'I', 'K', 'U(1)', 'U(2)', 'Y', 'Z', 'Z(1)']
letterlist = ['7']
letterresult = []

for letter in letterlist:
    img = cv2.imread(str('C:/Users/R5h2x/.spyder-py3/Plate pics/{} crop.jpg'.format(letter)))

    #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    #plt.subplot(121),plt.imshow(img)
    #plt.subplot(122),plt.imshow(dst)
    #plt.show()
    
    #Applying the Fourier Transform
    img = img[:,:,0]
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    
    #Examination of the Different Filters' Performance
    plt.figure(figsize=(6*2.5, 4*2.5), constrained_layout=False)
    
    plt.subplot(141), plt.imshow(RGB_img), plt.title("Original Image", fontsize = 20)
    
    HighPassCenter = center * idealFilterHP(50,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    plt.subplot(142), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Ideal", fontsize = 20)
    
    HighPassCenter = center * butterworthHP(50,img.shape,10)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    plt.subplot(143), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Butterworth", fontsize = 20)
    
    HighPassCenter = center * gaussianHP(50,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    plt.subplot(144), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Gaussian", fontsize = 20)
    
    plt.show()
    
    HighPassCenter = center * gaussianHP(50,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    img = np.abs(inverse_HighPass)
    
    
    
    ssim = []
    
    #SQL query to database containing all the Fourier Transformed letter files    
    db = connect(
            host="localhost",
            user="####",
            password="####",
            database = "####"
                )
    cursor = db.cursor()
    query = "SELECT img from letters"
    cursor.execute(query)
    record = cursor.fetchall()
    
    
    for i in record:
        
        file_like = io.BytesIO(i[0])    
        img2 = Image.open(file_like)
        img2 = np.array(img2)
        #img2 = cv2.fastNlMeansDenoisingColored(img2,None,75,10,7,21)
        img2 = img2[:,:,0]
        #print(img.shape)
        #print(img2.shape)
        
        newX, newY = int(img.shape[1]), (img.shape[0])
        img2 = cv2.resize(img2, (newX, newY))
        
        original = np.fft.fft2(img2)
        center = np.fft.fftshift(original)
        HighPassCenter = center * gaussianHP(50,img2.shape)
        HighPass = np.fft.ifftshift(HighPassCenter)
        inverse_HighPass = np.fft.ifft2(HighPass)
        img2 = np.abs(inverse_HighPass)
        
        #RGB_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        #cv2.imwrite("Resized {}m.jpg".format(i), img2)
        
        #print(img.shape)
        #print(img2.shape)
        
        #SSIM method to compare the likeness of each image
        (score, diff) = compare_ssim(img, img2, full=True)
        ssim.append(score)
        diff = (diff * 255).astype("uint8")
        
        #print("SSIM: {}. \n Difference: {}".format(score, diff))
        
        plt.figure()
        
        plt.subplot(121), plt.imshow(img)
        
        plt.subplot(122), plt.imshow(img2)#, plt.title("SSIM: {}".format(score))
        
        plt.show()
    
    print(max(ssim))    #Max Value and therefore best match returned
    print(ssim.index(max(ssim)))
    maxfind = ssim.index(max(ssim))
    
    cursor.execute("Select letter from letters limit 1 offset {}".format(maxfind))
    letterfind = cursor.fetchall()
    
    letterresult.append(str([str(j).strip("(',)") for j in letterfind]).strip("[']"))   #Prints full plate for multiple values in letterlist
    
print(" ".join(letterresult))
import os
import cv2
import random
import matplotlib.pyplot as plt

class ImageAugmentation:
    def __init__(self,path,file):
        self.path = path
        self.file = file
        self.image = cv2.imread(os.path.join(path,file))
        print(self.image)
    
    def rotate(self, image, angle=90, scale=1.0):
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image 
    
    def flip(self, image, vflip=False, hflip=False):
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
    
    def brightness_increase(self, image, value):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] += value
        '''if(hsv[:,:,2] > 255):
            hsv[:,:,2] = 255
        if(hsv[:,:,2] < 0):
            hsv[:,:,2] = 0'''
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def image_augment(self, save_path): 
        img = self.image.copy()
        img_flip = self.flip(img, vflip=True, hflip=False)
        img_rot = self.rotate(img)
        #img_gaussian = self.add_GaussianNoise(img)
        img_bright = self.brightness_increase(img,value = 40)
        image_all = self.flip(img, vflip=True, hflip=False)
        image_all = self.rotate(image_all)
        image_all = self.brightness_increase(image_all,value = 40)
        
        #plt.imshow(img_bright)
        name_int = self.file[:len(self.file)-4]
        path_value = os.path.join(save_path,str(name_int)+'_bright.JPG')
        path_value_2 = os.path.join(save_path,str(name_int)+'_vflip.JPG')
        path_value_3 = os.path.join(save_path,str(name_int)+'_rot.JPG')
        path_value_4 = os.path.join(save_path,str(name_int)+'_alltransformation.JPG')
        
        #cv2.imwrite(path_value_2, img_flip)
        #cv2.imwrite(path_value_3, img_rot)
        #cv2.imwrite(path_value_2, img_gaussian)
        cv2.imwrite(path_value,img_bright)
        #cv2.imwrite(path_value_4,image_all)

if __name__ == "__main__":
    path = 'D:/plantdisease/segmented_disease/Tomato/Tomato___Tomato_mosaic_virus' #you data path
    class_list = os.listdir(path)

    for image in class_list[:500]:
        
        raw_image = ImageAugmentation(path,image)
        raw_image.image_augment(path)
import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image

model = load_model('HAND2.h5')


def prediction(pred):
    return(str(pred))


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (1,28,28), interpolation = cv2.INTER_AREA)
  
    return img
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    l = []
    
    while True:
        
        cam_capture = cv2.VideoCapture(0)
        _, image_frame = cam_capture.read()  
    # Select ROI
        im2 = crop_image(image_frame, 300,300,200,200)
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
        image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
        im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)


        imagem = cv2.bitwise_not(im3)
        im4 = np.resize(imagem, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)
    

        pred_probab, pred_class = keras_predict(model, im5)
    
        curr = int(pred_class)
        curr = str(curr)
        
        cv2.putText(image_frame, curr, (400, 600), cv2.FONT_HERSHEY_COMPLEX, 4.0, (0, 0, 255), lineType=cv2.LINE_AA)
            
            
    
 
    # Display cropped image
        cv2.rectangle(image_frame, (300, 300), (500, 500), (255, 255, 00), 3)
        cv2.imshow("frame",image_frame)
        
        
    #cv2.imshow("Image4",resized_img)
        cv2.imshow("Image3",imagem)

        if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main()

cam_capture.release()
cv2.destroyAllWindows()

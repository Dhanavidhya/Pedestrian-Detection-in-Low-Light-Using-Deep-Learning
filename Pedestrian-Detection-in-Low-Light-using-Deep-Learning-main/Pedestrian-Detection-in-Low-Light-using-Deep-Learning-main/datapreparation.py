from PIL import Image,ImageEnhance
import os

img_path="C:\Users\User\Desktop\Main Project\Pedestrian\new_dataset"
output_path="new_dataset"

def enhancing(img_path):
    img=Image.open(img_path)
    img=ImageEnhance.Brightness(img)
    img=img.enhance(15.0) 
    return img

for filename in os.listdir(img_path):
    if filename.endswith(".png"): 
        image_path = os.path.join(img_path, filename)
        preprocessed_image = enhancing(image_path)
        output_filename = os.path.join(output_path, filename)
        preprocessed_image.save(output_filename)
    



 
            
            


    
        
        
        
        
        
        
        
        
    
        
        

        
        




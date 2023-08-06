from PIL import Image
import numpy as np

def increaseBrightness(image, amount):
    for row in range(len(image)):
        for pixel in range(len(image[row])):
            # Increase the value of each channel by the specified amount
            # print(pixel)
            rgb=image[row][pixel]
            r, g, b = rgb[0],rgb[1],rgb[2]
            r = min(r + amount, 255)  # Ensure the value is within the valid range [0, 255]
            g = min(g + amount, 255)
            b = min(b + amount, 255)
            image[row][pixel] = [r, g, b]
    return image
def changeContrast(image, contrast_factor):
    if contrast_factor == 0:
        return image

    image_normalized = image.astype(float)

    # Apply the contrast adjustment
    contrast_adjusted = (image_normalized - 128.0) * contrast_factor + 128.0

    # Clip the pixel values to the range [0, 255]
    contrast_adjusted = np.clip(contrast_adjusted, 0, 255)

    # Convert the pixel values back to the uint8 data type
    contrast_adjusted = contrast_adjusted.astype(np.uint8)

    return contrast_adjusted
def scroll(image,type=0):
    img = np.zeros_like(image)
    if(type==0):
        for row in range(len(image)):
            for pixel in range(len(image[row])):
                img[row][pixel] = image[len(image)-row-1][len(image[row][pixel])-pixel-1]
    if(type==1):
        for row in range(len(image)):
            for pixel in range(len(image[row])):
                img[row][pixel] = image[row][len(image[row][pixel])-pixel-1]      
    return img
def convert_to_gray(image):
    for row in range(len(image)):
        for pixel in range(len(image[row])):
            R,G,B=image[row][pixel]
            tr = 0.393*R + 0.769*G + 0.189*B
            tg = 0.349*R + 0.686*G + 0.168*B
            tb = 0.272*R + 0.534*G + 0.131*B
            r = min(tr, 255)  # Ensure the value is within the valid range [0, 255]
            g = min(tg, 255)
            b = min(tb, 255)
            image[row][pixel] = [r,g,b]
    return image
def blur(image):
    # matrix=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
    positions=[[0,0],[0,1],[1,1],[1,0],[-1,0],[0,-1],[-1,-1],[-1,1],[1,-1]]
    img = np.zeros_like(image,dtype=float)
    width=len(image)
    height=len(image[0])
    for row in range(1,width-1):
        for pixel in range(1,height-1):
            a=image[row][pixel]/9
            for position in positions:
                # if(row+position[0]>-1 and row+position[0]<width and pixel+position[1]>-1 and pixel+position[1]<height):
                    img[row+position[0]][pixel+position[1]]+=a
    return img.astype('uint8')
def tackleColor(a):
    if a>255:
        return 255
    if a<0:
        return 0
    return a
def sharpen(image):
    # matrix=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
    positions=[[0,1],[1,0],[-1,0],[0,-1]]
    img = np.zeros_like(image,dtype=float)
    width=len(image)
    height=len(image[0])
    for row in range(1,width-1):
        for pixel in range(1,height-1):
            for position in positions:
                img[row][pixel]-=image[row+position[0]][pixel+position[1]]
            a=np.array([image[row][pixel][0]*5,image[row][pixel][1]*5,image[row][pixel][2]*5])

            img[row][pixel]=a+img[row][pixel]
            img[row][pixel]=[tackleColor(img[row][pixel][0]),tackleColor(img[row][pixel][1]),tackleColor(img[row][pixel][2])]
    return img.astype('uint8')
def cut(image,size):
    img = np.zeros((size,size,3),dtype='uint8')
    width=len(image)
    x=int((width-size)/2)
    height=len(image[0])
    y=int((height-size)/2)
    for row in range(size):
        for pixel in range(size):
            img[row][pixel]=image[x+row][y+pixel]
    return img
def circel(image):
    width=len(image)
    height=len(image[0])
    centroids=[width/2,height/2]
    for row in range(0,width):
        for pixel in range(0,height):
            if (row-centroids[0])**2+(pixel-centroids[1])**2>(width/2)**2:
                image[row][pixel]=[0,0,0]
    return image
name=str(input('Input filename: '))
# name="challenge/owl.jpg"
# im = Image.open("./challenge/owl.jpg")
im = Image.open("./"+name)
name=name.split('.')[0]
im=np.array(im)

print('All                 [0]')
print('Increase Brightness [1]')
print('Constrast           [2]')
print('Rotate              [3]')
print('Convert 2 Gray      [4]')
print('Blur                [5]')
print('Sharpen             [6]')
print('Cut                 [7]')
print('Circle              [8]')
choice=int(input('Input number: '))
if choice==0:
    imcp=im.copy()
    img_result=increaseBrightness(imcp,100)
    image = Image.fromarray(img_result)
    image.save(name+'_increase_brightness.png')
    
    imcp=im.copy()
    img_result=changeContrast(imcp, 1.5)
    image = Image.fromarray(img_result)
    image.save(name+'_contrast.png')
    
    imcp=im.copy()
    img_result=scroll(imcp,0)
    image = Image.fromarray(img_result)
    image.save(name+'_Rotate.png')

    imcp=im.copy()
    img_result=convert_to_gray(imcp)
    image = Image.fromarray(img_result)
    image.save(name+'_gray.png')

    imcp=im.copy()
    img_result=blur(imcp)
    image = Image.fromarray(img_result)
    image.save(name+'_blur.png')
    
    imcp=im.copy()
    img_result=sharpen(imcp)
    image = Image.fromarray(img_result)
    image.save(name+'_sharp.png')

    imcp=im.copy()
    img_result=cut(imcp,225)
    image = Image.fromarray(img_result)
    image.save(name+'_cut.png')

    imcp=im.copy()
    img_result=circel(imcp)
    image = Image.fromarray(img_result)
    image.save(name+'_circle.png')

if choice==1:
    img_result=increaseBrightness(im,100)
    image = Image.fromarray(img_result)
    image.save(name+'_increase_brightness.png')
if choice==2:
    img_result=changeContrast(im, 1.5)
    image = Image.fromarray(img_result)
    image.save(name+'_contrast.png')
if choice==3:
    a=int(input('Rotate 180(0),rotate 90(1):'))
    img_result=scroll(im,a)
    image = Image.fromarray(img_result)
    image.save(name+'_Rotate.png')
if choice==4:
    img_result=convert_to_gray(im)
    image = Image.fromarray(img_result)
    image.save(name+'_gray.png')
if choice==5:
    img_result=blur(im)
    image = Image.fromarray(img_result)
    image.save(name+'_blur.png')
if choice==6:
    img_result=sharpen(im)
    image = Image.fromarray(img_result)
    image.save(name+'_sharp.png')
if choice==7:
    img_result=cut(im,225)
    image = Image.fromarray(img_result)
    image.save(name+'_cut.png')
if choice==8:
    img_result=circel(im)
    image = Image.fromarray(img_result)
    image.save(name+'_circle.png')
# img_result = img_result.reshape((im.shape[0], im.shape[1], im.shape[2]))
# image = Image.fromarray(img_result)
# image.save('ok.png')
# image.show()

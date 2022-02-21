import pytesseract as ocr
import numpy as np
import cv2 as cv2
import re
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from pytesseract import Output
from PIL import Image as PIL
from google.cloud import vision


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding
#def thresholding(image):
#    # threshold the image, setting all foreground pixels to
#    # 255 and all background pixels to 0
#    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# PATH Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

# tipando a leitura para os canais de ordem RGB
imagem = PIL.open('c:\CNH.jpg').convert('RGB')

img_orig = cv2.imread('c:\CNH.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2)

client_options = {'api_endpoint': 'vision.googleapis.com'}
client = vision.ImageAnnotatorClient(client_options=client_options)

# h, w, c = img.shape
# boxes = ocr.image_to_boxes(img)
# for b in boxes.splitlines():
#    b = b.split(' ')
#    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
#src = cv2.GaussianBlur(img_orig,(3, 3),0);

#img_blur = cv2.medianBlur(img_orig,5).astype('uint8')
#img = cv2.adaptiveThreshold(img_orig,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#img = cv2.threshold(img_orig,127,255,cv2.THRESH_BINARY)
#img_blur = cv2.GaussianBlur(img_orig,(5,5),0)

##laplacian = cv2.Laplacian(img_orig,cv2.CV_8UC1)
# Otsu's thresholding after Gaussian filtering
##threshold = cv2.threshold(laplacian,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
# Convert the image to grayscale
#src_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY);

#img_reduz = cv2.GaussianBlur(src_gray,(3,3),0)

# Output dtype = cv2.CV_8U
#sobelx8u = cv2.Sobel(src_gray,cv2.CV_8U,2,2,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
#sobelx64f = cv2.Sobel(img_reduz,cv2.CV_64F,2,2,ksize=5)
#abs_sobel64f = np.absolute(sobelx64f)
#sobel_8u = np.uint8(abs_sobel64f)

#plt.subplot(1,3,1),plt.imshow(img_orig,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
#plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
#plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
#plt.show()

G_x = cv2.Sobel(img_orig,cv2.CV_64F,0,1)
G_y = cv2.Sobel(img_orig,cv2.CV_64F,1,0)
#G = np.abs(G_x) + np.abs(G_y)
G = np.sqrt(np.power(G_x,2)+np.power(G_y,2))
height, width = G.shape[:2]
cv2.namedWindow('Vini', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vini', width, height)
G=(G/np.max(G))*2
cv2.imshow('Vini',G)

#cv2.imshow('Transf', laplacian)
#cv2.waitKey(0)

#clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
#res = clahe.apply(img_orig)
#height, width = sobel_8u.shape[:2]
#cv2.namedWindow('Res', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Res', width, height)
#cv2.imshow('Res', sobel_8u)

img_tratada = PIL.fromarray(G,mode="RGBA")
imag = np.asarray(img_tratada)
#img_tratada = cv2.imread(G)
#cv2.imshow('Vini',img_tratada)

d = ocr.image_to_data(imag, output_type=Output.DICT)
keys = list(d.keys())

date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
cpf_pattern = '/.(\d{3}.){2}\d{3}-\d{2}$/gm'

n_boxes = len(d['text'])
##
Padrao = False
for i in range(n_boxes):
    if int(d['conf'][i]) > 30:
        if re.match(date_pattern, d['text'][i]):
            print("DataOK")
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            imag = cv2.rectangle(imag, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x = d["left"][i]
            y = d["top"][i]
            w = d["width"][i]
            h = d["height"][i]
            # extract the OCR text itself along with the confidence of the
            # text localization
            text = d["text"][i]
            conf = int(d["conf"][i])
            print("Confiança: {}".format(conf))
            print("Texto: {}".format(text))
            print("")
            Padrao = True
        if re.match(cpf_pattern, d['text'][i]):
            print("CPFOK")
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            imag = cv2.rectangle(imag, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = d["text"][i]
            conf = int(d["conf"][i])
            print("Confiança: {}".format(conf))
            print("Texto: {}".format(text))
            print("")
            Padrao = True
        if Padrao==False:
            campo=d["text"][i]
            print("Campo:"+campo)
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            imag = cv2.rectangle(imag, (x, y), (x + w, y + h), (255, 0, 0), 2)
        Padrao = False

#show_images(img, 3, ["gray", "rnoise", "dilate", "erode", "thresh", "deskew", "opening", "canny"])
cv2.waitKey(0)

height, width = imag.shape[:2]
cv2.namedWindow('Data', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Data', width, height)
cv2.imshow('Data', imag)
phrase = ocr.image_to_string(imag, lang='por')
print("Resultado : ", phrase)
cv2.waitKey(0)
cv2.destroyAllWindows()
# convertendo em um array editável de numpy[x, y, CANALS]
# npimagem = np.asarray(imagem).astype(np.uint8)

# diminuição dos ruidos antes da binarização
# npimagem[:, :, 0] = 0 # zerando o canal R (RED)
# npimagem[:, :, 2] = 0 # zerando o canal B (BLUE)

# atribuição em escala de cinza
# im = cv2.cvtColor(npimagem, cv2.COLOR_RGB2GRAY)

# aplicação da truncagem binária para a intensidade
# pixels de intensidade de cor abaixo de 127 serão convertidos para 0 (PRETO)
# pixels de intensidade de cor acima de 127 serão convertidos para 255 (BRANCO)
# A atrubição do THRESH_OTSU incrementa uma análise inteligente dos nivels de truncagem
# ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# reconvertendo o retorno do threshold em um objeto do tipo PIL.Image
# binimagem = Image.fromarray(thresh)

# chamada ao tesseract OCR por meio de seu wrapper
# phrase = ocr.image_to_string(binimagem, lang='por')

# impressão do resultado
# print(phrase)

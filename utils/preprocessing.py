from skimage.transform import resize
import numpy as np
from PIL import Image

def load_image(image_path):
    img_gray_pil = Image.open(image_path).convert('L')
    image = np.array(img_gray_pil)
    image = image.astype(np.float32) / 255.
    image = 1. - image

    return image


def preprocess(img, input_size, border_size=8):
    
    h_target, w_target = input_size

    n_height = min(h_target - 2 * border_size, img.shape[0])
    
    scale = n_height / img.shape[0]
    n_width = min(w_target - 2 * border_size, int(scale * img.shape[1]))

    img = resize(image=img, output_shape=(n_height, n_width)).astype(np.float32)

    # right pad image to input_size
    img  = np.pad(img, ((border_size, h_target - n_height - border_size), (border_size, w_target - n_width - border_size),),
                                                    mode='median')

    return img




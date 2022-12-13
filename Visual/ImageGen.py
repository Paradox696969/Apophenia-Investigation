from PIL import Image
import random
import os

def overlayImageRandom(imgh, imgw, img_path, save_path, filename, alpha):
    img = Image.new("RGB", (imgw, imgh), (255, 255, 255))
    img2 = Image.open(img_path)
    img2.convert("RGB")
    img2 = img2.resize((imgw, imgh))
    print(img.mode, img2.mode)
    print(img.size, img2.size)

    for x in range(img.width):
        for y in range(img.height):
            img.putpixel((x, y), random.choice([(255, 255, 255), (0, 0, 0)]))
    try:
        img3 = Image.blend(img, img2, alpha=alpha)
    except:
        return
    try:
        img3.save(f"{save_path}{filename}")
    except FileNotFoundError or FileExistsError:
        img3.save(f"{save_path}{filename}")
    except Exception as e:
        print(e)
        return
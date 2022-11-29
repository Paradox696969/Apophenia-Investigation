from PIL import Image
import random

img = Image.new("RGBA", (256, 256), (255, 255, 255, 150))
img2 = Image.open("Paradox.png")
img2 = img2.resize((256, 256))

for x in range(img.width):
    for y in range(img.height):
        img.putpixel((x, y), random.choice([(255, 255, 255, 150), (0, 0, 0, 150)]))

img3 = Image.blend(img, img2, alpha=0.4)

img3.show()
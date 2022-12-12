import os
import shutil
import pandas

df = pandas.read_csv("Visual/images.csv")
df = df.reset_index()

path = "Visual/Normal Datasets/Clothing"

for index, row in df.iterrows():
    filename = f"{row['image']}.jpg"
    label = f"{row['label']}"
    try:
        shutil.move(f"Visual/Normal Datasets/Clothing/{filename}", f"Visual/Normal Datasets/Clothing/{label}/{filename}")
    except:
        os.mkdir(f"Visual/Normal Datasets/Clothing/{label}/")
        shutil.move(f"Visual/Normal Datasets/Clothing/{filename}", f"Visual/Normal Datasets/Clothing/{label}/{filename}")

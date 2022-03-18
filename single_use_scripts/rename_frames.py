import os

dir = "/media/andivanov/DATA/dynamic_backgrounds_captured/asfandyar/take03/take03/left"

for name in os.listdir(dir):
    frameid = int(name[:-4].split("_")[-1])
    new_name = f"{frameid:04d}.jpg"
    os.rename(os.path.join(dir, name), os.path.join(dir, new_name))

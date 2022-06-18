##test preprocessing
import os
import random
from shutil import copyfile

random.seed(0)

files_list = os.listdir("./data/Green-Unripped/")
data_size = len(files_list)
c0_dir = "./data/Green-Unripped/"
random.shuffle(files_list)
train_files = files_list[:round(0.75*data_size)]
val_files = files_list[round(0.75*data_size):]
for img in train_files:
    img_path = os.path.join(c0_dir, img)
    new_filename = 'C0_' + img.split(".")[0]
    copyfile(img_path, './data/train/'+new_filename+'.png')
for img in val_files:
    img_path = os.path.join(c0_dir, img)
    new_filename = 'C0_' + img.split(".")[0]
    copyfile(img_path, './data/val/'+new_filename+'.png')

files_list = os.listdir("./data/Orange/")
data_size = len(files_list)
c1_dir = "./data/Orange/"
random.shuffle(files_list)
train_files = files_list[:round(0.75*data_size)]
val_files = files_list[round(0.75*data_size):]
for img in train_files:
    img_path = os.path.join(c1_dir, img)
    new_filename = 'C1_' + img.split(".")[0]
    copyfile(img_path, './data/train/'+new_filename+'.png')
for img in val_files:
    img_path = os.path.join(c1_dir, img)
    new_filename = 'C0_' + img.split(".")[0]
    copyfile(img_path, './data/val/'+new_filename+'.png')

files_list = os.listdir("./data/Red-Ripped/")
data_size = len(files_list)
c2_dir = "./data/Red-Ripped/"
random.shuffle(files_list)
train_files = files_list[:round(0.75*data_size)]
val_files = files_list[round(0.75*data_size):]
for img in train_files:
    img_path = os.path.join(c2_dir, img)
    new_filename = 'C2_' + img.split(".")[0]
    copyfile(img_path, './data/train/'+new_filename+'.png')
for img in val_files:
    img_path = os.path.join(c2_dir, img)
    new_filename = 'C2_' + img.split(".")[0]
    copyfile(img_path, './data/val/'+new_filename+'.png')

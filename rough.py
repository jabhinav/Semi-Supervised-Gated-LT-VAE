import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

data_dir = './data'
import csv

# Read labels from the csv file
CELEBA_LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips',
                 'Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                 'Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
                 'Mustache','Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin','Pointy_Nose','Receding_Hairline',
                 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                      'Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', 'No_Beard', 'Pale_Skin',
                      'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']

labels = pd.read_csv(data_dir + '/list_attr_celeba.csv', names=['image_id'] + CELEBA_LABELS, header=0)
easy_labels = pd.read_csv(data_dir + '/list_attr_celeba.csv', names=['image_id'] + CELEBA_LABELS, header=0)
easy_labels = easy_labels[['image_id'] + CELEBA_EASY_LABELS]

with open(os.path.join(data_dir, 'list_attr_celeba.csv')) as csv_file:
    data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

headers = data[0]
data = data[0 + 1 :]

indices = [row[0].split(",")[0] for row in data]
data = [row[0].split(",")[1:] for row in data]
data_int = [list(map(int, i)) for i in data]

print(data_int[0])




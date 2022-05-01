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

sub_label_inds = [i for i in range(len(CELEBA_LABELS)) if CELEBA_LABELS[i] in CELEBA_EASY_LABELS]

with open(os.path.join(data_dir, 'list_attr_celeba.csv')) as csv_file:
    data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

headers = data[0]
data = data[0 + 1:]

indices = [row[0].split(",")[0] for row in data]
data = [row[0].split(",")[1:] for row in data]
data_int = [list(map(int, i)) for i in data]
data_np = np.array(data_int)
# Map labels -1 to 0 and 1 to 1
data_np[data_np == -1] = 0
data_np[data_np == 1] = 1

# Filter data to only include the labels we want
data_np = data_np[:, sub_label_inds]

# Collect the labels from CELEBA_EASY_LABELS corresponding to the indices of the 1s
where_one_x, where_one_y = np.nonzero(data_np)
cut_idx = np.flatnonzero(np.r_[True, where_one_x[1:] != where_one_x[:-1],True])
grouped_indices = [where_one_y[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
grouped_labels = [[CELEBA_EASY_LABELS[i] for i in group] for group in grouped_indices]

def create_gating_matrix(grouped_indices, n_labels):
    """
    Creates a gating matrix for observed data from the cooccurance matrix
    """
    n_elems = len(grouped_indices)
    cooccurance_matrix = np.zeros((n_labels, n_labels))
    for group in grouped_indices:
        for i in group:
            for j in group:
                if j != i:
                    cooccurance_matrix[i, j] += 1
    # Normalize the matrix (for relative frequencies)
    # cooccurance_matrix = cooccurance_matrix / cooccurance_matrix.sum(axis=1, keepdims=True)
    # We want absolute frequencies
    gating_matrix = cooccurance_matrix / n_elems
    # Set diagonal to 1
    np.fill_diagonal(gating_matrix, 1)
    return gating_matrix

gating_matrix = create_gating_matrix(grouped_indices, len(CELEBA_EASY_LABELS))
# Use Pandas to create a dataframe from the co-occurance matrix
gating_df = pd.DataFrame(gating_matrix, index=CELEBA_EASY_LABELS, columns=CELEBA_EASY_LABELS)
# Print the dataframe
print(gating_matrix)
# Save the dataframe to a csv file
gating_df.to_csv(os.path.join(data_dir, 'gating_matrix.csv'))


import os
import numpy as np

dir = './data'
threshold = 0.3
CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                      'Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', 'No_Beard', 'Pale_Skin',
                      'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']


gating_mat = np.load(os.path.join(dir, 'gating_matrix_1.0.npy'))

(all_z, all_y) = np.where(gating_mat>threshold)
for i, (z, y) in enumerate(zip(all_z, all_y)):
    if z != y:
        print("{}-{}: {}".format('z_{}'.format(z+1), CELEBA_EASY_LABELS[y], gating_mat[z, y]))

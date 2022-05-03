import os
import numpy as np

CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                      'Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', 'No_Beard', 'Pale_Skin',
                      'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']

# ########################################## Qualitative Analysis ########################################## #
threshold = 0.6

for sup in [1.0, 0.5, 0.2]:
    print("\nSupervision:", sup)
    print("-------------------- Init Gating Matrix --------------------")
    init_gating_mat = np.load(os.path.join('./data', 'gating_matrix_{}.npy'.format(sup)))
    (all_z, all_y) = np.where(init_gating_mat>threshold)
    for i, (z, y) in enumerate(zip(all_z, all_y)):
        if z != y:
            print("{}-{}: {}".format('z_{}'.format(z+1), CELEBA_EASY_LABELS[y], init_gating_mat[z, y]))

    print("-------------------- Learned Gating Matrix --------------------")
    learned_gating_mat = np.load(os.path.join('./models', 'params_{}_learnable'.format(sup), 'learned_gating_matrix_best.npy'))
    (all_z, all_y) = np.where(learned_gating_mat>threshold)
    for i, (z, y) in enumerate(zip(all_z, all_y)):
        if z != y:
            print("{}-{}: {}".format('z_{}({})'.format(z+1, CELEBA_EASY_LABELS[z]), CELEBA_EASY_LABELS[y], learned_gating_mat[z, y]))


# ########################################## Quantitative Analysis ########################################## #
init_cumulative_count ={
    '1.0': [],
    '0.5': [],
    '0.2': []
}
learned_cumulative_count ={
    '1.0': [],
    '0.5': [],
    '0.2': []
}

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for sup in [1.0, 0.5, 0.2]:
    init_gating_mat = np.load(os.path.join('./data', 'gating_matrix_{}.npy'.format(sup)))
    learned_gating_mat = np.load(
        os.path.join('./models', 'params_{}_learnable'.format(sup), 'learned_gating_matrix_best.npy'))
    for threshold in thresholds:
        (all_z, all_y) = np.where(init_gating_mat > threshold)
        non_equal_z_y = [(z, y) for z, y in zip(all_z, all_y) if z != y]
        init_cumulative_count[str(sup)].append((threshold, len(non_equal_z_y)))

        (all_z, all_y) = np.where(learned_gating_mat > threshold)
        non_equal_z_y = [(z, y) for z, y in zip(all_z, all_y) if z != y]
        learned_cumulative_count[str(sup)].append((threshold, len(non_equal_z_y)))

print(init_cumulative_count)

print(learned_cumulative_count)



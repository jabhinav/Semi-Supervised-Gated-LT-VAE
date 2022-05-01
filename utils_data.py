from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple
import os
import csv
import PIL
import numpy as np
import logging
from utils import create_gating_matrix
import pandas as pd

logging.basicConfig(filename="./logs", filemode='w', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CELEBA_LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips',
                 'Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                 'Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',
                 'Mustache','Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin','Pointy_Nose','Receding_Hairline',
                 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                      'Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', 'No_Beard', 'Pale_Skin',
                      'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']


CSV = namedtuple("CSV", ["header", "index", "data"])


class DataLoader:
    def __init__(self, data_dir: str, cached_data: CSV, batch_size, shuffle: bool = True):
        self.data_dir = data_dir
        self.cached_data = cached_data
        self.bs = batch_size
        self.n_s = len(cached_data.data)
        self.idxs = list(range(len(cached_data.data)))

        if shuffle:
            np.random.shuffle(self.idxs)

        # set the iterator to the beginning
        self.start = 0

        # Do one step to get the first batch
        self.Xs, self.ys = self.read_data(self.get_batch())

    def read_data(self, idxs, normalise=True):
        images = [self.cached_data.index[i] for i in idxs]
        labels = self.cached_data.data[idxs]
        X = []
        # Load the images
        for image in images:
            img = PIL.Image.open(os.path.join(self.data_dir, image))
            img = np.array(img)
            # Resize the image to 64x64 using PIL
            img = PIL.Image.fromarray(img).resize((64, 64))
            img = np.array(img, dtype=np.float32)
            if normalise:
                img = img / 255.0
            X.append(img)
        X = np.array(X)
        return X, labels

    def get_batch(self):
        if self.start + self.bs < self.n_s:
            batched_idxs = self.idxs[self.start:self.start + self.bs]
            self.start = self.start + self.bs
        else:
            batched_idxs = self.idxs[self.start:] + self.idxs[:self.bs - (self.n_s - self.start)]
            self.start = (self.start + self.bs) % self.n_s
        return batched_idxs

    def step(self):
        while True:
            yield self.Xs, self.ys
            self.Xs, self.ys = self.read_data(self.get_batch())

    def reset(self):
        self.start = 0


class CelebAReader:
    def __init__(self, root, sup_frac, batch_size):
        self.root = root

        self.split_map = {
            "train": 162770,
            "valid": 19867,
            "test": 19962,
        }

        self.sub_label_inds = [i for i in range(len(CELEBA_LABELS)) if CELEBA_LABELS[i] in CELEBA_EASY_LABELS]
        self.attr = self._load_csv("list_attr_celeba.csv", header=0)

        self.sup_frac = sup_frac
        self.batch_size = batch_size

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:

        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0].split(",")[0] for row in data]
        data = [row[0].split(",")[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        data_np = np.array(data_int)
        # Map labels -1 to 0 and 1 to 1
        data_np[data_np == -1] = 0
        data_np[data_np == 1] = 1

        # Filter data to only include the labels we want
        data_np = data_np[:, self.sub_label_inds]
        headers = ['image_id'] + CELEBA_EASY_LABELS

        return CSV(headers, indices, data_np)

    def load_split_data(self):
        cached_data = {}
        cached_data["train"] = CSV(self.attr.header, self.attr.index[:self.split_map["train"]], self.attr.data[:self.split_map["train"]])
        # Split the training data into supervised and unsupervised
        if self.sup_frac == 0.0:
            cached_data["unsup"] = cached_data["train"]
        elif self.sup_frac == 1.0:
            cached_data["sup"] = cached_data["train"]
        else:
            sup_data = int(self.split_map["train"] * self.sup_frac)
            cached_data["sup"] = CSV(self.attr.header, cached_data["train"].index[:sup_data], cached_data["train"].data[:sup_data])
            cached_data["unsup"] = CSV(self.attr.header, cached_data["train"].index[sup_data:], cached_data["train"].data[sup_data:])
        cached_data["valid"] = CSV(self.attr.header, self.attr.index[self.split_map["train"]:self.split_map["train"] + self.split_map["valid"]],
                                   self.attr.data[self.split_map["train"]:self.split_map["train"] + self.split_map["valid"]])
        cached_data["test"] = CSV(self.attr.header, self.attr.index[self.split_map["train"] + self.split_map["valid"]:],
                                  self.attr.data[self.split_map["train"] + self.split_map["valid"]:])
        return cached_data

    def set_gating_prob(self, cached_data):
        # Load the gating matrix if it exists
        if os.path.exists(os.path.join(self.root, "gating_matrix_{}.npy".format(self.sup_frac))):
            logger.info("Loading gating matrix from {}".format(os.path.join(self.root, "gating_matrix_{}.npy".format(self.sup_frac))))
            mu = np.load(os.path.join(self.root, "gating_matrix_{}.npy".format(self.sup_frac)))
            self.init_gating_prob = mu
        else:
            logger.info("No gating matrix found, initializing...")
            if self.sup_frac == 0.0:
                mu = np.ones((len(CELEBA_EASY_LABELS), len(CELEBA_EASY_LABELS)))/2.0
                np.fill_diagonal(mu, 1.)
            else:
                sup = cached_data["sup"].data
                valid = cached_data["valid"].data
                # Merge the supervised and validation data
                data = np.concatenate((sup, valid), axis=0)
                where_one_x, where_one_y = np.nonzero(data)
                cut_idx = np.flatnonzero(np.r_[True, where_one_x[1:] != where_one_x[:-1], True])
                grouped_indices = [where_one_y[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])]

                mu = create_gating_matrix(grouped_indices, len(CELEBA_EASY_LABELS))
            self.init_gating_prob = mu

            # Save the gating matrix
            np.save(os.path.join(self.root, "gating_matrix_{}.npy".format(self.sup_frac)), mu)
            indexes = ["z{}".format(i+1) for i in range(len(CELEBA_EASY_LABELS))]
            gating_df = pd.DataFrame(mu, index=indexes, columns=CELEBA_EASY_LABELS)
            gating_df.to_csv(os.path.join(self.root, "gating_matrix_{}.csv".format(self.sup_frac)))

            logger.info("Gating matrix saved to {}".format(os.path.join(self.root, "gating_matrix_{}.npy".format(self.sup_frac))))

    def setup_data_loaders(self):
        if self.sup_frac == 0.0:
            modes = ["unsup", "test"]
        elif self.sup_frac == 1.0:
            modes = ["sup", "test", "valid"]
        else:
            modes = ["unsup", "test", "sup", "valid"]

        cached_data = self.load_split_data()

        # Set the gating matrix
        self.set_gating_prob(cached_data)

        loaders = {}
        for mode in modes:
            loaders[mode] = DataLoader(os.path.join(self.root, 'img_align_celeba'), cached_data[mode],
                                       batch_size=self.batch_size, shuffle=True)

        return loaders


def do_test():
    ROOT = "./data"
    reader = CelebAReader(ROOT, 0.0, 16)
    loaders = reader.setup_data_loaders()
    test_iterator = iter(loaders["test"].step())
    X, y = next(test_iterator)
    print(X.shape, y.shape)
    print(y[0])
    X, y = next(test_iterator)
    print(X.shape, y.shape)
    print(y[0])
    print("Done")


if __name__ == '__main__':
    do_test()
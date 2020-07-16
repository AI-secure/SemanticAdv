import os
import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class CelebA_withname(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self,
                 image_dir,
                 attr_path,
                 selected_attrs,
                 transform,
                 test_idlist=[],
                 test_namelist=[],
                 sub_folder=''):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.test_namelist = test_namelist
        self.id_name_dic = dict(zip(test_namelist, test_idlist))
        self.preprocess()
        self.num_images = len(self.test_dataset)
        self.sub_folder = sub_folder

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        assert self.test_namelist != []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]

            if filename in self.test_namelist:
                id = self.id_name_dic[filename]

                values = split[1:]

                label = []
                for attr_name in self.selected_attrs:
                    idx = self.attr2idx[attr_name]
                    label.append(values[idx] == '1')

                self.test_dataset.append([filename, label, id])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.test_dataset
        filename, label, id = dataset[index]
        image = Image.open(os.path.join(self.image_dir, id, self.sub_folder, filename))
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir,
               attr_path,
               selected_attrs,
               crop_size=112,
               image_size=112,
               batch_size=16,
               num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA_withname(image_dir, attr_path, selected_attrs, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader


def create_dic(image_dir,
               attr_path,
               selected_attrs,
               crop_size=112,
               image_size=112,
               test_idlist=[],
               test_namelist=[],
               sub_folder=''):
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA_withname(image_dir, attr_path, selected_attrs, transform,
                              test_idlist, test_namelist, sub_folder)

    dic_label = {}
    dic_image = {}

    for i in range(len(dataset)):
        img, label, filename = dataset[i]
        dic_label[filename] = label
        dic_image[filename] = img

    return dic_label, dic_image

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from .sampling import noniid_nlp
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler

class DRODataset(Dataset):
    def __init__(self, dataset, n_groups, n_classes, batch_size):
        self.dataset = dataset
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.batch_size = batch_size
        group_array = []
        y_array = []

        for i in range(len(dataset)):
            x,y,g = self.get_items(i)
            group_array.append(g)
            y_array.append(y)
        
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def get_items(self,idx):
        return self.dataset[idx]

    def __getitem__(self, idx):
        return self.dataset[idx][:2]
     
    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g in self:
            return x.size()

    def get_loader(self, train, **kwargs):
        if not train: # Validation or testing
            shuffle = False
            sampler = None
       
        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            batch_size = self.batch_size,
            **kwargs)
        return loader


def prepare_confounder_data():
    full_dataset = CUBDataset()

    splits = ['train', 'val', 'test']
    
    subsets = full_dataset.get_splits(splits)

    # dro_subsets = [DRODataset(subsets[split], n_groups=4,
    #                           n_classes=2) \
    #                for split in splits]

    return subsets
    

class ConfounderDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
            self.eval_transform):
            img = self.eval_transform(img)
        # Flatten if needed
        
        x = img

        return x,y,g

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            if split == 'train':
                mask_0 = (self.split_array == self.split_dict[split]) & (self.confounder_array == 0)
                mask_1 = (self.split_array == self.split_dict[split]) & (self.confounder_array == 1)
                indices_0 = np.where(mask_0)[0]
                indices_1 = np.where(mask_1)[0]
                subsets['train_0'] = Subset(self, indices_0)
                subsets['train_1'] = Subset(self, indices_1)

            else:
                mask = (self.split_array == self.split_dict[split])
                indices = np.where(mask)[0]
                subsets[split] = Subset(self, indices)

        return subsets


class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self):

        self.data_dir = '/home/peterpan/scheduler/data/waterbirds'
        
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Set transform
        
        self.features_mat = None
        self.train_transform = get_transform_cub(
            train=True,
            )
        self.eval_transform = get_transform_cub(
            train=False,
            )


def get_transform_cub(train):
    scale = 256.0/224.0
    target_resolution = (224,224)
    assert target_resolution is not None

    if not train:
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
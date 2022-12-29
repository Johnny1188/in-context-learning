import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import random

DATA_PATH = os.getenv("DATA_PATH")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


class RandomLinearProjectionMNIST(Dataset):
    def __init__(
        self,
        orig_mnist_dataset,
        num_tasks=10,
        seq_len=10,
        permuted_images_frac=1.0,
        permuted_labels_frac=1.0,
        labels_shifted_by_one=False,
        spare_mem=False,
    ):
        self.orig_mnist_dataset = orig_mnist_dataset
        self.num_tasks = num_tasks
        self.seq_len = seq_len
        self.labels_shifted_by_one = labels_shifted_by_one
        self.spare_mem = spare_mem

        self.task_idxs = []
        self.lin_transforms = []
        self.label_perms = []
        for task_idx in range(self.num_tasks):
            # randomly sample a subset of the MNIST dataset
            # task = torch.utils.data.Subset(orig_mnist_dataset, torch.randperm(len(orig_mnist_dataset))[:self.seq_len])
            task_dataset_idxs = torch.randperm(len(orig_mnist_dataset))[:self.seq_len]
            self.task_idxs.append(task_dataset_idxs)
            
            # generate random linear projections for each task
            if np.random.rand() < permuted_images_frac:
                if self.spare_mem: # save only the random seed
                    lin_tranform = task_idx
                else:
                    lin_tranform = torch.normal(0, 1/784, (784, 784))
            else:
                if self.spare_mem:
                    lin_tranform = None
                else:
                    lin_tranform = torch.eye(784)
            self.lin_transforms.append(lin_tranform)

            # generate random label permutations for each task
            if np.random.rand() < permuted_labels_frac:
                label_perm = torch.randperm(10)
            else:
                label_perm = torch.arange(10)
            self.label_perms.append(label_perm)
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, idx):
        # get the task and projection for the given index
        task = self.get_task(idx)
        lin_tranform = self.get_lin_transform(idx)
        label_perm = self.label_perms[idx] # (10,)

        x, y = [], []
        for example in task:
            # standardized projection
            proj = (lin_tranform @ example[0].view(784)) # (784,)
            proj = (proj - proj.mean()) / (proj.std() + 1e-16)
            # permuted label
            perm_label = label_perm[example[1]]
            x.append(proj)
            y.append(perm_label)
        
        x = torch.stack(x)
        y = torch.stack(y)
        # concatenate the projected images and permuted labels
        if self.labels_shifted_by_one:
            # append labels to images ((x1,0), (x2,y1), ..., (xn-1, yn-2), (xn, yn-1)) - all except the first one
            y_shifted = torch.cat((torch.zeros(size=(1, 10)), F.one_hot(y[:-1], num_classes=10)), dim=0) # (seq_len, 10)
            x = torch.concat((x, y_shifted), dim=1) # (seq_len, 784 + 10), y (seq_len,)
        else:
            # append labels to images ((x1,y1), (x2,y2), ..., (xn-1, yn-1), (xn, 0)) - all except the last one
            y_masked_last = torch.cat((F.one_hot(y[:-1], num_classes=10), torch.zeros(size=(1, 10))), dim=0) # (seq_len, 10)
            x = torch.concat((x, y_masked_last), dim=1)
            y = y[-1] # (10,)

        return x, y # (seq_len, 784 + 10), (seq_len,) if self.labels_shifted_by_one else (1,)

    def get_lin_transform(self, task_idx):
        if self.spare_mem: # storing only the random seed
            if self.lin_transforms[task_idx] is None:
                lin_tranform = torch.eye(784)
            else:
                lin_tranform = torch.normal(0, 1/784, (784, 784), generator=torch.Generator().manual_seed(self.lin_transforms[task_idx]))
        else:
            lin_tranform = self.lin_transforms[task_idx] # (784, 784)
        return lin_tranform

    def get_task(self, task_idx):
        task_idxs = self.task_idxs[task_idx]
        return torch.utils.data.Subset(self.orig_mnist_dataset, task_idxs)

    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])


def select_from_classes(x, y, classes_to_select):
    samples_mask = np.array([s in classes_to_select for s in y])
    return x[samples_mask,:], y[samples_mask]


# Data transformations and loading - MNIST
def get_mnist_data_loaders(batch_size=32, flatten=False, drop_last=True, only_classes=None, img_size=28):
    # build transforms
    img_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if img_size < 28 and img_size >= 24:
        img_transformation.transforms.append(transforms.Resize(img_size))
    elif img_size < 24:
        img_transformation.transforms.append(transforms.CenterCrop(24))
        img_transformation.transforms.append(transforms.Resize(img_size))
    if flatten:
        img_transformation.transforms.append(transforms.Lambda(lambda x: torch.flatten(x)))

    train_dataset = datasets.MNIST(DATA_PATH, train=True, download=False, transform=img_transformation)
    if only_classes != None: # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(train_dataset.targets, only_classes if type(only_classes) == torch.Tensor else torch.tensor(only_classes))
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    
    test_dataset = datasets.MNIST(DATA_PATH, train=False, download=False, transform=img_transformation)
    if only_classes != None: # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(test_dataset.targets, only_classes if type(only_classes) == torch.Tensor else torch.tensor(only_classes))
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, datasets.MNIST.classes


# Data transformations and loading - EMNIST
def get_emnist_data_loaders(batch_size=32, drop_last=True):
    img_transformation = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(DATA_PATH, train=True, download=False, transform=img_transformation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    test_dataset = datasets.EMNIST(DATA_PATH, train=False, download=False, transform=img_transformation)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, datasets.EMNIST.classes

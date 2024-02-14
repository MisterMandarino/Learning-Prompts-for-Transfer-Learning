from torchvision.datasets import Food101, CIFAR100, CIFAR10, MNIST

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets
import wget
import tarfile

import os
HOME = os.getcwd()

# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class OxfordPetsDataset(Dataset):
    def __init__(self, directory, split='test', transform=None):

        file = os.path.join(directory, 'annotations', split+'.txt')
        self.data = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                self.data.append((line[0], int(line[1]) - 1))

        self.transform = transform
        self.directory = directory

    def __getitem__(self, index):
        img = self.load_image(index)
        class_idx = self.data[index][1]

        if self.transform:
            img = self.transform(img)
            if img.shape[0] == 4:
                img = img[1:]
        return img, class_idx

    def __len__(self):
        return len(self.data)

    def load_image(self, index):
        image_path = os.path.join(self.directory, 'images', self.data[index][0]+'.jpg')
        return Image.open(image_path)

## Download Oxford Pet Dataset
def get_oxford_pet_dataset(batchsize, install=True):
    def extract_tar_gz(file_path, extract_path):
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)

    if install:
        os.mkdir(f'{HOME}/DATA')
        DATA_PATH = os.path.join(HOME, 'DATA')

        if "oxford_pets" in os.listdir(DATA_PATH):
            print("Dataset already exists")
        else:
            os.mkdir(os.path.join(DATA_PATH, 'oxford_pets'))
            os.chdir(os.path.join(DATA_PATH, 'oxford_pets'))
            print("Downloading the data...")
            images = wget.download('https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
            annotation = wget.download('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
            print("Dataset downloaded!")
            print("Extracting data..")
            extract_tar_gz(images, os.path.join(DATA_PATH, 'oxford_pets'))
            extract_tar_gz(annotation, os.path.join(DATA_PATH, 'oxford_pets'))
            print("Extraction done!")

    file_path = os.path.join(HOME, 'DATA', 'oxford_pets', 'annotations', 'test.txt')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [(line.split(' ')[0], line.split(' ')[1]) for line in lines]
        pets_class_mapper = {}
        for line in lines:
            if line[1] not in pets_class_mapper:
                cls = line[0].replace('_',' ')
                pets_class_mapper[int(line[1])] = ''.join([i for i in cls if not i.isdigit()])

    ANN_DIR = os.path.join(HOME, 'DATA', 'oxford_pets')

    #Build datasets
    pets_train_data = OxfordPetsDataset(ANN_DIR, split='trainval', transform=train_transforms)
    pets_test_data = OxfordPetsDataset(ANN_DIR, split='test', transform=test_transforms)

    #Build dataloaders
    pets_train_loader = DataLoader(pets_train_data, batchsize, shuffle=True, drop_last=True)
    pets_test_loader = DataLoader(pets_test_data, batchsize, shuffle=True, drop_last=True)

    return pets_train_loader, pets_test_loader, pets_class_mapper

## Download Food 101 Dataset
def get_food_101_dataset(batchsize, install=True):
    Food101_path = os.path.join(HOME, 'DATA')
    food_train_data = Food101(Food101_path, split='train', download=install)
    food_test_data = Food101(Food101_path, split='test', download=install)

    food101_train_path = os.path.join(food_train_data.root, 'food-101', 'images')
    food101_test_path = os.path.join(food_test_data.root, 'food-101', 'images')

    food_train_data = datasets.ImageFolder(root=food101_train_path, transform=train_transforms, target_transform=None) 
    food_test_data = datasets.ImageFolder(root=food101_test_path, transform=test_transforms)
    food_class_mapper = food_train_data.class_to_idx

    #Build dataloaders
    food_train_loader = DataLoader(food_train_data, batchsize, shuffle=True, drop_last=True)
    food_test_loader = DataLoader(food_test_data, batchsize, shuffle=False, drop_last=True)

    return food_train_loader, food_test_loader, food_class_mapper

## Download Cifar100 Dataset
def get_Cifar_100_dataset(batchsize, install=True):
    cifar_path = os.path.join(HOME, 'DATA')
    cifar_train_data = CIFAR100(cifar_path, train=True, transform=train_transforms, download=install)
    cifar_test_data = CIFAR100(cifar_path, train=False, transform=test_transforms, download=install)

    #Get the class mapper
    cifar_class_mapper = cifar_train_data.class_to_idx

    #Build dataloaders
    cifar_train_loader = DataLoader(cifar_train_data, batchsize, shuffle=True, drop_last=True)
    cifar_test_loader = DataLoader(cifar_test_data, batchsize, shuffle=False, drop_last=True)

    return cifar_train_loader, cifar_test_loader, cifar_class_mapper

## Download Cifar10 Dataset
def get_Cifar_10_dataset(batchsize, install=True):
    cifar10_path = os.path.join(HOME, 'DATA')
    cifar10_train_data = CIFAR10(cifar10_path, train=True, transform=train_transforms, download=install)
    cifar10_test_data = CIFAR10(cifar10_path, train=False, transform=test_transforms, download=install)

    #Get the class mapper
    cifar10_class_mapper = cifar10_train_data.class_to_idx

    #Build dataloaders
    cifar10_train_loader = DataLoader(cifar10_train_data, batchsize, shuffle=True, drop_last=True)
    cifar10_test_loader = DataLoader(cifar10_test_data, batchsize, shuffle=False, drop_last=True)

    return cifar10_train_loader, cifar10_test_loader, cifar10_class_mapper

## Download MNIST Dataset
def get_MNIST_dataset(batchsize, install=True):
    mnist_path = os.path.join(HOME, 'DATA')
    mnist_train_data = MNIST(mnist_path, train=True, transform=train_transforms, download=install)
    mnist_test_data = MNIST(mnist_path, train=False, transform=test_transforms, download=install)

    #Get the class mapper
    mnist_class_mapper = mnist_train_data.class_to_idx

    #Build dataloaders
    mnist_train_loader = DataLoader(mnist_train_data, batchsize, shuffle=True, drop_last=True)
    mnist_test_loader = DataLoader(mnist_test_data, batchsize, shuffle=False, drop_last=True)

    return mnist_train_loader, mnist_test_loader, mnist_class_mapper
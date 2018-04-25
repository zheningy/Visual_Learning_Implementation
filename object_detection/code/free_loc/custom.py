import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index

    classes = imdb.classes
    class_to_idx = {}
    for item in classes:
        class_to_idx[item] = classes.index(item)

    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset
    images = []
    gt_classes = imdb.gt_roidb()
    for idx in range(imdb.num_images):
        gt_class = gt_classes[idx]['gt_classes']
        curt_image = (imdb.image_path_at(idx), gt_class.tolist())
        images.append(curt_image)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.features = nn.Sequential(
                            nn.Conv2d(3, 64, 11, stride=4, padding=2),
                            nn.ReLU(),
                            nn.MaxPool2d(3, stride=2),
                            nn.Conv2d(64, 192, 5, stride=1, padding=2),
                            nn.ReLU(),
                            nn.MaxPool2d(3, stride=2),
                            nn.Conv2d(192, 384, 3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(384, 256, 3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(256, 256, 3, stride=1, padding=1),
                            nn.ReLU()
            )

        self.classifier = nn.Sequential(
                        nn.Conv2d(256, 256, 3, stride=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, 1, stride=1),
                        nn.ReLU(),
                        nn.Conv2d(256, num_classes, 1, stride=1)
            )

    def forward(self, x):
        #TODO: Define forward pass
        features = self.features(x)
        return self.classifier(features)



class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetHighres, self).__init__()
        #TODO: Ignore for now until instructed



    def forward(self, x):
        #TODO: Ignore for now until instructed


        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whether it is pretrained or
    #not

    if pretrained == True:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        features_dict = {k: v for k, v in pretrained_dict.items() if 'features' in k}
        model_dict = model.state_dict()
        model_dict.update(features_dict)
        model.load_state_dict(model_dict) 




    return model

def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed






    return model



class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write the rest of this function

    images = []
    gt_classes = imdb.gt_roidb()
    for idx in range(imdb.num_images):
        gt_class = gt_classes[idx]['gt_classes']
        curt_image = (imdb.image_path_at(idx), gt_class.tolist())
        images.append(curt_image)

    return images

        img = imgs[index][0]
        target 


        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

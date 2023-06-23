import os
import cv2
import numpy as np
import torch
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class Fusion_Datasets(Dataset):
    """docstring for Fusion_Datasets"""

    def __init__(self, configs, transform=None):
        super(Fusion_Datasets, self).__init__()
        self.phase = configs['phase']
        self.patch_size = configs['patch_size'] if configs['phase'] == 'train' else None
        self.root_dir = configs['root_dir']
        self.transform = transform
        self.channels = configs['channels']
        self.sensors = configs['sensors']
        self.img_list = {i: os.listdir(os.path.join(self.root_dir, i)) for i in self.sensors}
        self.img_path = {i: [os.path.join(self.root_dir, i, j) for j in os.listdir(os.path.join(self.root_dir, i))]
                         for i in self.sensors}
        self.transform1 = transforms.Compose([self.transform, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_data = {}
        for i in self.sensors:
            img = Image.open(self.img_path[i][index])
            # print(self.img_path[i][index])
            if self.channels == 1:
                img = img.convert('L')
            elif self.channels == 3:
                img = img.convert('RGB')
            img_data.update({i: img})

        if self.phase == 'train':
            far, near, fm = np.asarray(img_data[self.sensors[0]]),np.asarray(img_data[self.sensors[1]]),np.asarray(img_data[self.sensors[2]])
            gt = np.asarray(img_data[self.sensors[3]])
            # H, W = far.shape[0], far.shape[1]
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            # rnd_h = random.randint(0, max(0, H - self.patch_size))
            # rnd_w = random.randint(0, max(0, W - self.patch_size))
            # patch_A = far[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # patch_B = near[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # patch_GT = gt[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            # patch_fm = fm[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0,7)
            patch_GT = augment_img(gt, mode=mode)
            patch_GT = Image.fromarray(np.uint8(patch_GT)).convert('RGB')
            patch_A, patch_B, patch_fm = augment_img(far, mode=mode), augment_img(near, mode=mode), augment_img(fm, mode=mode)
            patch_A, patch_B, patch_fm = Image.fromarray(np.uint8(patch_A)).convert('RGB'), Image.fromarray(np.uint8(patch_B)).convert('RGB'), Image.fromarray(np.uint8(patch_fm)).convert('RGB')
            if self.transform is not None:
                patch_A, patch_B, patch_fm = self.transform1(patch_A), self.transform1(patch_B), self.transform(patch_fm)
                patch_GT = self.transform(patch_GT)
            img_data.update({self.sensors[0]: patch_A, self.sensors[1]: patch_B, self.sensors[2]: patch_fm, self.sensors[3]: patch_GT})  #
        else:
            for i in self.sensors:
                if self.transform1 is not None:
                    img_data.update({i: self.transform1(img_data[i])})
        return img_data

    def __len__(self):
        img_num = [len(self.img_list[i]) for i in self.img_list]
        img_counter = Counter(img_num)
        assert len(img_counter) == 1, 'Sensors Has Different length'
        return img_num[0]


if __name__ == '__main__':
    datasets = Fusion_Datasets(configs={'phase':'train','patch_size':64,'root_dir':'D://anaconda//envs//pytorch//PaperNotebook//Pytorch_Image_Fusion-main//datasets//Dataset//', 'sensors':['Far', 'Near', 'focus_map', 'GT'],'channels':3},
                               transform=transforms.Compose([transforms.CenterCrop((256,256)), transforms.Resize((256,256)), transforms.ToTensor()]))
    train = DataLoader(datasets, 1, True)
    print(len(train))
    for i, data in enumerate(train):
        if i == 0:
            print(data['Near'].shape)
            print(data['Far'].shape)
            print(data['focus_map'].shape)
            print(data['GT'].shape)


        # img = data['Vis'][0].permute(1,2,0)
        # cv2.imshow('1',img.numpy())
        # img1 = data['Inf'][0].permute(1,2,0)
        # cv2.imshow('2',img1.numpy())
        # cv2.waitKey()
        # cv2.destroyAllWindows()

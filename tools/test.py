import cv2
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from core.util import post_remove_small_objects
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
def merge(R,G,B):
    '''
    r,g,b:RGB channal
    return : merge three channals
    '''
    # R = np.expand_dims(R, axis=-1)
    # G = np.expand_dims(G, axis=-1)
    # B = np.expand_dims(B, axis=-1)
    return np.concatenate([R, G, B], axis=1)

def testintrain(model, fusion_datasets, configs, load_weight_path, epoch, save_path=None):
    with torch.no_grad():
        model.eval()
        if load_weight_path:
            assert configs['TEST']['weight_path'] != 'None', 'Test Need To Resume Chekpoint'
            weight_path = load_weight_path
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model'].state_dict())
        is_use_gpu = torch.cuda.is_available()
        test_dataloader = DataLoader(fusion_datasets, batch_size=configs['TEST']['batch_size'], shuffle=False)
        test_num_iter = len(test_dataloader)
        dtransforms = transforms.Compose([transforms.ToPILImage()])
        far_path = r'.\datasets\test\Far\\'
        near_path = r'.\datasets\test\Near\\'
        fars = os.listdir(far_path)
        nears = os.listdir(near_path)
        with tqdm(total=test_num_iter) as test_bar:
            for iter, data in enumerate(test_dataloader):
                if is_use_gpu:
                    model = model.cuda()
                    data = {sensor: data[sensor].cuda() for sensor in data}
                coarsemaps, finemaps = model(data)
                coarsemaps = [i.sigmoid() for i in coarsemaps]
                finemaps = [i.sigmoid() for i in finemaps]
                test_bar.set_description(
                    'Epoch: {}/{}. TEST. Iter: {}/{}.'.format(
                        epoch, configs['TRAIN']['max_epoch'], iter + 1, test_num_iter))
                unnorm = transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5),
                                                                      (1 / 0.5, 1 / 0.5, 1 / 0.5))
                for sensor in data:
                    if sensor != 'focus_map':
                        data.update({sensor: unnorm(data[sensor])})
                    else:
                        data.update({sensor: data[sensor]})
                # final focus map
                segmapssb = finemaps[0]
                # binary segmentation
                for i in segmapssb:
                    i[i > 0.5], i[i <= 0.5] = 1, 0
                for batch in range(segmapssb.shape[0]):
                    if save_path is None:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/')
                        save_path = configs['TEST']['save_path']+str(epoch)+'/'
                    if iter < 9:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/init/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/init/')
                        name_init = os.path.join(save_path+'init/', 'lytro-0'+str(len(os.listdir(save_path+'init/'))+1))
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/final/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/final/')
                        name_final = os.path.join(save_path+'final/', 'lytro-0'+str(len(os.listdir(save_path+'final/'))+1))
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/fused/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/fused/')
                        name_fused = os.path.join(save_path+'fused/', 'lytro-0'+str(len(os.listdir(save_path+'fused/'))+1))
                    else:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/init/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/init/')
                        name_init = os.path.join(save_path+'init/', 'lytro-'+str(len(os.listdir(save_path+'init/'))+1))
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/final/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/final/')
                        name_final = os.path.join(save_path+'final/', 'lytro-'+str(len(os.listdir(save_path+'final/'))+1))
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/fused/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/fused/')
                        name_fused = os.path.join(save_path+'fused/', 'lytro-'+str(len(os.listdir(save_path+'fused/'))+1))
                    segmapf = segmapssb[batch]
                    segmapf = post_remove_small_objects(segmapf, iter).transpose(1, 2, 0)
                    # fuse resume
                    far = cv2.imread(far_path+fars[iter])
                    near = cv2.imread(near_path+nears[iter])
                    img = segmapf * far + (1 - segmapf) * near
                    # fused_img
                    cv2.imwrite(f'{name_fused}.jpg', img)
                    cv2.imwrite(f'{name_final}.png', segmapf*255)
                    # coarse maps and bounds
                    for i in range(len(coarsemaps)):
                        focusmap = coarsemaps[i]
                        segmap0 = dtransforms(focusmap[batch].to(dtype=torch.float32))
                        segmap0.save(f'{name_init}0'+str(i)+'.png')
                    # fine maps and bounds
                    for i in range(len(finemaps)):
                        boundary = finemaps[i]
                        boundary0 = dtransforms(boundary[batch].to(dtype=torch.float32))
                        boundary0.save(f'{name_init}1'+str(i)+'.png')
                test_bar.update(1)

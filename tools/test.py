import cv2
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from core.util import post_remove_small_objects
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from core.model import GaussianDiffusion3
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

def test(model, fusion_datasets, configs, load_weight_path=False, save_path=None):
    with torch.no_grad():
        model.eval()
        if load_weight_path:
            assert configs['TEST']['weight_path'] != 'None', 'Test Need To Resume Chekpoint'
            weight_path = configs['TEST']['weight_path']
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model'].state_dict())
        is_use_gpu = torch.cuda.is_available()
        test_dataloader = DataLoader(fusion_datasets, batch_size=configs['TEST']['batch_size'], shuffle=False)
        dtransforms = transforms.Compose([transforms.ToPILImage()])
        far_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Far\\'
        near_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Near\\'
        fars = os.listdir(far_path)
        nears = os.listdir(near_path)
        test_num_iter = len(test_dataloader)
        diffusion1 = GaussianDiffusion3(configs['TEST']['batch_size']).eval()
        with tqdm(total=test_num_iter) as test_bar:
            for iter, data in enumerate(test_dataloader):
                if is_use_gpu:
                    model = model.cuda()
                    data = {sensor: data[sensor].cuda() for sensor in data}
                f_noise, noise, t, n_noise = add_noise(data, diffusion1)
                focusmaps = model(data, f_noise, n_noise, t)
                focusmaps = focusmaps.sigmoid()
                f_noise = (f_noise + 1) * 0.5
                n_noise = (n_noise + 1) * 0.5
                focusmaps_ = focusmaps
                test_bar.set_description(
                    'TEST. Iter: {}/{}.'.format(iter + 1, test_num_iter))
                # final focus map
                # binary segmentation
                focusmaps[focusmaps > 0.5], focusmaps[focusmaps <= 0.5] = 1, 0
                for batch in range(focusmaps.shape[0]):
                    if save_path is None:
                        if not os.path.exists(configs['TEST']['save_path'] + '/'):
                            os.mkdir(configs['TEST']['save_path'] + '/')
                        save_path = configs['TEST']['save_path'] + '/'
                    if iter < 9:
                        if not os.path.exists(configs['TEST']['save_path'] + '/init/'):
                            os.mkdir(configs['TEST']['save_path'] + '/init/')
                        name_init = os.path.join(save_path + 'init/',
                                                 'lytro-0' + str(len(os.listdir(save_path + 'init/')) + 1))
                        if not os.path.exists(configs['TEST']['save_path'] + '/final/'):
                            os.mkdir(configs['TEST']['save_path'] + '/final/')
                        name_final = os.path.join(save_path + 'final/',
                                                  'lytro-0' + str(len(os.listdir(save_path + 'final/')) + 1))
                        if not os.path.exists(configs['TEST']['save_path'] + '/fused/'):
                            os.mkdir(configs['TEST']['save_path'] + '/fused/')
                        name_fused = save_path+'fused/'
                    else:
                        if not os.path.exists(configs['TEST']['save_path'] + '/init/'):
                            os.mkdir(configs['TEST']['save_path'] + '/init/')
                        name_init = os.path.join(save_path + 'init/',
                                                 'lytro-' + str(len(os.listdir(save_path + 'init/')) + 1))
                        if not os.path.exists(configs['TEST']['save_path'] + '/final/'):
                            os.mkdir(configs['TEST']['save_path'] + '/final/')
                        name_final = os.path.join(save_path + 'final/',
                                                  'lytro-' + str(len(os.listdir(save_path + 'final/')) + 1))
                        if not os.path.exists(configs['TEST']['save_path'] + '/fused/'):
                            os.mkdir(configs['TEST']['save_path'] + '/fused/')
                            name_fused = save_path + 'fused/'
                    segmapf = focusmaps[batch]
                    segmapf = post_remove_small_objects(segmapf, iter).transpose(1, 2, 0)
                    segmapf_b = cv2.GaussianBlur(segmapf * 255, ksize=[7, 7], sigmaX=2)
                    segmapf_b = np.expand_dims(segmapf_b, axis=-1) / 255
                    # segmapf_b1 = cv2.GaussianBlur((1 - segmapf) * 255, ksize=[7, 7], sigmaX=2)
                    # segmapf_b1 = np.expand_dims(segmapf_b1.astype(np.double) / 255, axis=-1)
                    # print(segmapf_b.shape)
                    # fuse resume
                    far = cv2.imread(far_path + fars[iter])
                    near = cv2.imread(near_path + nears[iter])
                    img = segmapf_b * far.astype(np.double) + (1 - segmapf_b) * near.astype(np.double)
                    # fused_img
                    # img = dtransforms(img.squeeze(0).to(dtype=torch.float32))
                    # cv2.imwrite(f'{name_fused}.png', img)
                    # cv2.imwrite(f'{name_fused}.jpg', img)
                    img = img.astype(np.uint8)
                    d = img
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if iter < 20:
                        if iter < 9:
                            img.save(name_fused + 'Lytro-0' + str(iter + 1) + '.jpg', quality=75)
                        else:
                            img.save(name_fused + 'Lytro-' + str(iter + 1) + '.jpg', quality=75)
                    elif iter >= 20 and iter < 33:
                        if iter < 29:
                            # img.save(name_fused + 'MFFW-0' + str(iter + 1 - 20) + '.jpg', quality=75)
                            img.save(name_fused + 'MFFW-0' + str(iter + 1 - 20) + '.png', quality=75)
                            cv2.imwrite(name_fused + 'MFFW-0' + str(iter + 1 - 20) + '.jpg', d)
                        else:
                            # img.save(name_fused + 'MFFW-' + str(iter + 1 - 20) + '.jpg', quality=75)
                            img.save(name_fused + 'MFFW-' + str(iter + 1 - 20) + '.png', quality=75)
                            cv2.imwrite(name_fused + 'MFFW-' + str(iter + 1 - 20) + '.jpg', d)

                    else:
                        if iter < 42:
                            img.save(name_fused + 'MFI-WHU-0' + str(iter + 1 - 33) + '.jpg', quality=75)
                        else:
                            img.save(name_fused + 'MFI-WHU-' + str(iter + 1 - 33) + '.jpg', quality=75)
                    # segmapf = dtransforms(segmapf.to(dtype=torch.float32))
                    cv2.imwrite(f'{name_final}.png', segmapf * 255)
                    # segmapf = dtransforms(segmapf.to(dtype=torch.float32))
                    cv2.imwrite(f'{name_final}_b.png', segmapf_b * 255)
                    # noise images
                    focusmap_ = dtransforms(focusmaps_[batch])
                    focusmap_.save(f'{name_init}' + '.png')
                    noiseimg = dtransforms(f_noise[batch])
                    noiseimg.save(f'{name_init}0' + '.png')
                    noiseimg = dtransforms(n_noise[batch])
                    noiseimg.save(f'{name_init}1' + '.png')
                    # noiseimg = dtransforms(noisefar[batch])
                    # noiseimg.save(f'{name_init}2' + '.png')
                    # noiseimg = dtransforms(noisenear[batch])
                    # noiseimg.save(f'{name_init}3' + '.png')
                torch.cuda.empty_cache()
                test_bar.update(1)


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
        far_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Far\\'
        near_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Near\\'
        fars = os.listdir(far_path)
        nears = os.listdir(near_path)
        with tqdm(total=test_num_iter) as test_bar:
            for iter, data in enumerate(test_dataloader):
                if is_use_gpu:
                    model = model.cuda()
                    data = {sensor: data[sensor].cuda() for sensor in data}
                # fusion_image = []
                # for i in range(configs['TEST_DATASET']['channels']):
                #     fusion_image.append(model({'Far':data['Far'][:,i,:,:].unsqueeze(0),'Near':data['Near'][:,i,:,:].unsqueeze(0)}))
                coarsemaps, finemaps = model(data)
                coarsemaps = [i.sigmoid() for i in coarsemaps]
                finemaps = [i.sigmoid() for i in finemaps]
                # segmap = [torch.argmax(i, dim=1).unsqueeze(1) for i in segmap]
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
                # small object removal
                # segmapssb = remove_small_objects(segmapssb)
                # fused = torch.cat([fusion_image[0],fusion_image[1],fusion_image[2]],dim=1)
                # input_imgs, fusion_imgs = debug(configs['MODEL'], configs['TEST_DATASET'], data, fused)
                # input_imgs = [input_imgs[sensor] for sensor in configs['MODEL']['input_sensors']]
                # imgs = input_imgs + [fusion_imgs]
                # imgs = torch.cat(imgs, dim=3)
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
                    # img = dtransforms(img.squeeze(0).to(dtype=torch.float32))
                    cv2.imwrite(f'{name_fused}.png', img)
                    cv2.imwrite(f'{name_fused}.jpg', img)
                    # segmapf = dtransforms(segmapf.to(dtype=torch.float32))
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

def add_noise(data, diffusion):
    with torch.no_grad():
        f_noise, noise, t, n_noise = diffusion(data['Far'], data['Near'])
        return f_noise, noise, t, n_noise

def testintrain2(model, diffusion, test_dataloader, configs, load_weight_path, epoch, save_path=None):
    with torch.no_grad():
        model.eval()
        if load_weight_path:
            assert configs['TEST']['weight_path'] != 'None', 'Test Need To Resume Chekpoint'
            weight_path = load_weight_path
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model'].state_dict())
        is_use_gpu = torch.cuda.is_available()
        test_num_iter = len(test_dataloader)
        dtransforms = transforms.Compose([transforms.ToPILImage()])
        far_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Far\\'
        near_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Near\\'
        fars = os.listdir(far_path)
        nears = os.listdir(near_path)
        with tqdm(total=test_num_iter) as test_bar:
            for iter, data in enumerate(test_dataloader):
                if is_use_gpu:
                    model = model.cuda()
                    data = {sensor: data[sensor].cuda() for sensor in data}
                f_noise, noise, t, n_noise = add_noise(data, diffusion)
                focusmaps = model(data, f_noise, n_noise, t)
                focusmaps = focusmaps.sigmoid()
                f_noise = (f_noise + 1) * 0.5
                n_noise = (n_noise + 1) * 0.5
                focusmaps_ = focusmaps
                test_bar.set_description(
                    'Epoch: {}/{}. TEST. Iter: {}/{}.'.format(
                        epoch, configs['TRAIN']['max_epoch'], iter + 1, test_num_iter))
                # final focus map
                # binary segmentation
                focusmaps[focusmaps > 0.5], focusmaps[focusmaps <= 0.5] = 1, 0
                for batch in range(focusmaps.shape[0]):
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
                        name_fused = save_path+'fused/'
                    else:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/init/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/init/')
                        name_init = os.path.join(save_path+'init/', 'lytro-'+str(len(os.listdir(save_path+'init/'))+1))
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/final/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/final/')
                        name_final = os.path.join(save_path+'final/', 'lytro-'+str(len(os.listdir(save_path+'final/'))+1))
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/fused/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/fused/')
                        name_fused = save_path+'fused/'
                    segmapf = focusmaps[batch]
                    segmapf = post_remove_small_objects(segmapf, iter).transpose(1, 2, 0)
                    segmapf_b = cv2.GaussianBlur(segmapf*255, ksize=[7, 7], sigmaX=2)
                    segmapf_b = np.expand_dims(segmapf_b, axis=-1) / 255
                    # segmapf_b1 = cv2.GaussianBlur((1 - segmapf) * 255, ksize=[7, 7], sigmaX=2)
                    # segmapf_b1 = np.expand_dims(segmapf_b1.astype(np.double) / 255, axis=-1)
                    # print(segmapf_b.shape)
                    # fuse resume
                    far = cv2.imread(far_path+fars[iter])
                    near = cv2.imread(near_path+nears[iter])
                    img = segmapf_b * far.astype(np.double) + (1 - segmapf_b) * near.astype(np.double)
                    # fused_img
                    # img = dtransforms(img.squeeze(0).to(dtype=torch.float32))
                    # cv2.imwrite(f'{name_fused}.png', img)
                    # cv2.imwrite(f'{name_fused}.jpg', img)
                    img = img.astype(np.uint8)
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if iter < 20:
                        if iter < 9:
                            img.save(name_fused+'Lytro-0' + str(iter + 1) + '.jpg', quality=75)
                        else:
                            img.save(name_fused+'Lytro-' + str(iter + 1) + '.jpg', quality=75)
                    elif iter >= 20 and iter < 33:
                        if iter < 29:
                            img.save(name_fused+'MFFW-0' + str(iter + 1 - 20) + '.jpg', quality=75)
                        else:
                            img.save(name_fused+'MFFW-' + str(iter + 1 - 20) + '.jpg', quality=75)
                    else:
                        if iter < 42:
                            img.save(name_fused+'MFI-WHU-0' + str(iter + 1 - 33) + '.jpg', quality=75)
                        else:
                            img.save(name_fused+'MFI-WHU-' + str(iter + 1 - 33) + '.jpg', quality=75)
                    # segmapf = dtransforms(segmapf.to(dtype=torch.float32))
                    cv2.imwrite(f'{name_final}.png', segmapf*255)
                    # segmapf = dtransforms(segmapf.to(dtype=torch.float32))
                    cv2.imwrite(f'{name_final}_b.png', segmapf_b * 255)
                    # noise images
                    focusmap_ = dtransforms(focusmaps_[batch])
                    focusmap_.save(f'{name_init}' + '.png')
                    noiseimg = dtransforms(f_noise[batch])
                    noiseimg.save(f'{name_init}0' + '.png')
                    noiseimg = dtransforms(n_noise[batch])
                    noiseimg.save(f'{name_init}1' + '.png')
                    # noiseimg = dtransforms(noisefar[batch])
                    # noiseimg.save(f'{name_init}2' + '.png')
                    # noiseimg = dtransforms(noisenear[batch])
                    # noiseimg.save(f'{name_init}3' + '.png')
                torch.cuda.empty_cache()
                test_bar.update(1)
def add_noise1(data, diffusion):
    with torch.no_grad():
        f_noise, noise, t = diffusion(data['Far'])
        return f_noise, noise, t

def testintrain3(model, diffusion, fusion_datasets, configs, load_weight_path, epoch, save_path=None):
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
        with tqdm(total=test_num_iter) as test_bar:
            for iter, data in enumerate(test_dataloader):
                if is_use_gpu:
                    model = model.cuda()
                    data = {sensor: data[sensor].cuda() for sensor in data}
                noise_, noise, t = add_noise1(data, diffusion)
                pred_noise = model(noise_, t)
                noise_ = noise_
                noise_ = (noise_ + 1) * 0.5
                test_bar.set_description(
                    'Epoch: {}/{}. TEST. Iter: {}/{}.'.format(
                        epoch, configs['TRAIN']['max_epoch'], iter + 1, test_num_iter))
                # unnorm = transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5),
                #                               (1 / 0.5, 1 / 0.5, 1 / 0.5))
                # noiseimgs = unnorm(noiseimgs)
                # for sensor in data:
                #     if sensor != 'focus_map':
                #         data.update({sensor: unnorm(data[sensor])})
                #     else:
                #         data.update({sensor: data[sensor]})
                # final focus map
                # binary segmentation
                for batch in range(pred_noise.shape[0]):
                    if save_path is None:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/')
                        save_path = configs['TEST']['save_path']+str(epoch)+'/'
                    if iter < 9:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/init/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/init/')
                        name_init = os.path.join(save_path+'init/', 'lytro-0'+str(len(os.listdir(save_path+'init/'))+1))
                    else:
                        if not os.path.exists(configs['TEST']['save_path']+str(epoch)+'/init/'):
                            os.mkdir(configs['TEST']['save_path']+str(epoch)+'/init/')
                        name_init = os.path.join(save_path+'init/', 'lytro-'+str(len(os.listdir(save_path+'init/'))+1))
                    # noise images
                    # noiseimg = dtransforms(pred_noise[batch])
                    # noiseimg.save(f'{name_init}' + '.png')
                    noiseimg = dtransforms(noise_[batch])
                    noiseimg.save(f'{name_init}0' + '.png')
                test_bar.update(1)

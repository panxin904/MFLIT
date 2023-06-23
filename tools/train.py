import torch
import torchvision
from tqdm import tqdm
import os
from core.loss import Loss_CFM, Loss_Bound, Loss_FM, Loss_Patch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
from core.util import debug1, build_scheduler, debug_segmap_, debug2, debug_segmap
from tools.test import testintrain

def train(model, train_datasets, test_datasets, configs):
    if not os.path.exists(os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'])):
        os.mkdir(os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name']))
    # weight_path = r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\work_dirs\MFLIT1\MFNeXt\model_100.pth'
    # checkpoint = torch.load(weight_path)
    # model.load_state_dict(checkpoint['model'].state_dict())
    model.train()
    '''define visual tools(Tensorboard)'''
    train_writer = SummaryWriter(log_dir=os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name']))

    if configs['TRAIN']['resume'] == 'None':
        start_epoch = 1
    else:
        start_epoch = torch.load(configs['TRAIN']['resume'])['epoch'] + 1
    '''to cuda'''
    is_use_gpu = torch.cuda.is_available()
    '''split train set and test data'''
    train_size = int(len(train_datasets)-20)
    val_size = int(20)
    train_dataset, val_dataset = torch.utils.data.random_split(train_datasets, [train_size, val_size])
    '''define dataloader for training'''
    train_dataloader = DataLoader(train_dataset, batch_size=configs['TRAIN']['batch_size'], shuffle=True, drop_last=True, num_workers=8)
    train_num_iter = len(train_dataloader)
    # '''define dataloader for validating'''
    val_dataloader = DataLoader(val_dataset, batch_size=configs['TRAIN']['batch_size'], shuffle=True, drop_last=True, num_workers=2)
    val_num_iter = len(val_dataloader)
    '''define optimizer'''
    if configs['TRAIN']['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), eps=configs['TRAIN']['EPS'], betas=configs['TRAIN']['BETAS'],
                    lr=configs['TRAIN']['lr'], weight_decay=configs['TRAIN']['weight_decay'])  #
        '''define scheduler,  "CosineStepLR"'''
        scheduler = build_scheduler(configs, optimizer=optimizer, n_iter_per_epoch=train_num_iter)
    else:
        optimizer = eval('torch.optim.' + configs['TRAIN']['opt'])(model.parameters(), lr=configs['TRAIN']['lr'], weight_decay=configs['TRAIN']['weight_decay'])
        '''define scheduler,  "MultiStepLR"'''
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs['TRAIN']['milestones'],
                                                     gamma=configs['TRAIN']['gamma'])
    '''load loss function from config'''
    loss_func = [eval(l)() for l in configs['TRAIN']['loss_func']]
    train_iter = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    val_iter = 0
    '''Training'''
    for epoch in range(start_epoch, configs['TRAIN']['max_epoch'] + 1):
        loss_epoch = 0
        loss_bound_epoch, loss_patch_epoch, loss_cfm_epoch, loss_fm_epoch = 0, 0, 0, 0
        # train_dataset, val_dataset = torch.utils.data.random_split(train_datasets, [int(len(train_datasets) * 0.875), int(len(train_datasets) * 0.125)])
        with tqdm(total=train_num_iter) as train_bar:
            for iter, data in enumerate(train_dataloader):
                if is_use_gpu:
                    model = model.cuda()
                    data = {sensor: data[sensor].cuda() for sensor in data}
                coarsemaps, finemaps = model(data)
                coarsemaps = [i.sigmoid() for i in coarsemaps]
                finemaps = [i.sigmoid() for i in finemaps]
                '''compute loss'''
                loss_cfm = loss_func[0](coarsemaps[0], data['focus_map'], epoch) * configs['TRAIN']['loss_weights'][0]
                loss_fm = loss_func[1](finemaps[0], data['focus_map'], epoch) * configs['TRAIN']['loss_weights'][1]
                loss_bound = loss_func[2](finemaps[1], data['Boundary'], epoch) * configs['TRAIN']['loss_weights'][2]
                # loss_patch = loss_func[3](finemaps, data['focus_map'], epoch) * configs['TRAIN']['loss_weights'][3]
                # if loss_fm < 0.1 * loss_bound:
                #     loss_bound *= 0.1
                # train_writer.add_scalars('Rate Local and Global', {'Global': model.proj_f.rate1, 'Local': model.proj_f.rate2}, global_step=train_iter)
                '''compute all loss'''
                loss_batch = loss_cfm + loss_fm + loss_bound #+ loss_patch
                '''optimize parameters'''
                loss_bound_epoch += loss_bound.item()
                loss_cfm_epoch += loss_cfm.item()
                # loss_patch_epoch += loss_patch.item()
                loss_fm_epoch += loss_fm.item()
                loss_epoch += loss_batch.item()
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
                if configs['TRAIN']['opt'] == 'AdamW':
                    scheduler.step_update(epoch*train_num_iter+iter)
                '''total loss->tensorboard ; print information'''
                train_writer.add_scalar('Train Iter Loss', loss_batch, global_step=train_iter)
                train_bar.set_description(
                    'Epoch: {}/{}. TRAIN. Iter: {}/{}. Content Loss: {:.5f}  All loss: {:.5f} LR: {:.10f}'.format(
                        epoch, configs['TRAIN']['max_epoch'], iter + 1, train_num_iter, loss_batch, loss_epoch / train_num_iter, optimizer.param_groups[0]['lr']))

                if configs['TRAIN']['debug_interval'] is not None and train_iter % configs['TRAIN']['debug_interval'] == 0:
                    """反归一化 将归一化的输入数据反归一化后方便测试和展示"""
                    unnorm = torchvision.transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5),
                                                                      (1 / 0.5, 1 / 0.5, 1 / 0.5))
                    for sensor in data:
                        if sensor == 'focus_map' or sensor == 'Boundary' or sensor == 'coarse_fm' or sensor == 'coarse_bound':
                            data.update({sensor: data[sensor]})
                        else:
                            data.update({sensor: unnorm(data[sensor])})
                    # input_imgs, fusion_imgs = debug(configs['MODEL'], configs['TRAIN_DATASET'], data, fusion_image)
                    input_imgs1 = debug1(configs['MODEL'], configs['TRAIN_DATASET'], data)
                    input_imgs1 = [input_imgs1[sensor] for sensor in configs['MODEL']['input_sensors'][0:2]]
                    input_imgs2 = debug2(configs['MODEL'], configs['TRAIN_DATASET'], data)
                    input_imgs2 = [input_imgs2[sensor] for sensor in configs['MODEL']['input_sensors'][2:4]]
                    # input_imgs3 = debug3(configs['MODEL'], configs['TRAIN_DATASET'], data)
                    # input_imgs3 = [input_imgs3[sensor] for sensor in configs['MODEL']['input_sensors'][4:]]
                    train_writer.add_image('Far and Near', torch.cat(input_imgs1, dim=2), train_iter, dataformats='NCHW')
                    train_writer.add_image('focus_map and Boundary', torch.cat(input_imgs2, dim=2), train_iter, dataformats='NCHW')
                    train_writer.add_image('coarse maps', torch.cat(debug_segmap(coarsemaps), dim=2), train_iter, dataformats='NCHW')
                    train_writer.add_image('fine maps and bounds', torch.cat(debug_segmap_(finemaps), dim=2), train_iter, dataformats='NCHW')
                train_iter += 1
                train_bar.update(1)
            if configs['TRAIN']['opt'] != 'AdamW':
                scheduler.step()
            '''lr->tensorboard'''
            train_writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
            if configs['TRAIN']['val_interval'] is not None and epoch in eval(configs['TRAIN']['val_interval']):
                torch.save({'model': model, 'epoch': epoch},
                           os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'],
                                        f'model_{epoch}.pth'))
            if configs['TRAIN']['valid_interval'] is not None and epoch in eval(configs['TRAIN']['valid_interval']):
                    testintrain(model, test_datasets, configs, epoch=epoch,
                                load_weight_path=os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'], f'model_{epoch}.pth'))
            '''loss->tensorboard'''
            train_writer.add_scalars('Train Loss', {'Coarse FM Loss': loss_cfm_epoch / train_num_iter,
                                                    # 'Patch FM Loss': loss_patch_epoch / train_num_iter,
                                                    'Fine FM Loss': loss_fm_epoch / train_num_iter,
                                                    'Fine Bound Detect Loss': loss_bound_epoch / train_num_iter
                                                    },
                                         global_step=epoch)
            train_writer.add_scalar('Train Loss epoch', loss_epoch / train_num_iter, global_step=epoch)
        # if configs['TRAIN']['valid_interval'] is not None and epoch in eval(configs['TRAIN']['valid_interval']):
        #     val_loss_epoch = 0
        #     with tqdm(total=val_num_iter) as val_bar:
        #         model.eval()
        #         with torch.no_grad():
        #             for iter, data in enumerate(val_dataloader):
        #                 if is_use_gpu:
        #                     model = model.cuda()
        #                     data = {sensor: data[sensor].cuda() for sensor in data}
        #                 coarsemaps, finemaps = model(data)
        #                 coarsemaps = [i.sigmoid() for i in coarsemaps]
        #                 finemaps = [i.sigmoid() for i in finemaps]
        #                 '''compute loss'''
        #                 loss_cfm = loss_func[0](coarsemaps[0], data['focus_map'], epoch) * \
        #                            configs['TRAIN']['loss_weights'][0]
        #                 loss_fm = loss_func[1](finemaps[0], data['focus_map'], epoch) * \
        #                           configs['TRAIN']['loss_weights'][1]
        #                 loss_bound = loss_func[2](finemaps[1], data['Boundary'], epoch) * \
        #                              configs['TRAIN']['loss_weights'][2]
        #                 # train_writer.add_scalars('Val Rate Local and Global',
        #                 #                          {'Global': model.proj_f.rate1, 'Local': model.proj_f.rate2},
        #                 #                          global_step=val_iter)
        #                 '''compute all loss'''
        #                 loss_batch = loss_cfm + loss_fm + loss_bound
        #                 '''optimize parameters'''
        #                 val_loss_epoch += loss_batch.item()
        #                 '''loss->tensorboard'''
        #                 train_writer.add_scalars('Val Train Loss', {'Coarse FM Loss': loss_cfm,
        #                                                             'Fine FM Loss': loss_fm,
        #                                                             'Fine Bound Detect Loss': loss_bound},
        #                                              global_step=val_iter)
        #                 '''total loss->tensorboard ; print information'''
        #                 train_writer.add_scalar('Val Iter Loss', loss_batch, global_step=val_iter)
        #                 val_bar.set_description(
        #                     'Epoch: {}/{}. Valid. Iter: {}/{}. Content Loss: {:.5f}  All loss: {:.5f} LR: {:.10f}'.format(
        #                         epoch, configs['TRAIN']['max_epoch'], iter + 1, val_num_iter, loss_batch,
        #                                                               loss_epoch / val_num_iter,
        #                         optimizer.param_groups[0]['lr']))
        #
        #                 if configs['TRAIN']['debug_interval'] is not None and val_iter % 1000 == 0:
        #                     """反归一化 将归一化的输入数据反归一化后方便测试和展示"""
        #                     unnorm = torchvision.transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5),
        #                                                               (1 / 0.5, 1 / 0.5, 1 / 0.5))
        #                     for sensor in data:
        #                         if sensor == 'focus_map' or sensor == 'Boundary' or sensor == 'coarse_fm' or sensor == 'coarse_bound':
        #                             data.update({sensor: data[sensor]})
        #                         else:
        #                             data.update({sensor: unnorm(data[sensor])})
        #                     input_imgs1 = debug1(configs['MODEL'], configs['TRAIN_DATASET'], data)
        #                     input_imgs1 = [input_imgs1[sensor] for sensor in configs['MODEL']['input_sensors'][0:2]]
        #                     input_imgs2 = debug2(configs['MODEL'], configs['TRAIN_DATASET'], data)
        #                     input_imgs2 = [input_imgs2[sensor] for sensor in configs['MODEL']['input_sensors'][2:4]]
        #                     train_writer.add_image('Val Far and Near', torch.cat(input_imgs1, dim=2), val_iter,
        #                                            dataformats='NCHW')
        #                     train_writer.add_image('valid focus_map and Boundary', torch.cat(input_imgs2, dim=2), val_iter,
        #                                            dataformats='NCHW')
        #                     train_writer.add_image('valid coarse maps', torch.cat(debug_segmap(coarsemaps), dim=2),
        #                                            val_iter, dataformats='NCHW')
        #                     train_writer.add_image('valid fine maps and bounds', torch.cat(debug_segmap_(finemaps), dim=2),
        #                                            val_iter, dataformats='NCHW')
        #                 val_iter += 1
        #                 val_bar.update(1)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from skimage.metrics import structural_similarity as SSIM
# from PIL import Image
# import torchvision.transforms as transforms
from core.loss.SSIM_Loss import SSIM_Loss as SSIM
# from core.loss.loss_ssim import ssim
# from core.loss.DICE_Loss import DiceLoss
# from core.loss.DetailLoss import DetailAggregateLoss
# from core.loss.FocalLoss import FocalLoss
# from core.loss.MSSIM import MSSSIM
# from core.loss.MS_SSIM_L1_LOSS import MS_SSIM_L1_LOSS
import math

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()
        # self.cannyconv = Canny()
    def forward(self, image_fused, image_gt):
        Loss_gradient = 0
        batchsize = image_fused.shape[0]
        image_fused, image_gt = image_fused.to(torch.uint8).cpu().numpy(), image_gt.to(torch.uint8).cpu().numpy()
        for b in range(batchsize):
            gradient_fused = cv2.Canny(image_fused[b], threshold1=0.1, threshold2=0.5, apertureSize=3) / 255
            gt_gradient = cv2.Canny(image_gt[b], threshold1=0.1, threshold2=0.5, apertureSize=3) / 255
            Loss_gradient += np.mean(np.abs(gradient_fused - gt_gradient))# + rate[1] * F.l1_loss(gradient_fused, image_B)
        Loss_gradient = Loss_gradient / batchsize
        return Loss_gradient
'''---------------------Sobel 边缘检测实现----------------------------------------'''
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
'''---------------------Canny 边缘检测实现----------------------------------------'''
from scipy.signal.windows import gaussian
class Canny(nn.Module):
    def __init__(self, filter_size=9, std=1.0, device='cuda', threshold1=50, threshold2=120):
        super(Canny, self).__init__()
        generated_filters = gaussian(filter_size, std=std).reshape([1, filter_size
                                                   ])

        self.gaussian_filter_horizontal = nn.Parameter(data=torch.FloatTensor(generated_filters[None, None, ...]),requires_grad=False).cuda()

        self.gaussian_filter_vertical =  nn.Parameter(data=torch.FloatTensor(generated_filters.T[None, None, ...]),requires_grad=False).cuda()

        self.sobel_filter_horizontal = nn.Parameter(data=torch.FloatTensor([[[
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]]]]),requires_grad=False).cuda()
        self.sobel_filter_vertical = nn.Parameter(data=torch.FloatTensor([[[
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]]]]),requires_grad=False).cuda()
        self.directional_filter = nn.Parameter(data=torch.FloatTensor([[[[ 0.,  0.,  0.],
              [ 0.,  1., -1.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0., -1.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0., -1.,  0.]]],
            [[[ 0.,  0.,  0.],
              [ 0.,  1.,  0.],
              [-1.,  0.,  0.]]],
            [[[ 0.,  0.,  0.],
              [-1.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[-1.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0., -1.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]],
            [[[ 0.,  0., -1.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  0.]]]]),requires_grad=False).cuda()
        self.connect_filter = nn.Parameter(data=torch.FloatTensor([[[
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.]]]]),requires_grad=False).cuda()
        # 配置运行设备
        self.device = device
        # 配置阈值
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.filter_size = filter_size
        # 高斯滤波器
        # self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), bias=False)
        # self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), bias=False)
        #
        # # Sobel 滤波器
        # self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        #
        # # 定向滤波器
        # self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
        #
        # # 连通滤波器
        # self.connect_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)


    def forward(self, img):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = F.conv2d(img_r, weight=self.gaussian_filter_horizontal,padding=(0,self.filter_size//2))
        blurred_img_r = F.conv2d(blur_horizontal, weight=self.gaussian_filter_vertical,padding=(self.filter_size//2,0))
        blur_horizontal = F.conv2d(img_g, weight=self.gaussian_filter_horizontal,padding=(0,self.filter_size//2))
        blurred_img_g = F.conv2d(blur_horizontal, weight=self.gaussian_filter_vertical,padding=(self.filter_size//2,0))
        blur_horizontal = F.conv2d(img_b, weight=self.gaussian_filter_horizontal,padding=(0,self.filter_size//2))
        blurred_img_b = F.conv2d(blur_horizontal, weight=self.gaussian_filter_vertical,padding=(self.filter_size//2,0))

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = F.conv2d(blurred_img_r, weight=self.sobel_filter_horizontal,padding=1)
        grad_y_r = F.conv2d(blurred_img_r, weight=self.sobel_filter_vertical,padding=1)
        grad_x_g = F.conv2d(blurred_img_g, weight=self.sobel_filter_horizontal,padding=1)
        grad_y_g = F.conv2d(blurred_img_g, weight=self.sobel_filter_vertical,padding=1)
        grad_x_b = F.conv2d(blurred_img_b, weight=self.sobel_filter_horizontal,padding=1)
        grad_y_b = F.conv2d(blurred_img_b, weight=self.sobel_filter_vertical,padding=1)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # Step3: 非最大抑制，边缘细化
        all_filtered = F.conv2d(grad_mag, weight=self.directional_filter,padding=1)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)

        indices = (inidices_positive.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        indices = (inidices_negative.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # Step4: 双阈值
        low_threshold = min(self.threshold1, self.threshold2)
        high_threshold = max(self.threshold1, self.threshold2)
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = F.conv2d(higher.float(), weight=self.connect_filter,padding=1)
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()

        return thresholded

'''-----------------------内容损失----------------------------------'''
class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_fused, image_gt):
        # image_A = image_A.unsqueeze(0)
        # image_B = image_B.unsqueeze(0)
        # intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        # print('intensity_joint shape:', intensity_joint.shape)
        # intensity_joint = torch.max(image_A, image_B)
        # L1 对异常值不敏感，鲁棒性好，最低点不可导，不易到达最优解，梯度稳定；L2 对异常值敏感，离最优解越远时梯度大，越近时梯度越小，能够到达最优解；
        # smooth l1 loss 平滑之后的L1，结合L1和L2的优点
        Loss_intensity = F.mse_loss(image_fused, image_gt)  # torch.sqrt(torch.pow((image_fused - image_gt), 2)).mean()  # F范数损失
        return Loss_intensity

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))

        return loss
'''-----------------------weight BCE----------------------------------'''

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights = bin_label_weights * valid_mask

    return bin_labels, bin_label_weights, valid_mask

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False,
                         **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
            Note: In bce loss, label < 0 is invalid.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.size(1) == 1:
        # For binary class segmentation, the shape of pred is
        # [N, 1, H, W] and that of label is [N, H, W].
        # As the ignore_index often set as 255, so the
        # binary class label check should mask out
        # ignore_index
        assert label[label != ignore_index].max() <= 1, \
            'For pred with shape [N, 1, H, W], its label must have at ' \
            'most 2 classes'
        pred = pred.squeeze()
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        # `weight` returned from `_expand_onehot_labels`
        # has been treated for valid (non-ignore) pixels
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.shape, ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    # average loss over non-ignored and valid elements
    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()

    loss = F.binary_cross_entropy(
        pred, label.float(), reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den
def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def dice_loss_with_weights(inputs, targets, positive_weight, negative_weight):
    smooth = 1.
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # calculate true positive, false positive, false negative counts
    tp = (positive_weight * targets * inputs + negative_weight * (1 - targets) * (1 - inputs)).sum()
    fp = (positive_weight * targets + negative_weight * (1 - targets)).sum()
    fn = (positive_weight * inputs + negative_weight * (1 - inputs)).sum()

    # calculate dice score and weighted loss
    dice = 1 - (2 * tp + smooth) / (fp + fn + smooth)

    return dice.mean()

def iou_loss(pred, mask):
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    return iou

# from core.loss.FocalLoss import WeightedBCELoss
class Loss_CFM(nn.Module):
    def __init__(self):
        super(Loss_CFM, self).__init__()

    def forward(self, coarsemaps, masks, epoch):
        return F.l1_loss(coarsemaps, masks)
class Loss_Patch(nn.Module):
    def __init__(self):
        super(Loss_Patch, self).__init__()

    def forward(self, finemaps, masks, epoch):
        # im_arr = finemaps[0].detach().cpu().round().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        # canny = np.zeros(finemaps[0].shape)
        # for i in range(finemaps[0].shape[0]):
        #     canny[i] = cv2.Canny(im_arr[i]*255, 50, 100)//255
        # canny = finemaps[0]
        bound = 1 - 2 * (finemaps[0] - 0.5).abs()
        mask = (finemaps[1] - 0.5 + masks).round()
        return 0.5 * F.l1_loss(finemaps[1], bound) + 0.5 * F.l1_loss(finemaps[0], mask)
class Loss_FM(nn.Module):
    def __init__(self):
        super(Loss_FM, self).__init__()

    def forward(self, finemaps, masks, epoch):
        return F.l1_loss(finemaps, masks)
# from core.loss.FocalLoss import WeigthedBCELoss
class Loss_Bound(nn.Module):
    def __init__(self):
        super(Loss_Bound, self).__init__()
        # self.focal = WeigthedBCELoss()
    def forward(self, boundmaps, bound_target, epoch):
        return 0.5 * dice_loss_func(boundmaps, bound_target) + F.binary_cross_entropy(boundmaps, bound_target)
class Loss_MFF(nn.Module):
    def __init__(self):
        super(Loss_MFF, self).__init__()
        # self.focal = WeigthedBCELoss()
        self.bce, self.dice, self.near = 0, 0, 0
        self.far_ssim, self.near_ssim = 0, 0
        self.ssim_loss = SSIM()

    def forward(self, focusmaps, target):
        self.bce = 0.6 * F.binary_cross_entropy(focusmaps, target)
        # self.far = F.mse_loss(noisefar, noise_f)
        # self.near = F.mse_loss(noisenear, noise_n)
        dice = dice_loss_func(focusmaps, target)
        self.dice = 0.4 * dice  # (torch.log((torch.exp(dice) + torch.exp(-dice)) / 2.0))
        # return self.bce + self.far + self.near
        return self.bce + self.dice

class Loss_DN(nn.Module):
    def __init__(self):
        super(Loss_DN, self).__init__()
        # self.focal = WeigthedBCELoss()
        self.l2, self.ssim = 0, 0
        self.ssim_loss = SSIM()

    def forward(self, pred_noise, noise, epoch):
        self.l2 = F.mse_loss(pred_noise, noise)
        return self.l2
# import torch
# from scipy import ndimage
# class EuclideanDistanceTransformLoss(torch.nn.Module):
#     def __init__(self):
#         super(EuclideanDistanceTransformLoss, self).__init__()
#
#     def forward(self, pred, target):
#         # 计算欧几里得距离变换
#         distance = self.eucledian_distance_transform(target)
#         # 计算损失
#         loss = torch.mean((pred - distance) ** 2)
#         return loss
#
#     def eucledian_distance_transform(self, target):
#         # 使用distance_transform_edt函数计算欧几里得距离变换
#         distance = torch.zeros_like(target)
#         for i in range(target.shape[0]):
#             for j in range(target.shape[1]):
#                 distance[i, j] = torch.tensor(scipy.ndimage.distance_transform_edt(target[i, j]))
#         return distance

from skimage.morphology import remove_small_objects
import yaml
import numpy as np
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.poly_lr import PolyLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config["TRAIN"]["max_epoch"] * n_iter_per_epoch)
    warmup_steps = int(config["TRAIN"]["WARMUP_EPOCHS"] * n_iter_per_epoch)
    decay_steps = int(config["TRAIN"]["DECAY_EPOCHS"] * n_iter_per_epoch)

    lr_scheduler = None
    if config["TRAIN"]["LR_SCHEDULER"] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            # t_mul=1.,
            lr_min=config["TRAIN"]["MIN_LR"],
            warmup_lr_init=config["TRAIN"]["WARMUP_LR"],
            warmup_t=warmup_steps,
            cycle_limit=2,
            cycle_decay=2,
            t_in_epochs=False,
        )
    elif config["TRAIN"]["LR_SCHEDULER"] == 'poly':
            lr_scheduler = PolyLRScheduler(
                optimizer,
                t_initial=num_steps,
                # t_mul=1.,
                power=0.5,
                lr_min=config["TRAIN"]["MIN_LR"],
                warmup_lr_init=config["TRAIN"]["WARMUP_LR"],
                warmup_t=warmup_steps,
                cycle_limit=1,
                cycle_decay=0.5,
                t_in_epochs=False,
            )
    elif config["TRAIN"]["LR_SCHEDULER"] == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config["TRAIN"]["WARMUP_LR"],
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config["TRAIN"]["LR_SCHEDULER"] == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config["TRAIN"]["DECAY_EPOCHS"],
            warmup_lr_init=config["TRAIN"]["WARMUP_LR"],
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
        return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def debug1(model_config, dataset_config, input_images):
    batch_szie, _, _, _ = input_images[model_config['input_sensors'][0]].shape
    input_imgs = {sensor: [] for sensor in model_config['input_sensors'][0:2]}
    dev = input_images[model_config['input_sensors'][0]].device
    for batch in range(batch_szie):
        img = {sensor: input_images[sensor][batch, :, :, :] for sensor in model_config['input_sensors'][0:2]}
        # img = {sensor: img[sensor] for sensor in model_config['input_sensors'][0:2]}

        for sensor in model_config['input_sensors'][0:2]:
            input_imgs[sensor].append(img[sensor])

    input_imgs = {sensor: torch.stack(input_imgs[sensor], dim=0).to(dev) for sensor in model_config['input_sensors'][0:2]}
    # fusion_imgs = torch.stack(fusion_imgs, dim=0).to(dev)
    return input_imgs #, fusion_imgs
def debug2(model_config, dataset_config, input_images):
    batch_szie, _, _, _ = input_images[model_config['input_sensors'][0]].shape
    input_imgs = {sensor: [] for sensor in model_config['input_sensors'][2:4]}
    # fusion_imgs = []

    dev = input_images[model_config['input_sensors'][0]].device
    for batch in range(batch_szie):
        img = {sensor: input_images[sensor][batch, :, :, :] for sensor in model_config['input_sensors'][2:4]}
        # fusion = fusion_images[batch, :, :, :]
        # channels = fusion.shape[0]
        # std = torch.Tensor(dataset_config['std']).to(dev).view(channels, 1, 1).expand_as(fusion) if channels == 3 \
        #     else torch.Tensor([sum(dataset_config['std']) / len(dataset_config['std'])]).to(dev).view(channels, 1,
        #                                                                                               1).expand_as(
        #     fusion)
        # mean = torch.Tensor(dataset_config['mean']).to(dev).view(channels, 1, 1).expand_as(fusion) if channels == 3 \
        #     else torch.Tensor([sum(dataset_config['mean']) / len(dataset_config['mean'])]).to(dev).view(
        #     channels, 1, 1).expand_as(fusion)
        # img = {sensor: img[sensor] * std + mean for sensor in model_config['input_sensors']}
        # fusion = fusion * std + mean
        # img = {sensor: img[sensor] for sensor in model_config['input_sensors'][2:]}

        for sensor in model_config['input_sensors'][2:4]:
            input_imgs[sensor].append(img[sensor])
        # fusion_imgs.append(fusion)

    input_imgs = {sensor: torch.stack(input_imgs[sensor], dim=0).to(dev) for sensor in model_config['input_sensors'][2:4]}
    # fusion_imgs = torch.stack(fusion_imgs, dim=0).to(dev)
    return input_imgs #, fusion_imgs
def debug3(model_config, dataset_config, input_images):
    batch_szie, _, _, _ = input_images[model_config['input_sensors'][0]].shape
    sensor = model_config['input_sensors'][0]
    input_imgs = {sensor: []}
    # fusion_imgs = []
    dev = input_images[model_config['input_sensors'][0]].device
    for batch in range(batch_szie):
        img = {sensor: input_images[sensor][batch, :, :, :]}
        input_imgs[sensor].append(img[sensor])
    input_imgs = {sensor: torch.stack(input_imgs[sensor], dim=0).to(dev)}
    return input_imgs
def debug4(model_config, dataset_config, input_images):
    batch_szie, _, _, _ = input_images[model_config['input_sensors'][0]].shape
    sensor = model_config['input_sensors'][2]
    input_imgs = {sensor: []}
    # fusion_imgs = []
    dev = input_images[model_config['input_sensors'][0]].device
    for batch in range(batch_szie):
        img = {sensor: input_images[sensor][batch, :, :, :]}
        input_imgs[sensor].append(img[sensor])
    input_imgs = {sensor: torch.stack(input_imgs[sensor], dim=0).to(dev)}
    return input_imgs
def debug_segmap(input_images):
    dev = input_images[0].device
    input_image = []
    batch_szie, _, _, _ = input_images[0].shape
    for b in range(batch_szie):
        # input_image[0].append(torch.unsqueeze(input_images[b][0], dim=0))
        # input_image[1].append(torch.unsqueeze(input_images[b][1], dim=0))
        input_image.append(input_images[0][b])
        # input_image[3].append(input_images[b][1].unsqueeze(0)*imgs['Far'][b] + input_images[b][0].unsqueeze(0)*imgs['Near'][b])
        # input_image.append(torch.unsqueeze(torch.argmax(input_images[b], dim=0), dim=0))
        # input_image.append(img)
    input_imgs = [torch.stack(input_image, dim=0).to(dev)]
    return input_imgs
def debug_segmap1(input_images):
    dev = input_images[0].device
    input_image = []
    batch_szie, _, _, _ = input_images.shape
    for b in range(batch_szie):
        # input_image[0].append(torch.unsqueeze(input_images[b][0], dim=0))
        # input_image[1].append(torch.unsqueeze(input_images[b][1], dim=0))
        input_image.append(input_images[b])
        # input_image[3].append(input_images[b][1].unsqueeze(0)*imgs['Far'][b] + input_images[b][0].unsqueeze(0)*imgs['Near'][b])
        # input_image.append(torch.unsqueeze(torch.argmax(input_images[b], dim=0), dim=0))
        # input_image.append(img)
    input_imgs = [torch.stack(input_image, dim=0).to(dev)]
    return input_imgs
def debug_segmap_(input_images):
    dev = input_images[0].device
    input_image = [[], []]
    batch_szie, _, _, _ = input_images[0].shape
    for b in range(batch_szie):
        input_image[0].append(input_images[0][b])
        input_image[1].append(input_images[1][b])
    input_imgs = [torch.stack(input_image[i], dim=0).to(dev) for i in range(2)]
    return input_imgs

def debug_fusedimg(input_images):
    dev = input_images[0].device
    input_image = []
    batch_szie, _, _, _ = input_images.shape
    for b in range(batch_szie):
        input_image.append(input_images[b])
    input_imgs = torch.stack(input_image, dim=0).to(dev)
    return input_imgs

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = image * self.std + self.mean  # (image - self.mean)/self.std
        return image
def debug_segmap_boundary(input_images):
    dev = input_images[0].device
    input_image = []
    batch_szie, _, _, _ = input_images.shape
    for b in range(batch_szie):
        input_image.append(input_images[b])
    input_imgs = [torch.stack(input_image, dim=0).to(dev)]
    return input_imgs

def post_remove_small_objects(input_image, iter):
    if iter < 20:
        _, H, W = input_image.shape
        size = 0.001*H*W
    elif iter >= 20 and iter < 26:
        _, H, W = input_image.shape
        size = 0.07 * H * W
    else:
        _, H, W = input_image.shape
        size = 0.001 * H * W
    if type(input_image) is torch.Tensor:
        ar = input_image.detach().cpu().numpy()
    ar=ar.astype(np.bool)
    tmp_image1 = remove_small_objects(ar, size)
    tmp_image2 = (1-tmp_image1).astype(np.bool)
    tmp_image3 = remove_small_objects(tmp_image2, size)
    tmp_image4 = 1-tmp_image3
    tmp_image4 = tmp_image4.astype(np.float)
    # if type(input_image) is torch.Tensor:
    #     tmp_image4 = torch.from_numpy(tmp_image4)
    #     tmp_image4 = tmp_image4.to(input_image.device)
    return tmp_image4
# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
import numpy as np
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
import pydensecrf.densecrf as dcrf

def CRF(source_imgs, decisionmaps, save_path):
    fn_im = source_imgs  # Far images
    fn_anno = decisionmaps  # decision maps
    fn_output = save_path  # final output path

    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR

    anno_rgb = imread(fn_anno).astype(np.uint32)
    # anno_rgb = anno_rgb.astype(np.uint32)

    # anno_rgb = anno_rgb.astype(np.uint32)
    anno_rgb[anno_rgb < 1] = 1
    anno_rgb[anno_rgb > 1] = 255

    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print(
            "Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print(
            "If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        sdims1 = 5
        feats = create_pairwise_gaussian(sdims=(sdims1, sdims1), shape=img.shape[:2])  # 3
        d.addPairwiseEnergy(feats, compat=5,  # 3
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        sdims = 110
        schan = 13
        feats = create_pairwise_bilateral(sdims=(sdims, sdims), schan=(schan, schan, schan),  # 50,20
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=6,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    # imwrite(fn_output, MAP.reshape(img.shape))
    imwrite(fn_output, MAP.reshape(imread(fn_anno).shape))

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    d.stepInference(Q, tmp1, tmp2)
# CRF(r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\test\Far\lytro-01.jpg',
#     r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\result\MFLIT\1\init\lyto-0410.jpg',
#     r'F:\A_Matte_MFIFGAN\Pytorch_Image_Fusion-main\datasets\result\MFLIT\1\init\lyto-04100.jpg')

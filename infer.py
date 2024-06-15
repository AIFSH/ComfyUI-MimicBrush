import os
import cv2
import torch
import numpy as np
from PIL import Image
from cuda_malloc import cuda_malloc_supported
import torch.nn.functional as F
from torchvision.transforms import Compose

from MimicBrush.depthanything.fast_import import depth_anything_model 
from MimicBrush.depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor

from MimicBrush.mimicbrush import MimicBrush_RefNet
from MimicBrush.models.depth_guider import DepthGuider
from MimicBrush.models.ReferenceNet import ReferenceNet
from MimicBrush.models.pipeline_mimicbrush import MimicBrushPipeline


from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download


now_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(now_dir,"weights")
os.makedirs(weights_path,exist_ok=True)

ms_snapshot_download('xichen/cleansd', cache_dir=weights_path)
print('=== Pretrained SD weights downloaded ===')
ms_snapshot_download('xichen/MimicBrush', cache_dir=weights_path)
print('=== MimicBrush weights downloaded ===')

cleansd_weights_path = os.path.join(weights_path,"xichen","cleansd")
mimicbrush_weights_path = os.path.join(weights_path,"xichen","MimicBrush")
# === load the checkpoint ===
base_model_path = os.path.join(cleansd_weights_path,"stable-diffusion-inpainting")
vae_model_path = os.path.join(mimicbrush_weights_path,"sd-vae-ft-mse")
image_encoder_path = os.path.join(mimicbrush_weights_path,"image_encoder")
ref_model_path = os.path.join(cleansd_weights_path,"stable-diffusion-v1-5")
mimicbrush_ckpt = os.path.join(mimicbrush_weights_path,"mimicbrush","mimicbrush.bin")
depth_model_path = os.path.join(mimicbrush_weights_path,"depth_model","depth_anything_vitb14.pth")
device = "cuda" if cuda_malloc_supported() else "cpu"

depth_guider = DepthGuider()
mask_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def collage_region(low, high, mask):
    mask = (np.array(mask) > 128).astype(np.uint8)
    low = np.array(low).astype(np.uint8) 
    low = (low * 0).astype(np.uint8) 
    high = np.array(high).astype(np.uint8)
    mask_3 = mask 
    collage = low * mask_3 + high * (1-mask_3)
    collage = Image.fromarray(collage)
    return collage

def pad_img_to_square(original_image, is_mask=False):
    width, height = original_image.size
    
    if height == width:
        return original_image
    
    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)
    
    if is_mask:
        new_image = Image.new("RGB", new_size, "black")
    else:
        new_image = Image.new("RGB", new_size, "white")
    
    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image

def crop_padding_and_resize(ori_image, square_image):
    ori_height, ori_width, _ = ori_image.shape
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(square_image, (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale)))
    padding_size = max(resized_square_image.shape[0] - ori_height, resized_square_image.shape[1] - ori_width)
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :,:]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right,:]
    return cropped_image


def vis_mask(image, mask):
    # mask 3 channle 255
    mask = mask[:,:,0]
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw outlines, using random colors
    outline_opacity = 0.5
    outline_thickness = 5
    outline_color = np.concatenate([ [255,255,255], [outline_opacity]  ])

    white_mask = np.ones_like(image) * 255

    mask_bin_3 = np.stack([mask,mask,mask],-1) > 128
    alpha = 0.5 
    image = ( white_mask * alpha + image * (1-alpha) ) * mask_bin_3 + image * (1-mask_bin_3)
    cv2.polylines(image, mask_contours, True, outline_color, outline_thickness, cv2.LINE_AA)
    return image 


mimicbrush_model = None
def infer_single(ref_image, target_image, target_mask, seed = -1, num_inference_steps=50, guidance_scale = 5, enable_shape_control = False):
    #return ref_image
    """
    mask: 0/1 1-channel  np.array
    image: rgb           np.array
    """
    global mimicbrush_model 
    if not mimicbrush_model:
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", in_channels=13, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(dtype=torch.float16)

        pipe = MimicBrushPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            unet=unet,
            feature_extractor=None,
            safety_checker=None,
        )
        depth_anything_model.load_state_dict(torch.load(depth_model_path))
        referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet").to(dtype=torch.float16)

        mimicbrush_model = MimicBrush_RefNet(pipe, image_encoder_path, mimicbrush_ckpt,  depth_anything_model, depth_guider, referencenet, device)


    ref_image = ref_image.astype(np.uint8)
    target_image = target_image.astype(np.uint8)
    target_mask  = target_mask .astype(np.uint8)

    ref_image = Image.fromarray(ref_image.astype(np.uint8)) 
    ref_image = pad_img_to_square(ref_image)

    target_image = pad_img_to_square(Image.fromarray(target_image))
    target_image_low = target_image


    target_mask = np.stack([target_mask,target_mask,target_mask],-1).astype(np.uint8) * 255
    target_mask_np = target_mask.copy()
    target_mask = Image.fromarray(target_mask) 
    target_mask = pad_img_to_square(target_mask, True)

    target_image_ori = target_image.copy()
    target_image = collage_region(target_image_low, target_image, target_mask)
    

    depth_image = target_image_ori.copy()
    depth_image = np.array(depth_image)
    depth_image = transform({'image': depth_image})['image']
    depth_image = torch.from_numpy(depth_image).unsqueeze(0) / 255

    if not enable_shape_control:
        depth_image = depth_image * 0

    mask_pt = mask_processor.preprocess(target_mask, height=512, width=512)

    pred, depth_pred = mimicbrush_model.generate(pil_image=ref_image, depth_image = depth_image, num_samples=1, num_inference_steps=num_inference_steps,
                            seed=seed, image=target_image, mask_image=mask_pt, strength=1.0, guidance_scale=guidance_scale)


    depth_pred = F.interpolate(depth_pred, size=(512,512), mode = 'bilinear', align_corners=True)[0][0]
    depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
    depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
    depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:,:,::-1]

    pred = pred[0]
    pred = np.array(pred).astype(np.uint8)
    return pred, depth_pred.astype(np.uint8)


def inference_single_image(ref_image, 
                           tar_image, 
                           tar_mask, 
                           ddim_steps, 
                           scale, 
                           seed,
                           enable_shape_control,
                           ):
    if seed == -1:
        seed = np.random.randint(10000)
    pred, depth_pred = infer_single(ref_image, tar_image, tar_mask, seed, num_inference_steps=ddim_steps, guidance_scale = scale, enable_shape_control = enable_shape_control)
    return pred, depth_pred
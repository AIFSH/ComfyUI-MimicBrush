import cv2
import torch
import numpy as np
import os,site
now_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]

for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/MimicBrush.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/MimicBrush\n"
                    % (now_dir,now_dir)
                )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/MimicBrush.pth" % (site_packages_root)):
    print("!!!MimicBrush path was added to " + "%s/MimicBrush.pth" % (site_packages_root) 
    + "\n if meet No module named 'MimicBrush' error,please restart comfyui")

from .infer import inference_single_image,crop_padding_and_resize,vis_mask

class MimicBrushNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "edit_img":("IMAGE",),
                "edit_mask":("MASK",),
                "ref_img":("IMAGE",),
                "step":("INT",{
                    "min":1,
                    "max":100,
                    "step":1,
                    "default":50,
                    "display":"slider"
                }),
                "guidance_scale":("FLOAT",{
                    "min":-30.,
                    "max":30.,
                    "step":0.1,
                    "default":5.,
                    "display":"slider"
                }),
                "seed":("INT",{
                    "default":-1
                }),
                "if_keep_shape":("BOOLEAN",{
                    "default": False
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "FSH_MimicBrush"

    def generate(self,edit_img,edit_mask,ref_img,step,guidance_scale,seed,if_keep_shape):
        #np.array(image).astype(np.float32) / 255.0
        image = edit_img.numpy()[0] * 255
        image = image.astype(np.uint8)
        # print(image)
        mask = edit_mask.numpy()[0]
        # print(mask)
        print(image.shape, mask.shape, mask.max(), mask.min())

        ref_image = ref_img.numpy()[0] * 255
        ref_image = ref_image.astype(np.uint8)

        if mask.sum() == 0:
            raise print('No mask for the edit image.')
    
        mask_3 = np.stack([mask,mask,mask],-1).astype(np.uint8) * 255

        mask_alpha = mask_3.copy()
        for i in range(10):
            mask_alpha = cv2.GaussianBlur(mask_alpha, (3, 3), 0)
        
        synthesis, depth_pred = inference_single_image(ref_image.copy(), image.copy(), mask.copy(),
                                                       ddim_steps=step,scale=guidance_scale,seed=seed,
                                                       enable_shape_control=if_keep_shape)


        synthesis = crop_padding_and_resize(image, synthesis)
        depth_pred = crop_padding_and_resize(image, depth_pred)


        mask_3_bin = mask_alpha / 255
        synthesis = synthesis * mask_3_bin + image * (1-mask_3_bin)
        
        vis_source = vis_mask(image, mask_3) / 255.0
        out_image = torch.from_numpy(np.stack((synthesis / 255.0 ,depth_pred / 255.0 ,vis_source,mask_3),axis=0))
        
        return (out_image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MimicBrushNode": MimicBrushNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MimicBrushNode": "MimicBrush Node"
}

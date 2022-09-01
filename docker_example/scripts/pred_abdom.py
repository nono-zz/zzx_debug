import os
# cwd = os.getcwd()
import nibabel as nib
import numpy as np
# from utils.model_noise import Unet
# from model_noise import Unet
from model_noise import UNet
import torch
# import cv2
print(torch.__version__)


# def predict_folder_pixel_abs(input_folder, target_folder):
#     for f in os.listdir(input_folder):

#         source_file = os.path.join(input_folder, f)
#         target_file = os.path.join(target_folder, f)

#         nimg = nib.load(source_file)
#         nimg_array = nimg.get_fdata()

#         nimg_array[nimg_array < 0.01] = 0.5

#         abnomal_score_array = np.abs(nimg_array - 0.5)
        

#         final_nimg = nib.Nifti1Image(abnomal_score_array, affine=nimg.affine)
#         nib.save(final_nimg, target_file)


def predict_folder_sample_abs(input_folder, target_folder):
    for f in os.listdir(input_folder):
        abnomal_score = np.random.rand()

        with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
            write_file.write(str(abnomal_score))
            
def cal_distance_map(input, target):
    d_map = np.full_like(input, 0)
    d_map = np.square(input - target)
    return d_map
            
def predict_folder_pixel_abs(input_folder, target_folder):

    n_input = 1
    n_classes = 1           # the target is the reconstructed image
    depth = 4
    wf = 6
    
    model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True)
    ckp_path = '/workspace/adbom_best.pth'
    model_state_dict = torch.load(ckp_path, map_location=torch.device('cpu'))
    
    # model_state_dict = torch.load('/home/zhaoxiang/output/abdom_0.0001_700_bs8_ws_skip_connection_Gaussian_noise/last.pth')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
      
    model.cuda()
    model.eval()
    
    for f in os.listdir(input_folder):
        
        print(f)
        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)
    
        # load data
        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()  
        img_tensor = torch.from_numpy(nimg_array)
        img_tensor = img_tensor.float()
        img_tensor = torch.permute(img_tensor, (2, 0, 1))
        
        pixelPred = np.zeros_like(nimg_array)
        
        with torch.no_grad():
            
            for i in range(img_tensor.shape[0]):
                
                img = img_tensor[i,:,:].unsqueeze(dim = 0).unsqueeze(dim = 0)
                img = img.cuda()
                input = img
                output = model(input)
                
                difference = cal_distance_map(output[0,0,:,:].to('cpu').detach().numpy(), input[0,0,:,:].to('cpu').detach().numpy())
                pixelPred[i,:,:] = difference
                
            pixelPred = np.transpose(pixelPred, [1,2,0])
            
        final_nimg = nib.Nifti1Image(pixelPred, affine=nimg.affine)
        nib.save(final_nimg, target_file)
        
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input", required=True, type=str)
    # parser.add_argument("-o", "--output", required=True, type=str)
    # parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

    parser.add_argument("-i", "--input", default='/home/datasets/mood/data/brain/toy', type=str)
    parser.add_argument("-o", "--output", default='/home/zhaoxiang/output', type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    
    if mode == "pixel":
        predict_folder_pixel_abs(input_dir, output_dir)
    elif mode == "sample":
        predict_folder_sample_abs(input_dir, output_dir)
    else:
        print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

    # predict_folder_sample_abs("/home/david/data/datasets_slow/mood_brain/toy", "/home/david/data/datasets_slow/mood_brain/target_sample")

import argparse
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def read_output_images(folder_path):
    images_list = []
    images_greyscale = []
    
    for filename in os.listdir(folder_path):
        
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            
            file_path = os.path.join(folder_path, filename)
            
            image = cv2.imread(file_path)
            img = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
            resized_image = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
            grey_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            
            images_list.append(resized_image)
            images_greyscale.append(grey_img)
    return images_list, images_greyscale

def read_input_images(folder_path):
    images_list_input = []
    images_greyscale_input = []
    
    for filename in os.listdir(folder_path):
        
        if filename.startswith("r_") and filename.endswith(".png") and len(filename.split('_')) == 2:
            
            file_path = os.path.join(folder_path, filename)
            
            image = cv2.imread(file_path)
            resized_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_AREA)
            grey_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            
            images_list_input.append(resized_image)
            images_greyscale_input.append(grey_img)
    return images_list_input, images_greyscale_input

def main(input_folder, output_folder):
    images_list_input, images_greyscale_input = read_input_images(input_folder)
    images_list, images_greyscale = read_output_images(output_folder)

    print("Number of input images found:", len(images_list_input))
    print("Number of output images found:", len(images_list))

    mse_loss = [np.sum((image1 - image2) ** 2) for image1, image2 in zip(images_list_input, images_list)]
    psnr = [float('inf') if loss == 0 else -10. * np.log(loss) / np.log(10.) for loss in mse_loss]

    ssim_list = [ssim(image1, image2, full=True, data_range=image2.max() - image2.min())[0] 
                 for image1, image2 in zip(images_greyscale_input, images_greyscale)]

    print(f"avg mse loss: ", np.mean(mse_loss))
    print(f"avg psnr loss: ", np.mean(psnr))
    print(f"avg ssim loss: ", np.mean(ssim_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image comparison script")
    parser.add_argument("--input_folder", help="Path to the folder containing original Test set images")
    parser.add_argument("--output_folder", help="Path to the folder containing output Test set images")

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
import os
import json
import numpy as np
import cv2
import imageio
import torch
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip
from skimage.metrics import structural_similarity as ssim

from NeRFModel import NerfModel

def get_rays(basedir):
    
    splits = ['train', 'val', 'test']
    final_arrays = {}
    
    for split in splits:
        with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
        for frame in meta['frames']:
            fname = os.path.join(basedir, frame['file_path'][2:] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # Normalize images
        if imgs.shape[3] == 4: 
            imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])

        H, W = imgs.shape[1], imgs.shape[2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        H //= 2
        W //= 2
        focal /= 2
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        N = imgs_half_res.shape[0]
        rays_o = np.zeros((N, H*W, 3))
        rays_d = np.zeros((N, H*W, 3))
        target_px_values = imgs_half_res.reshape((N, H*W, 3))

        for i in range(N):
            c2w = poses[i]
            u = np.arange(W)
            v = np.arange(H)
            u, v = np.meshgrid(u, v)
            dirs = np.stack((u - W / 2, -(v - H / 2), -np.ones_like(u) * focal), axis=-1)
            dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

            rays_d[i] = dirs.reshape(-1, 3)
            rays_o[i] += c2w[:3, 3]

        rays_o = rays_o.reshape(N*H*W, 3)
        rays_d = rays_d.reshape(N*H*W, 3)
        target_px_values = target_px_values.reshape(N*H*W, 3)

        final_array = np.concatenate((rays_o, rays_d, target_px_values), axis=1)

        # with open(f'{split}_set.pkl', 'wb') as f:
        #     pickle.dump(final_array, f)
        
        final_arrays[split] = final_array

    return final_arrays, H, W


def compute_accumulated_transmittance(alphas): #Refering to Equation 3 in paper
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn, hf, total_sampling):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, total_sampling, device=device).expand(ray_origins.shape[0], total_sampling)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(total_sampling, ray_directions.shape[0], 3).transpose(0, 1) 
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)


def ssim_function(m, n): #m = ground truth image, n = predicted image
    m_reshaped = m.reshape(32,32,3).data.cpu().numpy()
    n_reshaped = n.reshape(32,32,3).data.cpu().numpy()
    m_uint8 = cv2.convertScaleAbs(m_reshaped)
    m_gray = cv2.cvtColor(m_uint8, cv2.COLOR_BGR2GRAY)
    n_gray = cv2.cvtColor(n_reshaped, cv2.COLOR_BGR2GRAY)
    ssim_val, _ = ssim(m_gray, n_gray, full=True, data_range=m_gray.max() - m_gray.min())
    
    return ssim_val


def creatvideo():
    
    clip = ImageSequenceClip(output_dir, 30)
    clip.write_videofile('Output_NeRF.mp4')


def plot_metrics(train_values, val_values, test_values=None, ylabel='', title='', save_path=''):
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_values) + 1), torch.tensor(train_values).cpu().numpy(), label='Training', color='blue')
    plt.plot(range(1, len(val_values) + 1), torch.tensor(val_values).cpu().numpy(), label='Validation', color='green')
    if test_values is not None:
        plt.plot(range(1, len(test_values) + 1), torch.tensor(test_values).cpu().numpy(), label='Testing', color='red')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)  # Save the plot
    plt.show()
    
    
@torch.no_grad()
def test(hn, hf, dataset, img_index, total_sampling, H, W, chunk_size=10):
    
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    ground_truth_px_values = dataset[img_index * H * W: (img_index + 1) * H * W, 6:]
    
    data = []   # list of regenerated pixel values
    testing_loss = []
    test_psnr_values = []
    
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_test = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_test = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ground_truth_px_values_test = ground_truth_px_values[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_test, ray_directions_test, hn, hf, total_sampling)
        loss = torch.mean((ground_truth_px_values_test - regenerated_px_values) ** 2)
        
        # Calculate PSNR
        if loss == 0:
            psnr_value = float('inf')
        else: 
            psnr_value = -10. * torch.log(loss) / torch.log(torch.Tensor([10.]).to(device))
 
        testing_loss.append(loss.item())
        test_psnr_values.append(psnr_value)
        data.append(regenerated_px_values)
    
    average_testing_loss = sum(testing_loss) / len(testing_loss)
    average_psnr_loss = sum(test_psnr_values) / len(test_psnr_values)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'{output_dir}/img_{img_index}.png', bbox_inches='tight')
    plt.close()
    
    return average_testing_loss, average_psnr_loss


def validate(nerf_model, validation_dataset, device, hn, hf, total_sampling):
    
    with torch.no_grad():
        ray_origins = validation_dataset[:, :3].to(device)
        ray_directions = validation_dataset[:, 3:6].to(device)
        ground_truth_px_values = validation_dataset[:, 6:].to(device)
        
        regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn, hf, total_sampling)
        loss = torch.mean((ground_truth_px_values - regenerated_px_values) ** 2)
        
        ssim_value = ssim_function(ground_truth_px_values, regenerated_px_values)
        
        # Calculate PSNR
        if loss == 0:
            psnr_value = float('inf')
        else: 
            psnr_value = -10. * torch.log(loss) / torch.log(torch.Tensor([10.]).to(device)).item()
            
    return loss, psnr_value, ssim_value


def train(nerf_model, optimizer, scheduler, data_loader_train, data_loader_val, device, hn, hf, epochs, total_sampling, H, W):
    
    training_loss = []
    validation_loss = []
    train_psnr_values = []
    val_psnr_values = []
    train_ssim_values = []
    val_ssim_values = []
    testing_losslist = []
    test_psnr_valueslist = []
    
    
    for epoch in tqdm(range(epochs)):
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_psnr = 0.0
        epoch_val_psnr = 0.0
        epoch_train_ssim = 0.0
        epoch_val_ssim = 0.0
        
        num_batches_train = 0
        num_batches_val = 0
        
        for train_batch, val_batch in zip(data_loader_train, data_loader_val):
            ray_origins = train_batch[:, :3].to(device)
            ray_directions = train_batch[:, 3:6].to(device)
            ground_truth_px_values = train_batch[:, 6:].to(device)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn, hf, total_sampling) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            
            ssim_value = ssim_function(ground_truth_px_values, regenerated_px_values)
            
            if loss == 0:
                psnr_value = float('inf')
            else: 
                psnr_value = -10. * torch.log(loss) / torch.log(torch.Tensor([10.]).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.mean().item()
            epoch_train_psnr += psnr_value.mean().item()
            epoch_train_ssim += ssim_value
            num_batches_train += 1
            
            val_loss, val_psnr_value, ssim_value = validate(nerf_model, val_batch, device, hn, hf, total_sampling)
            epoch_val_loss += val_loss.mean().item()
            epoch_val_psnr += val_psnr_value.mean().item()
            epoch_val_ssim += ssim_value
            num_batches_val += 1
        
        # Calculate average loss and PSNR for the epoch
        epoch_train_loss /= num_batches_train
        epoch_train_psnr /= num_batches_train
        epoch_train_ssim /= num_batches_train
        epoch_val_loss /= num_batches_val
        epoch_val_psnr /= num_batches_val
        epoch_val_ssim /= num_batches_val
        
        training_loss.append(epoch_train_loss)
        validation_loss.append(epoch_val_loss)
        train_psnr_values.append(epoch_train_psnr)
        val_psnr_values.append(epoch_val_psnr)
        train_ssim_values.append(epoch_train_ssim)
        val_ssim_values.append(epoch_val_ssim)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.ckpt')
        torch.save({
            'Epoch': epoch,
            'model_state_dict': nerf_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_loss': epoch_train_loss,
            'validation_loss': epoch_val_loss,
            'train_psnr_values': epoch_train_psnr,
            'val_psnr_values': epoch_val_psnr,
            'train_ssim_values': epoch_train_ssim,
            'val_ssim_values': epoch_val_ssim
        }, checkpoint_path)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")
        
        testing_loss_list = []
        test_psnr_values_list = []
        
        for img_index in range(200):
            testing_loss, test_psnr_values =test(hn, hf, testing_dataset, img_index, total_sampling, H, W)
            testing_loss_list.append(testing_loss)
            test_psnr_values_list.append(test_psnr_values)
        
        average_testing_loss = sum(testing_loss_list) / len(testing_loss_list)
        average_psnr_loss = sum(test_psnr_values_list) / len(test_psnr_values_list)
        
        testing_losslist.append(average_testing_loss)
        test_psnr_valueslist.append(average_psnr_loss)
    
    
    plot_metrics(train_psnr_values, val_psnr_values, test_psnr_valueslist, 'PSNR', 'PSNR vs. Epoch', os.path.join(checkpoint_dir, 'psnr_vs_epoch.png'))

    plot_metrics(train_ssim_values, val_ssim_values, None, 'SSIM', 'SSIM vs. Epoch', os.path.join(checkpoint_dir, 'ssim_vs_epoch.png'))

    plot_metrics(training_loss, validation_loss, testing_losslist, 'MSE Loss', 'MSE Loss vs. Epoch', os.path.join(checkpoint_dir, 'mse_loss_vs_epoch.png'))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Neural Radiance Fields (NeRF) model.')
    parser.add_argument('--epoch', type=int, default=16, help='Number of Epochs')
    parser.add_argument('--lr_initial', type=int, default=5e-4, help='Initial Learning Rate')
    parser.add_argument('--lr_final', type=float, default=5e-5, help='Final Learning Rate')
    parser.add_argument('--L_x', type=int, default=10, help='Parameter L_x for Positional Encoding in NerfModel')
    parser.add_argument('--L_d', type=int, default=4, help='Parameter L_d for Positional Encoding in NerfModel')
    parser.add_argument('--hn', type=int, default=2, help='Near bound for NerfModel Volumetric Rendering')
    parser.add_argument('--hf', type=int, default=6, help='Far bound for NerfModel Volumetric Rendering')
    parser.add_argument('--fine', type=int, default=128, help='Number of Fine Sampling')
    parser.add_argument('--course', type=int, default=64, help='Number of Course Sampling')
    parser.add_argument('--basedir', type=str, default='data/lego', help='Base directory containing dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='Checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='Pred_test_output', help='Directory to save test predictions')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arguments()
    epoch = args.epoch
    lr_initial = args.lr_initial
    lr_final = args.lr_final
    L_x = args.L_x
    L_d = args.L_d
    hn = args.hn
    hf = args.hf
    basedir = args.basedir
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    total_sampling = args.fine + args.course
    
    device = 'cpu'
    rays_dict, H, W = get_rays(basedir)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = 'cuda'
    
    training_dataset = torch.from_numpy(rays_dict['train']).to(device)
    validation_dataset = torch.from_numpy(rays_dict['val']).to(device)
    testing_dataset = torch.from_numpy(rays_dict['test']).to(device)
    model = NerfModel(L_x, L_d).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr_initial)
    gamma = (lr_final / lr_initial) ** (1.0 / epoch)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma)
    data_loader_train = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    data_loader_val = DataLoader(validation_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader_train, data_loader_val, device, hn, hf, epoch, total_sampling, H, W)
    creatvideo()
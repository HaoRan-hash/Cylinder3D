import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import datetime
import shutil
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
from utils.log_util import create_logger

import warnings

warnings.filterwarnings("ignore")


def val(logger, model, val_dataset_loader, configs, pytorch_device, lovasz_softmax, loss_func,
        unique_label, unique_label_str):
    model.eval()
    hist_list = []
    val_loss_list = []
    with torch.no_grad():
        for val_xyz, val_pt_lab, val_pt_fea in tqdm(val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).to(dtype=torch.float32, device=pytorch_device) for i in val_pt_fea]
            val_pt_lab_ten = [torch.from_numpy(i).to(dtype=torch.int64, device=pytorch_device) for i in val_pt_lab]
            val_xyz_ten = [torch.from_numpy(i).to(dtype=torch.float32, device=pytorch_device) for i in val_xyz]
            val_batch_size = len(val_pt_fea_ten)
            outputs, voxel_label_tensor, cat_pt_ind = model(val_pt_fea_ten, val_pt_lab_ten, val_xyz_ten, val_batch_size,
                                                        configs)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs).detach(), voxel_label_tensor,
                                ignore=0) + loss_func(outputs.detach(), voxel_label_tensor)
            outputs = torch.argmax(outputs, dim=1)
            
            outputs = outputs.detach().cpu().numpy()
            cat_pt_ind = cat_pt_ind.detach().cpu().numpy()
            hist_list.append(fast_hist_crop(outputs[cat_pt_ind[:, 0], cat_pt_ind[:, 1], cat_pt_ind[:, 2], cat_pt_ind[:, 3]],
                                            np.concatenate(val_pt_lab), unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())
    model.train()
    iou = per_class_iu(sum(hist_list))
    logger.info('Validation per class iou:')
    for class_name, class_iou in zip(unique_label_str, iou):
        logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100

    return val_miou, np.mean(val_loss_list)


def main(args, logger, output_path):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    
    model_save_path = output_path

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    my_model.to(pytorch_device)
    # resume
    if args.model_ckpt:
        my_model.load_state_dict(torch.load(args.model_ckpt, map_location=pytorch_device))
    
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        pbar = tqdm(train_dataset_loader)
        for i_iter, (train_xyz, train_pt_lab, train_pt_fea) in enumerate(pbar):
            train_pt_fea_ten = [torch.from_numpy(i).to(dtype=torch.float32, device=pytorch_device) for i in train_pt_fea]
            train_pt_lab_ten = [torch.from_numpy(i).to(dtype=torch.int64, device=pytorch_device) for i in train_pt_lab]
            train_xyz_ten = [torch.from_numpy(i).to(dtype=torch.float32, device=pytorch_device) for i in train_xyz]
            train_batch_size = len(train_pt_fea_ten)
            # forward + backward + optimize
            outputs, voxel_label_tensor, _ = my_model(train_pt_fea_ten, train_pt_lab_ten, train_xyz_ten, train_batch_size, 
                                                    configs)   # 其实是voxel_label_tensor
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), voxel_label_tensor, ignore=0) + loss_func(
                outputs, voxel_label_tensor)
            
            loss.backward()
            optimizer.step()
            
            # 实时刷新
            if i_iter % configs['train_params']['show_gap'] == 0:
                pbar.set_postfix_str(f'loss={loss.item():.4f}')
            
            optimizer.zero_grad()
            global_iter += 1
        
        # if 里面是做 val
        if global_iter % check_iter == 0 and epoch >= 0:
            val_miou, val_loss = val(logger, my_model, val_dataset_loader, configs, pytorch_device, lovasz_softmax, loss_func,
                                        unique_label, unique_label_str)
            
            # save model if performance is improved
            if best_val_miou < val_miou:
                best_val_miou = val_miou
                torch.save(my_model.state_dict(), str(model_save_path / 'best.pth'))

            logger.info('Current val miou is %.3f while the best val miou is %.3f' %
                    (val_miou, best_val_miou))
            logger.info('Current val loss is %.3f \n' %
                    (val_loss))
            torch.cuda.empty_cache()   # val之后释放所有显存
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--model_ckpt')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    
    output_path = Path(f'outputs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    output_path.mkdir(mode=0o755)
    logger = create_logger(str(output_path / 'train_val.log'))
    shutil.copy(args.config_path, str(output_path))
    main(args, logger, output_path)

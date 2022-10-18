import datetime
import os
import sys
import argparse
import logging

import cv2
import time
import numpy as np

import random
import torch
import torch.utils.data
import torch.optim as optim

import tensorboardX
from torchsummary import summary

from utils.dataset_processing.evaluation import evaluation
from utils.saver import Saver
from models import get_network
from models.common import post_process_output
from models.loss import focal_loss
from utils.data.grasp_data import GraspDataset

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')
    
    # Network
    parser.add_argument('--network', type=str, default='ggcnn2', choices=['ggcnn', 'ggcnn2'], help='Network Name in .models')
    
    # Dataset & Data & Training
    parser.add_argument('--dataset-path', default='./dataset/cornell',type=str, help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')  

    parser.add_argument('--output-size', type=int, default=300, help='output size')

    parser.add_argument('--outdir', type=str, default='ggcnn_new_dataset/output', help='Training Output Directory')
    parser.add_argument('--modeldir', type=str, default='models', help='model saving path')
    parser.add_argument('--logdir', type=str, default='tensorboard', help='log directory ')
    parser.add_argument('--imgdir', type=str, default='ggecnn_new_dataset/output/img', help='predicted image path')
    parser.add_argument('--max_models', type=int, default=3, help='maximum save models')
    # device
    parser.add_argument('--device-name', type=str, default='cuda:0', choices=['cpu', 'cuda:0'], help='use GPU')
    # description
    parser.add_argument('--description', type=str, default='', help='Training description')
    

    parser.add_argument('--goon-train', type=bool, default=False, help='whether train from previous model')
    parser.add_argument('--model', type=str, default='', help='saved model')
    parser.add_argument('--start-epoch', type=int, default=146, help='begining epoch')
    args = parser.parse_args()

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(net, device, val_data, saver, args):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'accuracy': 0.0,
        'graspable': 0,
        'loss': 0,
        'losses': {
        }
    }

    ld = len(val_data)

    with torch.no_grad():     # no backwards
        batch_idx = 0
        for x, y in val_data:
            batch_idx += 1
            print ("\r Validating... {:.2f}".format(batch_idx/ld), end="")

            # predict and calculate loss
            lossd = focal_loss(net, x.to(device), y[0].to(device), y[1].to(device), y[2].to(device), y[3].to(device))

            
            pos_out, ang_out, wid_out = post_process_output(lossd['pred']['pred_pos'], 
                                                                 lossd['pred']['pred_cos'], 
                                                                 lossd['pred']['pred_sin'],
                                                                 lossd['pred']['pred_wid'])
            results['graspable'] += np.max(pos_out) / ld

            # evaluation 
            ang_tar = torch.atan2(y[2], y[1]) / 2.0
            ret = evaluation(pos_out, ang_out, wid_out, y[0], ang_tar, y[3])
            results['accuracy'] += ret / ld
            
            
            loss = lossd['loss']    
            results['loss'] += loss.item()/ld       
            for ln, l in lossd['losses'].items():   
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()/ld

    return results


def train(epoch, net, device, train_data, optimizer):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    sum_batch = len(train_data)
    for x, y in train_data:
        """
        x = (batch, 1, h, w)
        y = (batch, 1, h, w) (batch, 1, h, w) (batch, 1, h, w) (batch, 1, h, w) 
        """
        batch_idx += 1

      
        lossd = focal_loss(net, x.to(device), y[0].to(device), y[1].to(device), y[2].to(device), y[3].to(device))

        loss = lossd['loss']        

        if batch_idx % 1 == 0:
            logging.info('Epoch: {}, '
                        'Batch: {}/{}, '
                        'loss_pos: {:.5f}, '
                        'loss_cos: {:.5f}, '
                        'loss_sin: {:.5f}, '
                        'loss_wid: {:.5f}, '
                        'Loss: {:0.5f}'.format(
                epoch, batch_idx, sum_batch,
                lossd['losses']['loss_pos'], lossd['losses']['loss_cos'], lossd['losses']['loss_sin'], lossd['losses']['loss_wid'], 
                loss.item()))

        results['loss'] += loss.item()
        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results['loss'] /= batch_idx    
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def datasetloaders(Dataset, args):
    # load dataset 
    train_dataset = Dataset(args.dataset_path,
                            start=0.0, 
                            end=0.8,
                            output_size=args.output_size,
                            argument=True)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)


    train_val_dataset = Dataset(args.dataset_path,
                                start=0.0, 
                                end=0.2,
                                output_size=args.output_size,
                                argument=False)
    train_val_data = torch.utils.data.DataLoader(
        train_val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)


    val_dataset = Dataset(args.dataset_path,
                          start=0.8, 
                          end=1.0,
                          output_size=args.output_size,
                          argument=False)
    val_data = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)

    return train_data, train_val_data, val_data


def run():

    # setup_seed(2)
    args = parse_args()


    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    saver = Saver(args.outdir, args.logdir, args.modeldir, args.imgdir, net_desc)

    tb = saver.save_summary()


    logging.info('Loading Dataset...')
    train_data, train_val_data, val_data = datasetloaders(GraspDataset, args)
    print('>> train dataset: {}'.format(len(train_data) * args.batch_size))
    print('>> train_val dataset: {}'.format(len(train_val_data)))
    print('>> test dataset: {}'.format(len(val_data)))


    logging.info('Loading Network...')
    ggcnn = get_network(args.network)
    net = ggcnn()
    device_name = args.device_name if torch.cuda.is_available() else "cpu"
    if args.goon_train:
        
        pretrained_dict = torch.load(args.model, map_location=torch.device(device_name))
        net.load_state_dict(pretrained_dict, strict=True)   
    device = torch.device(device_name)     
    net = net.to(device)


    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)     

    # Print model architecture
    summary(net, (1, args.output_size, args.output_size))            
    saver.save_arch(net, (1, args.output_size, args.output_size))    

    # train 
    best_acc = 0.0
    start_epoch = args.start_epoch if args.goon_train else 0
    for _ in range(start_epoch):
        scheduler.step()
    for epoch in range(args.epochs)[start_epoch:]:
        logging.info('Beginning Epoch {:02d}, lr={}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    
        train_results = train(epoch, net, device, train_data, optimizer)
        scheduler.step()

   
        tb.add_scalar('train_loss/loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        if epoch % 1 == 0:
            logging.info('>>> Validating...')

   
            test_results = validate(net, device, val_data, saver, args)

            print('\n>>> test_graspable = {:.5f}'.format(test_results['graspable']))
            print('>>> test_accuracy: %f' % (test_results['accuracy']))

            tb.add_scalar('test_pred/test_graspable', test_results['graspable'], epoch)
            tb.add_scalar('test_pred/test_accuracy', test_results['accuracy'], epoch)
            tb.add_scalar('test_loss/loss', test_results['loss'], epoch)
            for n, l in test_results['losses'].items():
                tb.add_scalar('test_loss/' + n, l, epoch)

            train_val_results = validate(net, device, train_val_data, saver, args)

            print('\n>>> train_val_graspable = {:.5f}'.format(train_val_results['graspable']))
            print('>>> train_val_accuracy: %f' % (train_val_results['accuracy']))

            tb.add_scalar('train_val_pred/train_val_graspable', train_val_results['graspable'], epoch)
            tb.add_scalar('train_val_pred/train_val_accuracy', train_val_results['accuracy'], epoch)
            tb.add_scalar('train_val_loss/loss', train_val_results['loss'], epoch)
            for n, l in train_val_results['losses'].items():
                tb.add_scalar('train_val_loss/' + n, l, epoch)

            # save model
            accuracy = test_results['accuracy']
            if accuracy >= best_acc :
                print('>>> save model: ', 'epoch_%04d_acc_%0.4f.pth' % (epoch, accuracy))
                saver.save_model(net, 'epoch_%04d_acc_%0.4f.pth' % (epoch, accuracy))
                best_acc = accuracy
            else:
                print('>>> save model: ', 'epoch_%04d_acc_%0.4f_.pth' % (epoch, accuracy))
                saver.save_model(net, 'epoch_%04d_acc_%0.4f_.pth' % (epoch, accuracy))
                saver.remove_model(args.max_models)  

    tb.close()


if __name__ == '__main__':
    run()

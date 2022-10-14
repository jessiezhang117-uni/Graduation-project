import datetime
import datetime
from email import parser
import os
import sys
import argparse
import logging

import cv2
import time
import numpy as  np


import random
import torch
import torch.utils.data
import torch.optim as optim

import tensorboardX
from torch.summary import summary

from cornell import Cornell
from saver import Saver
from ggcnn import GGCNN
from ggcnn2 import GGCNN2
from function import post_process,detect_grasps,max_iou
from image import Image

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ggcnn')
    parser.add_argument('--network',type=str,default='ggcnn',choices=['ggcnn','ggcnn2'],help='network name')

    parser.add_argument('--dataset-path',default='ggcnn/cornell',type=str,help='dataset path')

    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--bathes-per-epoch', type=int, default=120, help='Batched per epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=2, help='Dataset workers') 

    parser.add_argument('--output-size', type=int, default=300, help='output size')

    parser.add_argument('--outdir', type=str, default='ggcnn/output', help='Training Output Directory')
    parser.add_argument('--modeldir', type=str, default='ggcnn/model', help='model save path')
    parser.add_argument('--logdir', type=str, default='ggcnn/output/tensorboard', help='summary save path')
    parser.add_argument('--imgdir', type=str, default='ggcnn/output/img', help='predicted image save path')
    parser.add_argument('--max_models', type=int, default=3, help='max number of saving models')
    
    # device
    parser.add_argument('--device-name', type=str, default='cuda:0', choices=['cpu', 'cuda:0'], help='whether use GPU')

    # train from exsiting model
    parser.add_argument('--goon-train', type=bool, default=False, help='whether train from existing model')
    parser.add_argument('--model', type=str, default='output/models/211128_1147_new/epoch_0145_acc_0.0000.pth', help='saved model')
    parser.add_argument('--start-epoch', type=int, default=146, help='beginning epoch')
    args = parser.parse_args()

    return args




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(net,device,val_data,batches_per_epoch):
    val_result = {
        'correct':0,
        'failed':0,
        'loss':0,
        'losses':{}
    }
    net.eval()

    length = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < (batches_per_epoch-1):
            for x,y,idx in val_data:
                batch_idx+=1

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                
                lossdict = net.compute_loss(xc,yc)

                loss = lossdict['loss']

                val_result['loss']+=loss.item()/length

                q_out,ang_out,width_out = post_process(lossdict['pred']['pos'],lossdict['pred']['cos'],
                                                        lossdict['pred']['sin'],lossdict['pred']['width'])
                grasps_pre = detect_grasps(q_out,ang_out,width_out,num_grasp=1)
                grasps_true = val_data.dataset.get_raw_grasps(idx)

                result = 0
                for grasp_pre in grasps_pre:
                    if max_iou(grasp_pre,grasps_true) >0.25:
                        result = 1
                        break
                
                if result:
                    val_result['correct'] +=1
                else:
                    val_result['failed'] +=1
                
        # if vis:
        #     if len(grasps_pre)>0:
        #         visualization(val_data,idx,grasps_pre,grasps_true)

        print('acc:{}'.format(val_result['correct']/(val_result['correct']+val_result['failed'])))
    return(val_result)


def train(epoch,net,device,train_data,optimizer,batches_per_epoch):
    """
    :return : loss for this epoch
    """
   
    results = {
        'loss': 0,
        'losses': {
        }
    }
    

    net.train()
    
    batch_idx = 0
    
    # start training 
    while batch_idx < batches_per_epoch:
        for x, y,idx in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            
            #to GPU
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            
            lossdict = net.compute_loss(xc,yc)
            
            #get loss 
            loss = lossdict['loss']
            
            #print training process
            if batch_idx % 1 == 0:
                logging.info('Epoch: {}, '
                        'Batch: {}, '
                        'loss_pos: {:.5f}, '
                        'loss_cos: {:.5f}, '
                        'loss_sin: {:.5f}, '
                        'loss_wid: {:.5f}, '
                        'Loss: {:0.5f}'.format(
                epoch, batch_idx, 
                loss['losses']['loss_pos'], loss['losses']['loss_cos'], loss['losses']['loss_sin'], loss['losses']['loss_wid'], 
                loss.item()))
                
                
            
            #calculate total loss
            results['loss'] += loss.item()
            #record loss fro pos,cos,sin,width
            for ln, l in lossdict['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
            
            #backward optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #calculate total mean loss
        results['loss'] /= batch_idx
        
        #calculate mean loss for each loss
        for l in results['losses']:
            results['losses'][l] /= batch_idx

        return results

def datasetloaders(Dataset,args):
    # train data
    train_dataset = Dataset(args.dataset_path,
                            start=0.0, 
                            end=0.01,
                            output_size=args.output_size,batches_per_epoch
                            argument=True)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    # train val data
    train_val_dataset = Dataset(args.dataset_path,
                                start=0.0, 
                                end=0.01,
                                output_size=args.output_size,
                                argument=False)
    train_val_data = torch.utils.data.DataLoader(
        train_val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=1)

    # val data
    val_dataset = Dataset(args.dataset_path,
                          start=0.99, 
                          end=1.0,
                          output_size=args.output_size,
                          argument=False)
    val_data = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=1)

    return train_data, train_val_data, val_data

def run(net):

    # setup_seed(2)
    args = parse_args()
    
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    saver = Saver(args.outdir, args.logdir, args.modeldir, args.imgdir, net_desc)
    
    # init tensorboard
    tb = saver.save_summary()
   
    net = net.to(device)
    
    # load dataset
    logging.info('Loading Dataset...')
    train_data, train_val_data, val_data = datasetloaders(Cornell, args)
    print('>> train dataset: {}'.format(len(train_data) * args.batch_size))
    print('>> train_val dataset: {}'.format(len(train_val_data)))
    print('>> test dataset: {}'.format(len(val_data)))

    # load network
    logging.info('Loading Network...')
    if args.network =='ggcnn':
        net = GGCNN()
    if args.network =='ggcnn2':
        net = GGCNN2()
    device_name = args.device_name if torch.cuda.is_available() else "cpu"
    if args.goon_train:
        # load pretrained model
        pretrained_dict = torch.load(args.model, map_location=torch.device(device_name))
        net.load_state_dict(pretrained_dict, strict=True)  
    device = torch.device(device_name)    
    net = net.to(device)
    
    
   
    # optimizer 
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)     
    logging.info('optimizer Done')

    # print network structure
    summary(net, (1, args.output_size, args.output_size))            
    saver.save_arch(net, (1, args.output_size, args.output_size))    

    # train


    best_acc = 0.0
    start_epoch = args.start_epoch if args.goon_train else 0
    for _ in range(start_epoch):
        scheduler.step()
    for epoch in range(args.epochs)[start_epoch:]:
        logging.info('Beginning Epoch {:02d}, lr={}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        # training 
        train_results = train(epoch, net, device, train_data, optimizer,args.batches_per_epoch)
        scheduler.step()

        # save to tensorboard
        tb.add_scalar('train_loss/loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        if epoch % 1 == 0:
            logging.info('>>> Validating...')

            # ====================== start validation ======================
            test_results = validate(net, device, val_data, saver, args.batches_per_epoch)
            # logging 
            print('\n>>> test_correct = {:.5f}'.format(test_results['correct']))
            print('>>> test_accuracy: %f' % (test_results['correct']/(test_results['correct']+test_results['failed'])))
            # save to tensorboard
            tb.add_scalar('test_pred/test_correct', test_results['correct'], epoch)
            tb.add_scalar('test_pred/test_accuracy', test_results['correct']/(test_results['correct']+test_results['failed']), epoch)
            tb.add_scalar('test_loss/loss', test_results['loss'], epoch)
            for n, l in test_results['losses'].items():
                tb.add_scalar('test_loss/' + n, l, epoch)

            # ====================== use train_val to validate ======================
            train_val_results = validate(net, device, train_val_data, saver, args)

            print('\n>>> train_val_correct = {:.5f}'.format(train_val_results['correct']))
            print('>>> train_val_accuracy: %f' % (train_val_results['correct']/(train_val_results['correct']+train_val_results['failed'])))

            tb.add_scalar('train_val_pred/train_val_correct', train_val_results['correct'], epoch)
            tb.add_scalar('train_val_pred/train_val_accuracy', train_val_results['correct']/(train_val_results['correct']+train_val_results['failed']), epoch)
            tb.add_scalar('train_val_loss/loss', train_val_results['loss'], epoch)
            for n, l in train_val_results['losses'].items():
                tb.add_scalar('train_val_loss/' + n, l, epoch)

            # save model
            accuracy = test_results['correct']/(test_results['correct']+test_results['failed'])
            if accuracy >= best_acc :
                print('>>> save model: ', 'epoch_%04d_acc_%0.4f.pth' % (epoch, accuracy))
                saver.save_model(net, 'epoch_%04d_acc_%0.4f.pth' % (epoch, accuracy))
                best_acc = accuracy
            else:
                print('>>> save model: ', 'epoch_%04d_acc_%0.4f_.pth' % (epoch, accuracy))
                saver.save_model(net, 'epoch_%04d_acc_%0.4f_.pth' % (epoch, accuracy))
                saver.remove_model(args.max_models)  # remove extra models

    tb.close()
    
if __name__ == '__main__':
    run()
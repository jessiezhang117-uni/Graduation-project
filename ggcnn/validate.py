import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2

from ggcnn import GGCNN
from cornell import Cornell
from function import post_process,detect_grasps,max_iou
from image import Image

batch_size = 32
batches_per_epoch = 120
epochs = 600
lr = 0.001

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
            if batch_idx % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            
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


def validate(net,device,val_data,batches_per_epoch,vis=False):
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
                
        if vis:
            if len(grasps_pre)>0:
                visualization(val_data,idx,grasps_pre,grasps_true)

        print('acc:{}'.format(val_result['correct']/(val_result['correct']+val_result['failed'])))
    return(val_result)

def run(net):

    device = torch.device("cuda:0")
   
    net = net.to(device)
    
    # train data
    train_data = Cornell("./ggcnn/cornell",output_size=300)

    train_dataset = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle = True)
    
    # validation data
    val_data = Cornell("./ggcnn/cornell",output_size=300)

    val_dataset = torch.utils.data.DataLoader(val_data,batch_size = 1,shuffle = True)
    
    optimizer = optim.Adam(net.parameters())
    
    
    for epoch in range(epochs):
        train_results = train(epoch, net, device, train_dataset, optimizer, batches_per_epoch)
        print('validating...')
        validate_results = validate(net,device,val_dataset,batches_per_epoch,vis=True)
    return train_results,validate_results

def visualization(val_data,idx,grasps_pre,grasps_true):
    img = Image.from_file(val_data.dataset.rgbf[idx])
    left = val_data.dataset._get_crop_attrs(idx)[1]
    top = val_data.dataset._get_crop_attrs(idx)[2]
    img.crop((left,top),(left+300,top+300)) # (top left, bottom right)

    a = img.img
    
    a_points = grasps_pre[0].as_gr.points.astype(np.uint8)
    b_points = grasps_true.points

    color1 = (255,255,0)
    color2 = (255,0,0)
    for i in range(3):
        img = cv2.line(a,tuple(a_points[i]),tuple(a_points[i+1]),color1 if i%2==0 else color2,1)
    img = cv2.line(a,tuple(a_points[3]),tuple(a_points[0]),color2,1)

    color1 = (0,0,0)
    color2 =(0,255,0)

    for b_point in b_points:
        for i in range(3):
            img = cv2.line(a,tuple(b_point[i]),tuple(b_point[i+1]),color1 if i%2==0 else color2,1)
        img = cv2.line(a,tuple(b_point[3]),tuple(b_point[0]),color2,1)
    cv2.imwrite('ggcnn/img.png',a)

if __name__ == '__main__':
    net = GGCNN(4)
    run(net)
    torch.save(net,'ggcnn/model/model_v.pt')

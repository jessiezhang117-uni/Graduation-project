
import torch
import torch.optim as optim

from cornell import Cornell
from ggcnn import GGCNN
import numpy as np
batch_size = 64
batches_per_epoch = 1200
epochs = 60
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
        for x, y,in train_data:
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

def run(net):
    
    device = torch.device("cuda:0")
    
   
    net = net.to(device)
    
    # get dataset
    cornell_data = Cornell("./ggcnn/cornell",output_size=300)

    dataset = torch.utils.data.DataLoader(cornell_data,batch_size = batch_size,shuffle = True)
    
   
    optimizer = optim.Adam(net.parameters())
    
    
    for epoch in range(epochs):
        train_results = train(epoch, net, device, dataset, optimizer, batches_per_epoch)
    
    return train_results
    
if __name__ == '__main__':
    net = GGCNN(4)
    run(net)
    torch.save(net,'ggcnn/model/best_model.pt')
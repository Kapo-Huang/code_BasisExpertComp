import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.optim as optim
import time
from model import *
from utils import *
from skimage.io import imsave
from skimage.io import imread
from skimage import data,img_as_float,img_as_int
import lpips


def _build_checkpoint_payload(model, args, dataset, epoch):
    return {
        'model_state': model.state_dict(),
        'y_mean': 0.0,
        'y_std': 1.0,
        'meta': {
            'dataset': args.dataset,
            'application': args.application,
            'epoch': int(epoch),
            'batch_size': int(args.batch_size),
            'factor': int(args.factor),
            'coord_order': 'x_y_z_t',
            'output': 'scalar_value',
        }
    }


def _load_model_state(path):
    payload = torch.load(path)
    if isinstance(payload, dict) and 'model_state' in payload:
        return payload['model_state']
    return payload


def _epoch_sample_points(dataset, args):
    points_per_timestep = dataset.dim[0] * dataset.dim[1] * dataset.dim[2]
    if args.application == 'extrapolation':
        return dataset.training_samples * args.factor * args.batch_size
    if args.application in ['temporal', 'super-spatial']:
        return len(dataset.samples) * args.factor * args.batch_size
    if args.application == 'spatial':
        if args.factor * args.batch_size >= points_per_timestep:
            return len(dataset.samples) * points_per_timestep
        return len(dataset.samples) * args.factor * args.batch_size
    return len(dataset.samples) * args.factor * args.batch_size

def trainNet(model,args,dataset):
    device = torch.device(getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if args.application == 'spatial' or args.application == 'super-spatial':
        loss = open(args.model_path+args.dataset+'/'+'loss-'+args.application+'-'+str(args.scale)+'-'+str(args.init)+'-'+str(args.factor)+'.txt','w')
    elif args.application == 'temporal':
        loss = open(args.model_path+args.dataset+'/'+'loss-'+args.application+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.factor)+'-'+str(args.active)+'.txt','w')
    else:
        loss = open(args.model_path+args.dataset+'/'+'loss-'+args.application+'-'+str(args.factor)+'.txt','w')

    optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999),weight_decay=1e-6)
    criterion = nn.MSELoss()
    sampled_points_per_epoch = _epoch_sample_points(dataset, args)
    sampled_mb_per_epoch = sampled_points_per_epoch * 4.0 / (1024.0 * 1024.0)
    print('Per-epoch sampled points: {:,}'.format(sampled_points_per_epoch))
    print('Per-epoch sampled data (float32): {:.2f} MB'.format(sampled_mb_per_epoch))

    t = 0
    for itera in range(1,args.num_epochs+1):
        train_loader = dataset.GetTrainingData()
        x = time.time()

        print('======='+str(itera)+'========')
        print('Epoch {} sampling budget: {:,} points ({:.2f} MB)'.format(itera, sampled_points_per_epoch, sampled_mb_per_epoch))
        loss_mse = 0
        loss_grad = 0
        
        for batch_idx, (coord,v) in enumerate(train_loader):
            t1 = time.time()
            coord = coord.to(device)
            v = v.to(device)
            optimizer.zero_grad()
            v_pred = model(coord)
            mse = criterion(v_pred.view(-1),v.view(-1))
            mse.backward()
            loss_mse += mse.mean().item()
            optimizer.step()
            #print(time.time()-t1)
        
        y = time.time()
        t += y-x
        print(y-x)
        print("Epochs "+str(itera)+": loss = "+str(loss_mse))
        loss.write("Epochs "+str(itera)+": loss = "+str(loss_mse))
        loss.write('\n')

        if itera%args.checkpoint == 0 or itera==1:
            payload = _build_checkpoint_payload(model, args, dataset, itera)
            if args.application == 'spatial' or args.application == 'super-spatial':
                torch.save(payload,args.model_path+args.dataset+'/'+args.application+'-'+str(args.scale)+'-'+str(args.init)+'-'+str(args.factor)+'-'+str(itera)+'.pth')
            elif args.application == 'temporal':
                torch.save(payload,args.model_path+args.dataset+'/'+args.application+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.factor)+'-'+str(args.active)+'-'+str(itera)+'.pth')
            else:
                torch.save(payload,args.model_path+args.dataset+'/'+args.application+'-'+str(args.factor)+'-'+str(itera)+'.pth')

    loss.write("Time = "+str(t))
    loss.write('\n')
    loss.close()

def adjust_lr(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def inf(dataset,args):
    device = torch.device(getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    if args.application != 'viewsynthesis':
        if args.active == 'sine':
            model =  CoordNet(4,1,args.init,args.num_res)
        elif args.active == 'relu':
            model = CoordNetReLU(4,1,args.init,args.num_res)
    if args.application in ['spatial','super-spatial']:
        model.load_state_dict(_load_model_state(args.model_path+args.dataset+'/'+args.application+'-'+str(args.scale)+'-'+str(args.init)+'-'+str(args.factor)+'-'+str(args.num_epochs)+'.pth'))
    elif args.application == 'temporal':
        model.load_state_dict(_load_model_state(args.model_path+args.dataset+'/'+args.application+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.factor)+'-'+str(args.active)+'-'+str(args.num_epochs)+'.pth'))
    elif args.application == 'AO':
        model.load_state_dict(_load_model_state(args.model_path+args.dataset+'/'+args.application+'-'+str(args.factor)+'-'+str(args.num_epochs)+'.pth'))
    '''
    elif args.application == 'viewsynthesis':
        model.load_state_dict(torch.load(args.model_path+args.dataset+'/'+args.application+'-'+str(args.res)+'-'+str(args.angle)+'-'+str(args.num_epochs)+'.pth'))
    '''
    if args.application == 'viewsynthesis':
        model = torch.load(args.model_path+args.dataset+'/'+args.application+'-'+str(args.res)+'-'+str(args.angle)+'-'+str(args.num_epochs)+'.pth')
    model = model.to(device)

    if args.application in ['spatial','temporal']:
        idx = 0
        if args.hint == 'super':
            coords = dataset.GetTestingData()
            for i in range(0,dataset.total_samples-1):
                print(i+1)
                s = coords[i*dataset.dim[0]*dataset.dim[1]*dataset.dim[2]:(i+1)*dataset.dim[0]*dataset.dim[1]*dataset.dim[2],:,]
                e = coords[(i+1)*dataset.dim[0]*dataset.dim[1]*dataset.dim[2]:(i+2)*dataset.dim[0]*dataset.dim[1]*dataset.dim[2],:,]
                for t in range(0,6):
                    d = t/6*e+(1-t/6)*s
                    train_loader = DataLoader(dataset=torch.FloatTensor(d), batch_size=args.batch_size, shuffle=False)
                    v = []
                    for batch_idx, coord in enumerate(train_loader):
                        coord = coord.to(device)
                        with torch.no_grad():
                            v_pred = model(coord)
                            v += list(v_pred.view(-1).detach().cpu().numpy())
                    v = np.asarray(v,dtype='<f')
                    v.tofile('../Result/'+args.dataset+'/'+args.application+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.factor)+'-'+'{:04d}'.format(idx+1)+'.dat',format='<f')
                    idx += 1
        else:
            samples = dataset.dim[2]*dataset.dim[1]*dataset.dim[0]
            coords = get_mgrid([dataset.dim[0],dataset.dim[1],dataset.dim[2]],dim=3)
            time = np.zeros((samples,1))
            idx = 1
            for t in range(0,dataset.total_samples):
                print(t)
                t = t/(dataset.total_samples-1)
                t -= 0.5
                t *= 2
                time.fill(t)
                train_loader = DataLoader(dataset=torch.FloatTensor(np.concatenate((coords[:, [2,1,0]], time),axis=1)), batch_size=args.batch_size, shuffle=False)
                v = []
                for batch_idx, coord in enumerate(train_loader):
                    coord = coord.to(device)
                    with torch.no_grad():
                        v_pred = model(coord)
                        v += list(v_pred.view(-1).detach().cpu().numpy())
                v = np.asarray(v,dtype='<f')
                if args.application == 'temporal':
                    if args.active == 'sine':
                        v.tofile('../Result/'+args.dataset+'/'+args.application+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.factor)+'-'+'{:04d}'.format(idx)+'.dat',format='<f')
                    else:
                        v.tofile('../Result/'+args.dataset+'/'+args.application+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.factor)+'-'+args.active+'-'+'{:04d}'.format(idx)+'.dat',format='<f')
                elif args.application == 'spatial':
                    v.tofile('../Result/'+args.dataset+'/'+args.application+'-'+str(args.scale)+'-'+str(args.init)+'-'+str(args.factor)+'-'+'{:04d}'.format(idx)+'.dat',format='<f')
                idx += 1
    elif args.application == 'super-spatial':
        '''
        samples = dataset.dim[0]*self.scale*dataset.dim[1]*self.scale*dataset.dim[0]*self.scale
        coords = get_mgrid([dataset.dim[0]*self.scale,dataset.dim[1]*self.scale,dataset.dim[0]*self.scale],dim=3)
        time = np.zeros((samples,1))
        for t in range(0,dataset.total_samples):
            t = t/(dataset.total_samples-1)
            t -= 0.5
            t *= 2
            time.fill(t)
            train_loader = DataLoader(dataset=torch.FloatTensor(np.concatenate((coords[:, [2,1,0]], time),axis=1)), batch_size=args.batch_size, shuffle=False)
            v = []
            for batch_idx, coord in enumerate(train_loader):
                coord = coord.to(device)
                with torch.no_grad():
                    v_pred = model(coord)
                    v += list(v_pred.view(-1).detach().cpu().numpy())
            v = np.asarray(v,dtype='<f')
            v.tofile('/afs/crc.nd.edu/user/j/jhan5/CoordNet/Result/'+args.dataset+'/'+args.application+'-'+str(args.scale)+'-'+str(args.init)+'-'+str(args.factor)+'-'+'{:04d}'.format(i+1)+'.dat',format='<f')
        '''
        pixel_coords = np.stack(np.mgrid[24*4:224*4,24*4:224*4,400*4:600*4], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (248*4 - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (248*4 - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (600*4 - 1)
        pixel_coords -= 0.5
        pixel_coords *= 2.
        pixel_coords = np.reshape(pixel_coords,(-1,3))
        samples = 200*4*200*4*200*4
        time = np.zeros((samples,1))
        for t in range(0,50):
            print(t)
            t_ = t/49
            t_ -= 0.5
            t_ *= 2.
            time.fill(t_)
            train_loader = DataLoader(dataset=torch.FloatTensor(np.concatenate((pixel_coords, time),axis=1)), batch_size=args.batch_size, shuffle=False)
            v = []
            for batch_idx, coord in enumerate(train_loader):
                coord = coord.to(device)
                with torch.no_grad():
                    v_pred = model(coord)
                    v += list(v_pred.view(-1).detach().cpu().numpy())
            v = np.asarray(v,dtype='<f')
            v.tofile('../Result/'+args.dataset+'/'+args.application+'-'+str(args.scale)+'-'+str(args.init)+'-'+str(args.factor)+'-'+'{:04d}'.format(t+51)+'.dat',format='<f')
    elif args.application == 'viewsynthesis':
        import time
        coords = get_mgrid([args.res,args.res],dim=2)
        theta = np.zeros((args.res*args.res,1))
        phi = np.zeros((args.res*args.res,1))
        count = 0
        clock = 0
        for t in range(0,180):
            theta_ = t/179.0
            theta_ -= 0.5
            theta_ *= 2.0
            theta.fill(theta_)
            for p in range(234,235):
                phi_ = p/359.0
                phi_ -= 0.5
                phi_ *= 2.0
                phi.fill(phi_)
                train_loader = DataLoader(dataset=torch.FloatTensor(np.concatenate((coords,theta,phi),axis=1)), batch_size=args.batch_size, shuffle=False)
                r = []
                g = []
                b = []
                temp = time.time()
                for batch_idx, coord in enumerate(train_loader):
                    coord = coord.to(device)
                    with torch.no_grad():
                        v_pred = model(coord).permute(1,0)
                        r += list(v_pred[0].view(-1).detach().cpu().numpy())
                        g += list(v_pred[1].view(-1).detach().cpu().numpy())
                        b += list(v_pred[2].view(-1).detach().cpu().numpy())
                clock += time.time()-temp
                r = np.asarray(r)
                g = np.asarray(g)
                b = np.asarray(b)
                r = r.reshape(args.res,args.res).transpose()
                g = g.reshape(args.res,args.res).transpose()
                b = b.reshape(args.res,args.res).transpose()
                img = np.asarray([r,g,b])
                img /= 2.0
                img += 0.5
                img *= 255
                img = img.transpose(1,2,0)
                img = img.astype(np.uint8)
                imsave('../Images/'+args.dataset+'/'+str(args.res)+'/'+'{:05d}'.format(count)+'.png',img)
                count += 1
        print('Inference Time ='+str(clock/count)+'s')

    elif args.application == 'AO':
        samples = dataset.dim[2]*dataset.dim[1]*dataset.dim[0]
        coords = get_mgrid([dataset.dim[0],dataset.dim[1],dataset.dim[2]],dim=3)
        time = np.zeros((samples,1))
        idx = 0
        for t in range(0,dataset.total_samples):
            print(t+1)
            t = t/(dataset.total_samples-1)
            t -= 0.5
            t *= 2
            time.fill(t)
            train_loader = DataLoader(dataset=torch.FloatTensor(np.concatenate((coords[:, [2,1,0]], time),axis=1)), batch_size=args.batch_size, shuffle=False)
            v = []
            for batch_idx, coord in enumerate(train_loader):
                coord = coord.to(device)
                with torch.no_grad():
                    v_pred = model(coord)
                    v += list(v_pred.view(-1).detach().cpu().numpy())
            v = np.asarray(v,dtype='<f')
            v.tofile('../AO/'+args.dataset+'/MTNet/'+'{:04d}'.format(idx+1)+'.dat',format='<f')
            idx += 1







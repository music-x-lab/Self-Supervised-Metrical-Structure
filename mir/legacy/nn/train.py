import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mir.common import WORKING_PATH
import os
import numpy as np

class NetworkBehavior(nn.Module):

    def __init__(self):
        super().__init__()
        self.use_gpu=torch.cuda.device_count()>0
        self.use_data_parallel=False

    def forward(self, x):
        raise NotImplementedError()

    def init_settings(self, is_training):
        if(self.use_gpu):
            self.cuda()
        else:
            self.cpu()
        if(is_training):
            self.train()
        else:
            self.eval()
        if(self.use_data_parallel):
            self.parallel_net=[nn.DataParallel(self)]

    def feed(self, x):
        if(self.use_data_parallel):
            return self.parallel_net[0](x)
        else:
            return self(x)

    def loss(self, x, y):
        raise NotImplementedError()

    def inference(self, x):
        raise NotImplementedError()

class NetworkInterface:

    def __init__(self, net, save_name, load_checkpoint=False):
        self.net=net
        if(not isinstance(self.net,NetworkBehavior)):
            raise Exception('Invalid network type')
        if('(p)' in save_name):
            self.net.use_data_parallel=True
        self.net.init_settings(False)
        self.save_name=save_name
        save_path=os.path.join(WORKING_PATH,'cache_data/%s.sdict'%save_name)
        cp_save_path=os.path.join(WORKING_PATH,'cache_data/%s.cp.sdict'%save_name)
        self.finalized=False
        self.optimizer=optim.Adam(self.net.parameters())
        self.counter=0
        if(os.path.exists(save_path)):
            state_dict=torch.load(save_path)
            # The following codes are for torch 4.0 compatibility
            # new_state_dict={}
            # for key in state_dict['net']:
            #     if('num_batches_tracked' not in key):
            #         new_state_dict[key]=state_dict['net'][key]
            # self.net.load_state_dict(new_state_dict)
            self.net.load_state_dict(state_dict['net'])
            self.counter=state_dict['counter']
            self.optimizer.load_state_dict(state_dict['opt'])
            self.finalized=True
        elif(load_checkpoint and os.path.exists(cp_save_path)):
            state_dict=torch.load(cp_save_path)
            # The following codes are for torch 4.0 compatibility
            # new_state_dict={}
            # for key in state_dict['net']:
            #     if('num_batches_tracked' not in key):
            #         new_state_dict[key]=state_dict['net'][key]
            # self.net.load_state_dict(new_state_dict)
            self.net.load_state_dict(state_dict['net'])
            self.counter=state_dict['counter']
            self.optimizer.load_state_dict(state_dict['opt'])

    def train_supervised(self, train_set, val_set, batch_size, learning_rates_dict=1e-3,
              round_per_print=20, round_per_val=100, round_per_save=500):
        if(self.finalized):
            print('Model is already finalized. To perform new training, delete the save file.')
            return
        print('Training model %s'%self.net.__class__.__name__)
        if(isinstance(learning_rates_dict,dict)):
            learning_rates_list=sorted(learning_rates_dict.items(),reverse=True)
        else:
            learning_rates_list=[(learning_rates_dict,1.0)]
        # TODO: multi-thread support
        train_set_loader=DataLoader(train_set,batch_size=batch_size,shuffle=train_set.need_shuffle,num_workers=0)
        val_set_loader=DataLoader(val_set,batch_size=batch_size,shuffle=train_set.need_shuffle,num_workers=0)
        train_set_iter=iter(train_set_loader)
        val_set_iter=iter(val_set_loader)
        learning_rates=[x[0] for x in learning_rates_list]
        learning_rates_batch_count=[int(np.ceil(len(train_set)*x[1]/batch_size)) if x[1]>0
                                    else -int(x[1]) for x in learning_rates_list]
        print(len(train_set),'samples','batch_size =',batch_size)
        for (learning_rate,batch_count) in zip(learning_rates,learning_rates_batch_count):
            print(batch_count,'mini batches for learning rate =',learning_rate,flush=True)
        current_counter=0
        self.net.init_settings(True)
        for (learning_rate,batch_count) in zip(learning_rates,learning_rates_batch_count):
            i=0
            if(current_counter<self.counter):
                del_count=min(self.counter-current_counter,batch_count)
                i+=del_count
                current_counter+=del_count
            if(i<batch_count):
                for param_group in self.optimizer.param_groups:
                    param_group['lr']=learning_rate
                running_loss=0.0
                running_loss_count=0
                while(i<batch_count):
                    # print('Data fetch begin')
                    try:
                        input_tuple=next(train_set_iter)
                    except:
                        train_set_iter=iter(train_set_loader)
                        input_tuple=next(train_set_iter)
                    # print('Data fetch end')
                    if(self.net.use_gpu):
                        input_tuple=(var.cuda() for var in input_tuple)
                    self.optimizer.zero_grad()
                    loss=self.net.loss(*input_tuple)
                    loss.backward()
                    self.optimizer.step()
                    running_loss+=loss.item()
                    running_loss_count+=1
                    if(i%round_per_print==round_per_print-1):
                        print('[%f, %.2f%% (%d/%d)] loss: %.6f' %
                              (learning_rate,(i+1)/batch_count*100,i+1,batch_count,
                               running_loss/running_loss_count),flush=True)
                        running_loss=0.0
                        running_loss_count=0
                    if(i%round_per_val==round_per_val-1):
                        val_loss=0.0
                        for j in range(round_per_print):
                            try:
                                val_input_tuple=next(val_set_iter)
                            except:
                                val_set_iter=iter(val_set_loader)
                                val_input_tuple=next(val_set_iter)
                            if(self.net.use_gpu):
                                val_input_tuple=(var.cuda() for var in val_input_tuple)
                            with torch.no_grad():
                                val_loss+=self.net.loss(*val_input_tuple).item()
                        print('[%f, %.2f%% (%d/%d)] val_loss: %.6f' %
                              (learning_rate,(i+1)/batch_count*100,i+1,batch_count,
                               val_loss/round_per_print),flush=True)

                    if(i%round_per_save==round_per_save-1 or i+1==batch_count):
                        if(self.net.use_gpu):
                            self.net.cpu()
                        torch.save({'net':self.net.state_dict(),
                                    'opt':self.optimizer.state_dict(),
                                    'counter':current_counter+1
                                    },os.path.join(WORKING_PATH,'cache_data/%s.cp.sdict'%self.save_name))
                        if(self.net.use_gpu):
                            self.net.cuda()
                        print('[%f, %.2f%% (%d/%d)] checkpoint created' %
                              (learning_rate,(i+1)/batch_count*100,i+1,batch_count),flush=True)
                    i+=1
                    current_counter+=1
        if(self.net.use_gpu):
            self.net.cpu()
        torch.save({'net':self.net.state_dict(),
                    'opt':self.optimizer.state_dict(),
                    'counter':current_counter
                    },os.path.join(WORKING_PATH,'cache_data/%s.sdict'%self.save_name))
        if(self.net.use_gpu):
            self.net.cuda()

    def inference(self, *args):
        self.net.init_settings(False)
        inputs=[torch.tensor(arg,dtype=torch.float if arg.dtype in [np.float16,np.float32,np.float64] else torch.int)
                for arg in args]
        if(self.net.use_gpu):
            inputs=[input.cuda() for input in inputs]

        return self.net.inference(*inputs)


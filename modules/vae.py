import torch.nn as nn
import torch

def init_hidden(batch_size,hidden_dim,z,layer_num=1,direction=2):
    h_0=torch.zeros(direction*layer_num,batch_size,hidden_dim//2,device=z.device)
    c_0=torch.zeros(direction*layer_num,batch_size,hidden_dim//2,device=z.device)
    return h_0,c_0

def make_one_hot(vector,num_classes):
    if(len(vector.shape)==2):
        return torch.zeros((vector.shape[0],vector.shape[1],num_classes),device=vector.device)\
                .scatter_(dim=2,index=vector[:,:,None],value=1)
    elif(len(vector.shape)==1):
        return torch.zeros((vector.shape[0],num_classes),device=vector.device)\
                .scatter_(dim=1,index=vector[:,None],value=1)
    else:
        raise NotImplementedError()

def hard_max(x):
    '''
    :param x: (*,feature_dim)
    :return: (*,feature_dim), one-hot version
    '''
    feature_dim=x.shape[-1]
    raw_shape=x.shape
    if(len(raw_shape)!=2):
        x=x.view((-1,feature_dim))
    idx=x.max(1)[1]
    range_obj=torch.arange(x.shape[0],dtype=torch.long,device=x.device)
    result=torch.zeros_like(x,device=x.device)
    result[range_obj,idx]=1.0
    if(len(raw_shape)!=2):
        result=result.view(raw_shape)
    return result

class Reparameterizer(nn.Module):

    def __init__(self,input_hidden_size,z_dim):
        super(Reparameterizer, self).__init__()
        self.z_dim=z_dim
        self.linear_mu=nn.Linear(input_hidden_size,z_dim)
        self.linear_sigma=nn.Linear(input_hidden_size,z_dim)
        self.supress_warning=False

    def forward(self, z, is_training=None):
        '''
        :param z: (..., input_hidden_size)
        :return: (..., z_dim)
        '''
        if(is_training is None):
            if(not self.supress_warning):
                print('[Warning] The reparameterizer now requires a new explicit parameter is_training. Please fix your code.')
                self.supress_warning=True
            is_training=self.training
        mu=self.linear_mu(z)
        if(is_training):
            logvar=self.linear_sigma(z)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu),mu,logvar
        else:
            return mu,None,None

    def collect_statistics(self, z):
        mu=self.linear_mu(z)
        logvar=self.linear_sigma(z)
        return mu,logvar

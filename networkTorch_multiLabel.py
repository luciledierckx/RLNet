import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F
from sklearn.metrics import hamming_loss, zero_one_loss
from dataset_processing import load_and_transform_data
from sklearn.model_selection import  KFold
import sys

nb_epochs = 0

class L0Linear(nn.Linear):
    """
    This class initially comes from https://github.com/Joeyonng/decision-rules-network/blob/master/sparse_linear.py
        and was only modified with small changes (see in code)
    """
    def __init__(self, in_features, out_features,  bias=True, linear=F.linear, loc_mean=0, loc_sdev=0.01, 
                 beta=2 / 3, gamma=-0.1, zeta=1.1, loc_fct=F.linear, **kwargs):
        super(L0Linear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self._size = self.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.linear = linear
        self.penalty = 0
        self.register_buffer("uniform", torch.zeros(self._size))
        self.loc_fct = loc_fct

    def forward(self, x):
        mask, self.penalty = self._get_mask()
        masked_weight = self.weight * mask
        output = self.linear(x, masked_weight, self.bias)
        return output

    def masked_weight(self):
        mask, _ = self._get_mask()
        masked_weight = self.weight * mask
        return masked_weight
    
    def regularization(self, mean=True, axis=None):
        regularization = self.penalty
        if mean:
            regularization = regularization.mean() if axis == None else regularization.mean(axis)
        return regularization
    
    def _get_mask(self):
        def hard_sigmoid(x):
            return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))
        if self.training:
            self.uniform.uniform_()
            u = torch.autograd.Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio)
        else:
            s = torch.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            # We changed the penalty that was zero before because we do our callback based on the "eval" loss
            # and still need to include the lambda and regularisation in that loss which uses the penalty
            penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio)
        return hard_sigmoid(s), penalty
    
class Network(torch.nn.Module):
    """
    The architecture of our RLNet for multi-label classification
    """
    def __init__(self, size_in, nbRules, nbOutput, loc_mean=0, weights=None, labels=None, locs=None):
        """
        Args:
            size_in (int):  the size of the input features
            nbRules (int):  the maximum number of rules the network can use
            nbOutput (int): the number of output classes
            loc_mean (int): the initial value of the probability distribution of the hidden weights of the AND layer.
            weights (np array): left as None for a classical use of the network, 
                            can be used to initialize the AND layer weights when starting the training
            labels (np array): left as None for a classical use of the network,
                            can be used to initialize the labels weights and freeze them when starting the training
            locs (np array): left as None for a classical use of the network,
                            can be used to initialize the hidden weights of the AND layer when starting the training
        """

        super(Network, self).__init__()
        
        self.and_layer = L0Linear(size_in, nbRules, linear=AndFct.apply, loc_mean=loc_mean)
        self.and_layer.bias.data.fill_(1)
        self.and_layer.bias.requires_grad = False
        
        self.choice = nn.Linear(nbRules, nbRules+1)
        w = np.zeros((nbRules+1, nbRules))
        w[0][0] = 1
        w[-1,:] =-1
        for i in range(1, nbRules):
            for j in range(i+1):
                if i!= j: w[i][j] = -1
                else: w[i][j] = 1
                
        self.choice.weight.data = torch.FloatTensor(w)
        self.choice.weight.requires_grad = False
        b = np.zeros(nbRules+1)
        b[-1] = 1
        self.choice.bias.data = torch.FloatTensor(b)
        self.choice.bias.requires_grad = False
        self.choice.activation = nn.ReLU()
        
        self.output = nn.Linear(nbRules+1, nbOutput, bias=False)
        if labels is not None and weights is None and locs is None:
            self.output.weight.data = torch.FloatTensor(labels)
        #When we know the values we want to use for the weights but not for loc
        if weights is not None and locs is None and labels is None:
            self.and_layer.weight.data = torch.FloatTensor(weights)
            aux = (np.abs(weights)-0.5)*10
            self.and_layer.loc.data = torch.FloatTensor(aux)
        if weights is not None and locs is not None and labels is not None:
            self.and_layer.weight.data[:weights.shape[0]]=torch.FloatTensor(weights)
            self.and_layer.loc.data[:weights.shape[0]]=torch.FloatTensor(locs)
            self.output.weight.data[:,:weights.shape[0]]=torch.FloatTensor(labels)
            self.and_layer.weight.requires_grad = True
            self.and_layer.loc.requires_grad = True
            self.output.weight.requires_grad = True
        self.output.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.and_layer(x)
        x = Binarization.apply(x)
        x = self.choice(x) 
        x = self.choice.activation(x)
        y = x[:,-1]
        x = self.output(x)
        x = self.output.activation(x)
        return x, y
    
    def regularization(self):
        regularization = (self.and_layer.regularization(axis=1)).mean()
        return regularization
    
    def string(self):
        return f'AND masked weights = {self.and_layer.masked_weight()}, AND bias = {self.and_layer.bias}, OR masked weights = {self.weights2}'

class AndFct(torch.autograd.Function):
    """
    This class comes from https://github.com/Joeyonng/decision-rules-network/blob/master/DRNet.py
    """
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        output = x.mm(weight.t())
        output = output + bias.unsqueeze(0).expand_as(output)
        output = output - (weight * (weight > 0)).sum(-1).unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(x) - grad_output.sum(0).unsqueeze(1).expand_as(weight) * (weight > 0)
        grad_bias = grad_output.sum(0)
        grad_bias[(bias >= 1) * (grad_bias < 0)] = 0
        return grad_x, grad_weight, grad_bias        

class Binarization(torch.autograd.Function):
    '''
    This class comes from https://github.com/Joeyonng/decision-rules-network/blob/master/DRNet.py
    The autograd function for the binarization activation in the Rules Layer.
    The forward function implements the equations (2) in the DR-Net paper. Note here 0.999999 is used to cancel the rounding error.
    The backward function implements the STE estimator with equation (3) in the DR-Net paper.
    '''
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = (x > 0.999999).float()
        return output 

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[(x < 0)] = 0
        grad_x[(x > 1) * (grad_output < 0)] = 0
        return grad_x

def train(model, X, Y, X_val, Y_val, batch_size, nbOutput, learning_rate=1e-2, lambda_and=1e-3, epochs=3000, device="cpu", callback=False, l2_lambda=0):    
    """
    Args:
        model (Network): the neural network that has to be trained
        X (np array): the training data
        Y (np array): the training labels
        X_val (np array): the validation data
        Y_val (np array): the validation labels
        batch_size (int): the training batch size
        nbOutput (int): the number of output classes
        learning_rate (float): the learning rate of the training
        lambda_and (float): the scaling factor for the AND layer regularization
        epochs (int): the number of training epochs
        device (string): the device to run the model on
        callback (bool): whether callback on the validation data is performed
        l2_lambda (float): the scaling factor of the L2 regularization
    """
    model.to(device)
    model.train()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.LongTensor(Y)
    x_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.LongTensor(Y_val)
    training_set = torch.utils.data.TensorDataset(x, y)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    print(model.choice.weight.data)
    
    
    torch.set_printoptions(precision=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss().to(device)
    
    # Compute and print loss
    y_pred, _ = model(x)
    l2 = sum(torch.linalg.norm(p,2) for p in model.output.parameters())
    loss = criterion(y_pred, y.float()) + lambda_and*model.regularization() + l2_lambda*l2
    print("init loss", loss.item())
    
    hamming = hamming_loss(y, (y_pred.detach()>=0.5).int())
    print('init hamming', hamming)
    print_rules(model)
    best_loss = sys.maxsize
    best_epoch = -1
    best_w = None
    best_loc = None
    best_label = None
    
    for t in range(epochs):      
        batch_losses = []
        batch_hamming = []
                
        for inputs, lab in training_loader:
            inputs = inputs.to(device)
            lab_onehot = lab.to(device).float()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred, _ = model(inputs)
    
            l2 = sum(torch.linalg.norm(p,2) for p in model.output.parameters()) 
            loss = criterion(y_pred, lab_onehot) + lambda_and * model.regularization() + l2_lambda*l2
            
            batch_losses.append(loss.item())
            hamming = hamming_loss(lab, (y_pred>=0.5).int())
            batch_hamming.append(hamming)
        
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_hamming = torch.Tensor(batch_hamming).mean().item()
        epoch_loss = torch.Tensor(batch_losses).mean().item()
        
        if callback:
            model.eval()
            glob_pred_val, _ = model(x_val)
            glob_pred_train, _ = model(x)
            model.train()
            
            y_onehot_val = y_val.float()
            l2 = sum(torch.linalg.norm(p,2) for p in model.output.parameters())
            glob_loss_val = criterion(glob_pred_val, y_onehot_val) + lambda_and * model.regularization() + l2_lambda*l2
            glob_loss_val = glob_loss_val.item()
            glob_hamming_val = hamming_loss(y_val, (glob_pred_val>=0.5).int())
            y_onehot_train = y.float()
            glob_loss_train = criterion(glob_pred_train, y_onehot_train) + lambda_and * model.regularization() + l2_lambda*l2
            glob_loss_train = glob_loss_train.item()
            glob_hamming_train = hamming_loss(y, (glob_pred_train>=0.5))
            
            if glob_loss_val < best_loss:
                best_loss = glob_loss_val
                best_epoch = t
                best_w = model.and_layer.weight.data.detach().clone()
                best_loc = model.and_layer.loc.data.detach().clone()
                best_label = model.output.weight.data.detach().clone()
        if t % 250 == 0 or t == epochs-1: print(t, "loss =", epoch_loss, "hamming =", 1-epoch_hamming, "glob_train_loss =", glob_loss_train, "glob_train_hamming =", 1-glob_hamming_train, "glob_val_loss =", glob_loss_val, "glob_val_hamming =", 1-glob_hamming_val)
    model.to('cpu')
    if callback:
        model.and_layer.weight.data = best_w
        model.and_layer.loc.data = best_loc
        model.output.weight.data = best_label
    model.eval()
    nbCond = np.sum(model.and_layer.masked_weight().data.numpy() != 0)
    return nbCond, best_epoch
    
def predict(model, X):
    x = torch.tensor(X, dtype=torch.float)
    pred, _ = model(x)
    return (pred>=0.5).int()

def print_rules(model):
    for j, r in enumerate(model.and_layer.masked_weight()):
        txt = str(j)+": IF "
        for i,attr in enumerate(r):
            if attr > 0: txt += ""+str(i) + " "
            elif attr < 0: txt += "NOT "+str(i) + " "
        txt += "THEN "+ str(np.where(model.output.weight.data[:,j]>=0.5)[0])
        print(txt)
    print("ELSE "+str(np.where(model.output.weight.data[:,-1]>=0.5)[0]))

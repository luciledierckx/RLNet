import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import sys

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
    The architecture of our RLNet for binary and multi-class classification
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
            self.output.weight.requires_grad = False
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
        self.output.activation = nn.Softmax(dim=1)
    
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

def train(model, X, Y, X_val, Y_val, batch_size, nbOutput, learning_rate=1e-2, lambda_and=1e-3, epochs=3000, device="cpu", callback=False, class_weights=False, limit=1000, l2_lambda=0):
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
        class_weights (bool): whether to start the training with a balanced loss or not
        limit (int): the number of epoch the balanced loss should be used
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
    
    torch.set_printoptions(precision=6)
    if not class_weights: criterion = torch.nn.CrossEntropyLoss().to(device)
    else: 
        cl_weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
        criterion = torch.nn.CrossEntropyLoss(weight= torch.Tensor(cl_weights)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Compute and print loss
    y_pred, _ = model(x)
    l2 = sum(torch.linalg.norm(p,2) for p in model.output.parameters())
    loss = criterion(y_pred, nn.functional.one_hot(y, num_classes=nbOutput).float()) + lambda_and*model.regularization() + l2_lambda*l2
    print("init loss", loss.item())
    y_predr = torch.argmax(y_pred, dim=1)
    acc = accuracy_score(y, y_predr)
    print('init acc', acc)
    print_rules(model)
    print(model.and_layer.loc[0])
    best_loss = sys.maxsize
    best_epoch = -1
    best_w = None
    best_loc = None
    best_label = None
    
    for t in range(epochs):
        if t == limit and class_weights:
            criterion = torch.nn.CrossEntropyLoss().to(device)
        
        batch_losses = []
        batch_acc = []
                
        for inputs, lab in training_loader:
            inputs = inputs.to(device)
            lab_onehot = nn.functional.one_hot(lab.to(device), num_classes=nbOutput).float()
            # Forward pass: Compute predicted y by passing x to the model
            y_pred, else_activation = model(inputs)
            err = (lab_onehot-y_pred)**2
            err = torch.mul(else_activation,torch.transpose(err,0,1))
            l2 = sum(torch.linalg.norm(p,2) for p in model.output.parameters())                
            loss = criterion(y_pred, lab_onehot) + lambda_and*model.regularization() + l2_lambda*l2
                
                
            batch_losses.append(loss.item())
            y_predr = torch.argmax(y_pred, dim=1)
            acc = accuracy_score(lab, y_predr)
            batch_acc.append(acc)
        
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        epoch_acc = torch.Tensor(batch_acc).mean().item()
        epoch_loss = torch.Tensor(batch_losses).mean().item()
                
        if callback:                
            model.eval()
            glob_pred_val, _ = model(x_val)
            glob_pred_train, _ = model(x)
            model.train()
            
            y_onehot_val = nn.functional.one_hot(y_val, num_classes=nbOutput).float()
            l2 = sum(torch.linalg.norm(p,2) for p in model.output.parameters())
            glob_loss_val = criterion(glob_pred_val, y_onehot_val) + lambda_and * model.regularization() + l2_lambda*l2
            glob_loss_val = glob_loss_val.item()
            glob_acc_val = accuracy_score(y_val, torch.argmax(glob_pred_val, dim=1))
            y_onehot_train = nn.functional.one_hot(y, num_classes=nbOutput).float()
            glob_loss_train = criterion(glob_pred_train, y_onehot_train) + lambda_and * model.regularization() + l2_lambda*l2
            glob_loss_train = glob_loss_train.item()
            glob_acc_train = accuracy_score(y, torch.argmax(glob_pred_train, dim=1))

            if glob_loss_val < best_loss: 
                best_loss = glob_loss_val 
                best_epoch = t
                best_w = model.and_layer.weight.data.detach().clone()
                best_loc = model.and_layer.loc.data.detach().clone()
                best_label = model.output.weight.data.detach().clone()
        if t % 250 == 0 or t == epochs-1: print(t, "loss =", epoch_loss, "acc =", epoch_acc, "glob_train_loss =", glob_loss_train, "glob_train_acc =", glob_acc_train, "glob_val_loss =", glob_loss_val, "glob_val_acc =", glob_acc_val)#, "conf =", conf)#, "rule_reg =", epoch_rule_reg)
        if t == limit and class_weights: 
            print(model.and_layer.loc[0])
            print_rules(model)
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
    return torch.argmax(pred, dim=1)

def print_rules(model):
    for j, r in enumerate(model.and_layer.masked_weight()):
        txt = str(j)+": IF "
        for i,attr in enumerate(r):
            if attr > 0: txt += ""+str(i) + " "
            elif attr < 0: txt += "NOT "+str(i) + " "
        txt += "THEN "+ str(np.argmax(model.output.weight.data[:,j]))
        print(txt)
    print("ELSE "+str(np.argmax(model.output.weight.data[:,-1])))
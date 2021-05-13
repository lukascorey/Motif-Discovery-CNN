import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys
import datetime
import seqlogo
import os

LOG = True 
splits = 5 
drop_rate = 0.5 # drop rate used to prevent overfitting
learning_rate = [.0002] # learning rate, 
in_chann = 4
n_filters = 32 #
filter_size = 9 # length of motifs looking for
output_layer_number = 32 # number of layers in the second convolutional layer
d_input = 16 # changes number of neurons in linear layer
d_output = 1 # do not change
decayRate = 0.95 
max_epochs = 1000
batch_size = 128 
cutoff_value = 1 # cutoff value for sequence logo generation
mut_file = 'data/mut3.npy'
wt_file = 'data/wt3.npy'


class GeneticDataset(Dataset):

    def __init__(self,ids,preprocessed_data,labels):
        self.dataset = preprocessed_data[ids]
        self.label = labels[ids]
        self.id = ids
        
    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        label = self.label[idx]
        return sample,label



class CNN_model(nn.Module):
    def __init__(self,in_chann,n_filters,d_input,d_output,filter_size,drop_rate, output_layer_number):
        super(CNN_model,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_chann,out_channels=n_filters,stride=1,kernel_size=filter_size,padding=filter_size-1,bias=False)        
        self.l1 = nn.Linear(output_layer_number*2,d_input) 
        self.l2 = nn.Linear(d_input,d_output)
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=output_layer_number,stride=1,kernel_size=filter_size,padding=filter_size-1,bias=False)
        self.drop = nn.Dropout(drop_rate)

    def forward(self,x):
        x_conv = F.relu(self.conv1(x))
        x_conv2 = F.relu(self.conv2(x_conv)) 
        
        x_maxpool = F.max_pool1d(x_conv2,x_conv2.shape[2])
        x_avepool = F.avg_pool1d(x_conv2,x_conv2.shape[2])
        x_c= torch.cat((x_maxpool.reshape(x_maxpool.shape[0],-1),x_avepool.reshape(x_avepool.shape[0],-1)),1).view(x_avepool.shape[0],-1)
        x_c = self.drop(x_c)
        x1 = F.relu(self.l1(x_c))
        x2 = self.l2(x1)

        return x2,x_conv  


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 

    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    
    model.train()
    conv_result = []
    loss = 0
    train_acc = 0
    for train_batch, train_label in iterator:
        optimizer.zero_grad()

        prediction,x_conv = model(train_batch.permute(0,2,1))
        conv_result.append(x_conv)
        acc= binary_accuracy(prediction.view(-1),train_label)
        
        train_loss = criterion(prediction.view(-1), train_label)
        
        
        
        train_loss.backward()
        optimizer.step()
        train_acc += acc
        loss += train_loss.item()
        
    loss = loss / len(iterator)
    train_acc = train_acc / len(iterator)
    #print('loss train_dc',loss,train_dc)
    return loss,train_acc, conv_result


def evaluate(model, iterator):
    if LOG:
        logfile = open(foldername + "/log.txt", "a")
        logfile.write("start evaluation\n")
        logfile.close()
    else:
        print('start evaluation')
    conv_result = []
    
    model.eval()
    with torch.no_grad():
        valid_acc = 0
        for valid_batch, valid_label in iterator:
            prediction,x_conv = model(valid_batch.permute(0,2,1))
            
            conv_result.append(x_conv)
            acc = binary_accuracy(prediction.view(-1),valid_label)
            valid_acc+=acc
        valid_acc = valid_acc / len(iterator)

    return valid_acc,conv_result



def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)


def make_pwm(splits, lr, epochs, filter_size, cutoff_value):

    conv_relu_result_expand = torch.load(foldername + "/test_cnn.pt")
    x = []
    for i in range(len(conv_relu_result_expand)):
        for j in range(len(conv_relu_result_expand[i])): 
            x.append(conv_relu_result_expand[i][j].detach().numpy())

    conv_relu_result_expand = np.array(x)
    if LOG: 
        logfile = open(foldername + "/log.txt", "a")
        logfile.write("loading test data\n")
        logfile.close()
    else:
        print("loading test data")
    test = []

    
    for test_seq,test_label in training_generator:
        test.append(test_seq) 

    new_test = []
    for i in range(len(test)):
        for j in range(len(test[i])):
            new_test.append(test[i][j])

    
    if LOG:
        logfile = open(foldername + "/log.txt", "a")
        logfile.write(str(len(new_test)) + " -- test length\n")
        logfile.write(str(len(conv_relu_result_expand)) + "-- conv_relu_result_expand length\n")
        logfile.write(str(len(x[0])) + " filters")
        logfile.close()
    else:
        print(len(new_test), " -- test length")
        print(len(conv_relu_result_expand), "-- conv_relu_result_expand length")
        print(str(len(x[0])) + " filters")

    seq_filter = []
    
    for f in range(len(x[0])):
        c=0
        if LOG:
            logfile = open(foldername + "/log.txt", "a")
            logfile.write("processing filter " + str(f) + "\n")
            logfile.close()
        else:
            print('processing filter',f)
        filter_f = []
        for i in range(len(new_test)):
            filter_i_f = conv_relu_result_expand[i][f]#.detach().numpy()

            m = max(filter_i_f)
            if m>cutoff_value: 
                index = filter_i_f.tolist().index(m)
            

                seq = new_test[i]
                if index<len(seq)-filter_size: 
                    filter_f.append(seq[index:index+filter_size])
                    c+=1

        seq_filter.append(filter_f)
        if LOG: 
            logfile = open(foldername + "/log.txt", "a")
            logfile.write("total motifs found: " + str(c) + "\n")
            logfile.write("Dimensions: " + str(len(seq_filter)) + ", " + str(len(seq_filter[0])) + ", " + str(len(seq_filter[0][0])) + "\n") 
            logfile.close()
        else:
            print('a total of motifs found',c)
            print("dimensions: ")
            print(len(seq_filter))
            print(len(seq_filter[0]))
            print(len(seq_filter[0][0]))

    pfm_all = []
    for i in range(len(x[0])): 
        filter_i = seq_filter[i]
        pfm = np.zeros((filter_size,4)) 
        for f in filter_i:
            pfm+=f.detach().numpy()
        pfm_all.append(pfm)

    newstr = foldername + "/pfms/pfms_" + str(splits) + "_" + str(lr) + "_" + str(epochs)
    os.system("mkdir " + newstr)
    out = open((newstr + "/pfms_" + str(splits) + "_" + str(lr) + "_" + str(epochs) + ".txt"), "w+") 
    out.write(str(pfm_all))
    out.close()

    for i in range(len(pfm_all)): 
        x = pfm_all[i]
        try:
            pwm = seqlogo.pfm2pwm(x)
            pwm = seqlogo.Ppm(seqlogo.pwm2ppm(pwm))
            
            seqlogo.seqlogo(pwm, ic_scale = False, format = 'png', size = 'medium', filename=newstr + "/logo" + str(i) + ".png")
        except: 
            print("cannot make logo " + str(i))



if __name__ == "__main__":
    
    mt = np.load(mut_file)
    wt = np.load(wt_file)

    foldername = "outputs/" + str(round(datetime.datetime.now().timestamp()))
    os.system("mkdir " + foldername)
    os.system("mkdir " + foldername + "/pfms")

    if LOG:
        logfile = open(foldername + "/log.txt", "w+")
        logfile.write(" ".join(sys.argv) + "\n")
        logfile.write(str(round(datetime.datetime.now().timestamp())) + "\n")
        logfile.close()

    mixed = np.concatenate((mt,wt),0)
    mixed_label = [0]*mt.shape[0] + [1]*wt.shape[0]


    ids = np.arange(mixed.shape[0])
    np.random.shuffle(ids)


    shuffled_data = [mixed[i] for i in ids]
    shuffled_label = [mixed_label[i] for i in ids]


    shuffled_data = np.array(shuffled_data,dtype=np.float32)
    shuffled_label = np.array(shuffled_label,dtype=np.float32)

    n_sample = shuffled_data.shape[0]
    train_split_index,test_split_index = Kfold(n_sample,splits)


    for k in range(splits):
        if LOG: 
            logfile = open(foldername + "/log.txt", "a")
            logfile.write("current split is " + str(k) + "\n")
            logfile.close()
        else:
            print('current split is',k)

        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]
        
        training_set = GeneticDataset(train_index,shuffled_data,shuffled_label)
        training_generator = DataLoader(training_set, batch_size=batch_size,shuffle=True)

        validation_set = GeneticDataset(valid_index,shuffled_data,shuffled_label)
        validation_generator = DataLoader(validation_set, batch_size=batch_size,shuffle=True)
        
        test_set = GeneticDataset(test_index,shuffled_data,shuffled_label)
        test_generator = DataLoader(test_set,  batch_size=batch_size,shuffle=False)
        
        best_ac = 0
        
        for l_r in learning_rate:
            if LOG:
                logfile = open(foldername + "/log.txt", "a")
                logfile.write("learning rate is " + str(l_r) + "\n") 
                logfile.close()
            else: 
                print('learning rate is', l_r)
            
            model = CNN_model(in_chann,n_filters,d_input,d_output,filter_size,drop_rate, output_layer_number)
            model = model.float()
            optimizer = optim.Adam(model.parameters(), lr=l_r,weight_decay=1e-5) # started 1e-5
            criterion = nn.BCEWithLogitsLoss()        
            # Loop over epochs
            for epoch in range(max_epochs):
                if epoch%5 == 0:
                    if LOG: 
                        logfile = open(foldername + "/log.txt", "a")
                        logfile.write("epoch: " + str(epoch) + " \n")
                        logfile.close()
                    else: 
                        print(epoch)
                train_loss,train_ac,conv_result = train(model, training_generator, optimizer, criterion)
                if epoch%100 == 0:
                    valid_ac,_ = evaluate(model, validation_generator)
                    if LOG:
                        logfile = open(foldername + "/log.txt", "a")
                        logfile.write("epoch : {}/{}, loss = {:.6f}\n".format(epoch + 1, max_epochs, train_loss))
                        logfile.write('train ac is ' + str(train_ac.item()) + "\n")
                        logfile.write('valid ac is ' + str(valid_ac.item()) + "\n")
                        logfile.close()
                    else:
                        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, train_loss))
                        print('    train ac is ',train_ac.item())
                        print('valid ac is',valid_ac.item())

            torch.save(conv_result, foldername + "/test_cnn.pt")
            test_ac,conv_result = evaluate(model, test_generator) 
            if LOG:
                logfile = open(foldername + "/log.txt", "a")
                logfile.write('    train ac is ' + str(train_ac.item()) + "\n")
                logfile.write('test ac is ' + str(test_ac.item()) + "\n")
                logfile.close()
            else:
                print('test ac is ',test_ac.item())
        make_pwm(k, l_r, max_epochs,filter_size, cutoff_value)


    if LOG:
        logfile = open(foldername + "/log.txt", "a")
        logfile.write(str(round(datetime.datetime.now().timestamp())) + "\n")
        logfile.close()
from utils import opt, torch,nn
import argparse
from dataloaders import DataLoader
from model import Alexnet, Mobilenet
import random

class Train:
    # pass
    def __init__(self, num_epoch , lr, pin_memory, model_name):
        self.num_epochs = num_epoch
        self.pin_memory = pin_memory
        self.learning_rate = lr
        self.loss = nn.BCELoss()
        self.data_l = DataLoader()
        self.net = model_name
        self.alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r','s', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def process(self, train_dir, test_dir, batch_size, shuffle, num_worker, save_model):
        ### deta prepared
        train_loader, _ = self.data_l.prepare_data(train_dir, test_dir, batch_size, shuffle, num_worker)
        ## defining optimize
        opt = opt.Adam(self.net.paramters(), lr = self.learning_rate)
        rand_int = random.randint(1,999999)
        for epoch in range(self.num_epochs):
            total_loss = 0

            for i,batch in enumerate(train_loader):
                images, label = batch
                pred = self.net(images)
                loss = self.loss(pred, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss+=loss.item()

            print("Epoch", epoch, "loss", total_loss)
            
        save_model_name = str(rand_int)+self.alpha
        torch.save(self.net, f'{save_model}/{save_model_name}.pt')
        # self.net.save_model(f"{save_model}/{i}.pth")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Training model hyperparameter')
    parser.add_argument('--epochs', type=int, default=10  , help='pass num of epoch')
    parser.add_argument('--lr', default=0.001 ,help='pass learning rate')
    parser.add_argument('--batch_size', type=int , default=4,  help='pass batch_size')
    parser.add_argument('--shuffle', type=bool , default=True ,help='whether to shuffle the data for trianing or not set True or False')
    parser.add_argument('--num_worker', default=1 ,help='set num_worker to increase training speed')
    parser.add_argument('--train_dir',default='./data/train_dir/' , help='pass training directory')
    parser.add_argument('--test_dir', default='./data/test_dir/' , help='pass test data directory')
    parser.add_argument('--model_name', default='Alexnet' , help='pass model name on which it is train')
    parser.add_argument('--save_model', default='./' , help='location to save model')

    args = parser.parse_args()

    train_obj = Train(args.epochs, args.lr, args.pin_memory, args.model_name)
    train_obj.process(args.train_dir, args.test_dir, args.batch_size, args.shuffle, args.num_worker , args.save_model)

    
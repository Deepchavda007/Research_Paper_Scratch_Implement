from utils import opt, torch,nn
import argparse
from dataloaders import DataLoader
from model import Alexnet, Mobilenet
from sklearn.metrics import f1_score

class Test:
    # pass
    def __init__(self, model_path):
        self.data_l = DataLoader()
        self.net = torch.load(model_path)
        self.net.eval()

    def process(self, train_dir, test_dir, batch_size, num_worker,):
        ### deta prepared
        _, test_data = self.data_l.prepare_data(train_dir, test_dir, batch_size, num_worker)

        with torch.no_grad():
            preds = []
            ground_truth = []
            correct = 0
            for i, batch in enumerate(test_data):
                images, label = batch
                pred = self.net(images)
                _, predicted = torch.max(pred, 1)
                n_correct += (predicted == label).sum().item()
                preds.append(predicted)
                ground_truth.append(label)

            f1_score = f1_score(preds, ground_truth)

            print("F1_score is ", f1_score)
            print("Total_correct ", correct, "out of total ", len(test_data))

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Training model hyperparameter')
    parser.add_argument('--batch_size', type=int , default=1,  help='pass batch_size')
    parser.add_argument('--num_worker', default=1 ,help='set num_worker to increase training speed')
    parser.add_argument('--test_dir', default='' , help='pass test data directory')
    parser.add_argument('--test_img', default='./data/test_dir/1.png' , help='pass test image')
    parser.add_argument('--model_path', default='Alexnet' , help='pass model path to get prediction')

    args = parser.parse_args()

    train_obj = Test(args.model_path)
    if args.test_dir:
        train_obj.process('' ,args.test_dir, args.batch_size, args.num_worker , args.save_model)
    
    if args.test_img:
        train_obj.process('' ,args.test_img, args.batch_size, args.num_worker , args.save_model)


    
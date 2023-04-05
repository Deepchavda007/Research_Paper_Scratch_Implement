from utils import DataLoader, Dataset, datasets
from utils import transforms

class Dataloader:
    def __init__(self):
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    def prepare_data(self, train_dir, test_dir, batch_size, shuffle, num_worker):
        if train_dir:
            train_data = datasets.ImageFolder(root=train_dir, 
                                          transforms = self.transforms)
        
            # print("len of train_data", len(train_data))
            train_dataloader = DataLoader(train_data, batch_size = batch_size, num_worker = num_worker, shuffle=shuffle)
        else:
            train_data = ''

        if test_data:
            test_data = datasets.ImageFolder(root=test_dir, 
                                          transforms = self.transforms)
        
            # print("len of test_data", len(test_data))
            test_dataloader = DataLoader(test_data, batch_size = batch_size, num_worker = num_worker, shuffle=shuffle)
        else:
            test_data = ''

        # print("train dataloader ", train_dataloader)
        # print("test dataloader ", test_dataloader)

        return train_dataloader, test_dataloader
        
        

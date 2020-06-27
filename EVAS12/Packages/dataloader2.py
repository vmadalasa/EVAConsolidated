import torch
import torchvision
from torchvision import datasets, transforms

class Imagenet_data_loader:
    def __init__():
        super().__init__()
    
    def data_transform_train():
        data_transform_train = transforms.Compose([
                                                    transforms.RandomRotation(5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4770698,  0.44741154, 0.3993873 ],
                                                    std=[0.27973273, 0.27094352, 0.28073433])
        ])
        return data_transform_train

    def data_transform_test():
        data_transform_test = transforms.Compose([
                                                    #transforms.RandomRotation(5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4770698,  0.44741154, 0.3993873 ],
                                                    std=[0.27973273, 0.27094352, 0.28073433])
        ])
        return data_transform_test

    def image_folder (folder_path, transform=None):
        return datasets.ImageFolder(root=folder_path,
                                           transform=transform)
                                           
    def data_loader (dataset, batch_size, num_workers, shuffle=False):
        return torch.utils.data.DataLoader (dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

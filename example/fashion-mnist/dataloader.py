
import numpy as np
import pickle

from context import image_classifier
from image_classifier.dataloader import Dataloader

if __name__ == '__main__':

    NORMALIZED_DATA_PATH = './example/fashion-mnist/data/normalized_data.pkl'

    with open(NORMALIZED_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    trainloader = Dataloader().data_loader(data['train_x'], data['train_y'], batch_size=64)
    devloader = Dataloader().data_loader(data['valid_x'], data['valid_y'], batch_size=1000)
    testloader = Dataloader().data_loader(data['test_x'], data['test_y'], batch_size=1000) 

    dataloaders = {
        "trainloader": trainloader,
        "devloader": devloader,
        "testloader": testloader,
    }

    with open('./example/fashion-mnist/data/dataloaders.pkl', 'wb') as f:
        pickle.dump(dataloaders, f, pickle.HIGHEST_PROTOCOL)
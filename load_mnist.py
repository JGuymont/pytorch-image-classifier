import pickle
import numpy as np

def load_mnist():
    """
    load MNIST data and split into
    predefined train/dev/test
    """
    pkl_in = open("./data/mnist.pkl.npy", "rb")
    data = pickle.load(pkl_in)

    _train_x = np.asarray(data[0][0])
    _train_y = np.asarray(data[0][1])
    print(_train_y)
    print(len(_train_y))
    _valid_x = np.asarray(data[1][0])
    _valid_y = np.asarray(data[1][1])
    _test_x = np.asarray(data[2][0])
    _test_y = np.asarray(data[2][1])
    return [_train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y]
        
if __name__ == '__main__':
    load_mnist()

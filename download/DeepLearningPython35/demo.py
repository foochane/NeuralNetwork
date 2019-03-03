import network
import mnist_loader
import numpy as np
import h5py




training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data = list(training_data)

# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)




# def vectorized_result(j):
#     e = np.zeros((10, 1))
#     e[j] = 1.0
#     return e
    
# train_dataset = h5py.File('cat_datasets/train_catvnoncat.h5', "r")
# train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
# train_set_y_orig = np.array(train_dataset["train_set_y"][:])

# test_dataset = h5py.File('cat_datasets/test_catvnoncat.h5', "r")
# test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
# test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

# m_train = train_set_x_orig.shape[0]
# m_test = test_set_y_orig.shape[0]

# train_set_x = train_set_x_orig.reshape(m_train,-1) #这里reshape第一个值设-1时会根据后面那个值自动调整
# test_set_x = test_set_x_orig.reshape(m_test,-1)

# train_set_y = train_set_y_orig.reshape((m_train,-1))
# test_set_y = test_set_y_orig.reshape((m_test,-1))


# training_inputs = [np.reshape(x, (12288, 1)) for x in train_set_x]
# training_results = [vectorized_result(y) for y in train_set_y]
# training_data = zip(training_inputs, training_results)

# test_inputs = [np.reshape(x, (12288, 1)) for x in test_set_x]
# test_data = zip(test_inputs, test_set_y)

# net = network.Network([12288, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



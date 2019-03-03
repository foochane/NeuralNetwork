# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    返回：
    training_data: 2元元组；training_data[0]是一个shape为(50000, 784)numpy数组，包含50000个图像数据；
                    training_data[1]是一个shape为(50000,)的numpy数组，表示0~9中的一个数字
    validation_data:2元元组；validation_data[0]是一个shape为(10000, 784)numpy数组，包含10000个图像数据；
                    validation_data[1]是一个shape为(10000,)的numpy数组，表示0~9中的一个数字
    test_data:     2元元组；test_data[0]是一个shape为(50000, 784)numpy数组，包含10000个图像数据；
                    test_data[1]是一个shape为(50000,)的numpy数组，表示0~9中的一个数字
    
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    # print("training_data type:",type(training_data))
    # print("training_data size:",len(training_data))
    # print("training_data[0] type:",type(training_data[0]))
    # print("training_data[0] shape:",training_data[0].shape)
    # print("training_data[1] type:",type(training_data[1]))
    # print("training_data[1] shape:",training_data[1].shape)

    # print("validation_data type:",type(validation_data))
    # print("validation_data size:",len(validation_data))
    # print("validation_data[0] type:",type(validation_data[0]))
    # print("validation_data[0] shape:",validation_data[0].shape)
    # print("validation_data[1] type:",type(validation_data[1]))
    # print("validation_data[1] shape:",validation_data[1].shape)

    # print("test_data type:",type(test_data))
    # print("test_data size:",len(test_data))
    # print("test_data[0] type:",type(test_data[0]))
    # print("test_data[0] shape:",test_data[0].shape)
    # print("test_data[1] type:",type(test_data[1]))
    # print("test_data[1] shape:",test_data[1].shape)

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.
    
    返回：元组：(training_data, validation_data, test_data)
    training_data: zip类型,使用时先转换为list，training_data = list(training_data)；
                    training_data 包含5000个元组，每个元组包含一副图像的数据和标签(x,y)
                    training_data[0] type: <class 'tuple'>
                    training_data[0][0] shape: (784, 1)  --->x
                    training_data[0][1] shape: (10, 1)  ----y(10维向量)
    validation_data: zip类型；使用时先转换为list，training_data = list(training_data)；
                    training_data 包含1000个元组，每个元组包含一副图像的数据和标签(x,y),这里的y是数字
                    validation_data type: <class 'zip'>
                    validation_data type: 10000
                    validation_data[0] type: <class 'tuple'>
                    validation_data[0][0] shape: (784, 1)
                    validation_data[0][1] type: <class 'numpy.int64'>
    test_data:      zip类型；使用时先转换为list，test_data = list(test_data)；
                    test_data 包含1000个元组，每个元组包含一副图像的数据和标签(x,y),这里的y是数字
                    test_data type: <class 'zip'>
                    test_data type: 10000
                    test_data[0] type: <class 'tuple'>
                    test_data[0][0] shape: (784, 1)
                    test_data[0][1] type: <class 'numpy.int64'>    
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    # print("training_data type:",type(training_data))
    # training_data = list(training_data)
    # print("training_data type:",len(training_data))
    # print("training_data[0] type:",type(training_data[0]))
    # print("training_data[0][0] shape:",training_data[0][0].shape)
    # print("training_data[0][1] shape:",training_data[0][1].shape)

    # print("validation_data type:",type(validation_data))
    # validation_data = list(validation_data)
    # print("validation_data type:",len(validation_data))
    # print("validation_data[0] type:",type(validation_data[0]))
    # print("validation_data[0][0] shape:",validation_data[0][0].shape)
    # print("validation_data[0][1] type:",type(validation_data[0][1]))

    # print("test_data type:",type(test_data))
    # test_data = list(test_data)
    # print("test_data type:",len(test_data))
    # print("test_data[0] type:",type(test_data[0]))
    # print("test_data[0][0] shape:",test_data[0][0].shape)
    # print("test_data[0][1] type:",type(test_data[0][1]))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

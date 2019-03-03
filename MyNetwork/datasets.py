import numpy as np
import h5py
import matplotlib.pyplot as plt    
    
def load_dataset_orig():
    '''
    导入图片数据
    每张图片的大小 : (64, 64, 3)
    train_set_x_orig:训练集原始数据，维度为(209, 64, 64, 3)，表示209张64x64的图片
    train_set_y_orig：训练集标签，维度为(209,)，0表示是不是猫，1表示是猫
    test_set_x_orig：测试集原始数据: (50, 64, 64, 3)，表示50张64x64的图片
    test_set_y_orig: 测试集标签，维度(50,)，0表示是不是猫，1表示是猫
    '''
    train_dataset = h5py.File('cat_datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('cat_datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def load_dataset_wrapper():
    '''
    数据预处理
    训练集的数量: m_train = 209
    测试集的数量 : m_test = 50
    train_set_x :训练集数据，维度为(12288, 209)
    train_set_y：训练集标签，维度为(1,209)，0表示是不是猫，1表示是猫
    test_set_x_orig：测试集原始数据: (12288,50)，表示50张64x64的图片
    test_set_y_orig: 测试集标签，维度(1,50)，0表示是不是猫，1表示是猫
    '''
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig  = load_dataset_orig()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_y_orig.shape[0]

    train_set_x = train_set_x_orig.reshape(-1,m_train) #这里reshape第一个值设-1时会根据后面那个值自动调整
    test_set_x = test_set_x_orig.reshape(-1,m_test)

    train_set_y = train_set_y_orig.reshape((-1, m_train))
    test_set_y = test_set_y_orig.reshape((-1, m_test))
    
    return m_train, train_set_x, train_set_y, m_test, test_set_x, test_set_y



# #测试
# train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig  = load_dataset_orig()
# #查看前64张图片
# index = 0
# while(index<64):
#     plt.xticks([]) # 不显示坐标轴
#     plt.yticks([])
#     plt.axis('off') #不显示刻度
#     plt.subplot(8,8,index+1)
#     plt.imshow(train_set_x_orig[index])
#     index+=1
# plt.show()

# m_train, train_set_x, train_set_y, m_test, test_set_x, test_set_y = load_dataset_wrapper()
# print("m_train:",m_train)
# print("train_set_x:",train_set_x.shape)
# print("train_set_y:",train_set_y.shape)
# print("m_test:",m_test)
# print("test_set_x:",test_set_x.shape)
# print("test_set_y:",test_set_y.shape)
 
import numpy as np
from datasets import load_dataset_wrapper

def initialize_parameters(sizes):
    """
    初试化参数
    weights的维度：(当前层神经元个数，前一层的神经元个数)
    biases的维度：(当前层神经元个数，1)

    输入：sizes
        如sizes=[2,3,1],表示一个输入为2维的2层神经网络，
        隐藏层的神经元个数分别为3,1

    返回：weights,biases
        weights = [[W1],W2,...] : W1,W2,...为矩阵
        biases = [[b1],[b2],...] ：b1,b2,...为向量
    """
    # 初试为均值为0，方差为1的高斯分布
    weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    biases = [np.random.randn(y, 1) for y in sizes[1:]]  
    return  weights,biases

def feedforward(x,W,b):
    """
    # 前项传播
    参数：
        A - 来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）
        W - 权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）
        b - 偏向量，numpy向量，维度为（当前图层节点数量，1）

    返回：
         A
    """
    activation = x  # 第0层的激活值A^[0]
    zs = [] # 存储所有的Z
    activations = [] # 存储所有的激活值，A^[0],A^[1],......

    activations.append(activation) 
    
    for w,b in zip(W, b):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    return activations[-1]

def backprop(x, y,weights,biases,sizes):
        """
        反向传播算法
        返回偏导数
        """
        num_layers = len(sizes)

        m = len(x[0])  # 输入m组数据
        learning_rate = 0.1 # 学习率

        dW = [np.zeros(w.shape) for w in weights]
        db = [np.zeros(b.shape) for b in biases]

        # 前项传播
        activation = x  # 第0层的激活值A^[0]
        zs = [] # 存储所有的Z
        activations = [] # 存储所有的激活值，A^[0],A^[1],......

        activations.append(activation) 
        
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # print("zs:",zs)
        # print("activations:",activations)
        


        # 反向传播
        dZ = cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        dW[-1] = np.dot(dZ, activations[-2].transpose())
        db[-1] = dZ

        for l in range(2, num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            dZ = np.dot(weights[-l+1].transpose(), dZ) * sp
            
            dW[-l] = np.dot(dZ, activations[-l-1].transpose())
            db[-l] = dZ

        #更新参数
        weights = [W_temp-(learning_rate/m)*dW_temp for W_temp,dW_temp in zip(weights,dW)]
        biases = [b_temp-(learning_rate/m)*db_temp for b_temp,db_temp in zip(biases,db)]


        return  weights,biases

def predict(x, y,weights,biases):
        # test_results = [(np.argmax(feedforward(x,weights,biases)), y)
        #                 for (x, y) in zip(x,y)]
        # return sum(int(x == y) for (x, y) in test_results)/len(x)

        y_hat=np.zeros(y.shape)
        A =feedforward(x,weights,biases)

        for i in range(len(A[0])):
            if(A[0][i]>0.5):
                y_hat[0][i] = 1
        # print(y)
        # print(y_hat)
        return (sum(int(y1 == y2) for (y1, y2) in zip(y_hat[0],y[0]))/len(y[0])) 


def cost_derivative(output_activations, y):
    """
    返回第L层：dL(a,y)/dz
    """
    return (output_activations-y)

def sigmoid(z):
    """
    sigmoid 函数.
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    """sigmoid 函数的导数."""
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    """
    RELU 函数
    """
    return np.maximum(0,z)


# 定义神经网络
sizes = [12288,7,9,1]  

#初始化参数
w,b = initialize_parameters(sizes) 

# 加载数据
m_train, train_set_x, train_set_y, m_test, test_set_x, test_set_y = load_dataset_wrapper()

print("train_set_x:",train_set_x.shape)
print("train_set_y:",train_set_y.shape)
for i in range(1000):
    w ,b = backprop(test_set_x, test_set_y,w,b,sizes)
    if i%10==0:
        print("第",i,"轮：",predict(test_set_x, test_set_y,w,b))

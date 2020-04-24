import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation, DirichletBCPoint, numBatches):
    '''
    :param domain: The domain of the definition of the equation.
    :param n: Discretize the domain into n grid points.
    :param realSolution: The true solution of the equation.
    :param StructureOfNeuralNetwork = [n1,...,ni,...,no]:
            Number of layers of neural network is len(StructureOfNeuralNetwork)
            n1: Number of neurons in input layer, whose value is equal to the number of variables in the equation
            ni: The number of neurons in the ith hidden layer, whose value is selected according to the
                complexity and oscillation of the equation.
            no: The number of neurons in the output layer must be 1.
    :param ImplicitSchemeOfEquation: Implicit scheme of differential equation.
    :param DirichletBCPoint = [x1,x2,x3,x4,x5]: There are at most five Dirichlet boundary conditions,
                        that is, at most five order differential equations are supported.
    :param numBatches: Number of training iterations.
    :return y_output: The numerical solution predicted by Deep Learning is returned in the form of row vector.
    '''


    #创建互动会话
    sess = tf.InteractiveSession()


    #构造训练数据:将定义域domain离散为n个均匀网格点
    x_train = np.linspace(domain[0], domain[1], n, endpoint=True)
    '''
    #二维区域均匀离散
    a = np.linspace(0,1,100,endpoint=True)
    b = a[:]
    res = []
    for i in a:
        for j in b:
            res.append([i,j])
    x_train = np.array(res)
    '''


    #数据预处理
    for x in DirichletBCPoint:
        if x not in x_train:                                    #防止边界点不在离散的x_train中
            x_train = list(x_train)
            x_train.append(x)                                   #为离散定义域添加边界点
            x_train.pop(n//2)                                   #同时需要删除一个点,以保持离散点的个数恒为n
            x_train.sort()
            x_train = np.array(x_train)
    y_real = realSolution(x_train)                              #真解,用于求误差
    #将行向量x_train转换为列向量x_t
    x_t = np.zeros((len(x_train), 1))                           #声明x_t为len(x_train)行1列的数组;()也可用[]代替
    for i in range(len(x_train)):                               #赋值x_t
        x_t[i] = x_train[i]


    #构造神经网络
    n = StructureOfNeuralNetwork
    if len(n) == 3:                                             #如果设置为3层神经网络
        #输入层
        x0 = tf.placeholder(tf.float32, [None, n[0]])           #输入层有n[0]个神经元,n[0]为方程变量的个数,None代表minibath每轮输入数目不限
        #声明从输入层到隐含层一的权和偏置
        W0 = tf.Variable(tf.random_normal([n[0], n[1]]))        #声明权矩阵(输入层只有1个神经元,隐含层一含n[1]个神经元)
        b0 = tf.Variable(tf.random_normal([n[1]]))              #声明偏置,tf.random_normal()从服从指定正太分布的数值中取出随机数
        #隐含层一的输出
        y1 = tf.nn.sigmoid(tf.matmul(x0, W0) + b0)              #经过试验,使用tf.sin()作为激活函数不如tf.nn.sigmoid效果要好
        #声明从隐含层一到输出层的权和偏置
        W1 = tf.Variable(tf.random_normal([n[1], n[2]]))        #隐含层含n[1]个神经元,输出层只有1个神经元(即n[-1]=1)
        b1 = tf.Variable(tf.random_normal([n[2]]))
        #输出层的输出
        y = tf.matmul(y1, W1) + b1                              #试验证明输出层施加激活函数效果极差
    elif len(n) == 4:                                           #如果设置为4层神经网络
        #输入层
        x0 = tf.placeholder(tf.float32, [None, n[0]])
        #声明从输入层到隐含层一的权重和偏置
        W0 = tf.Variable(tf.random_normal([n[0], n[1]]))
        b0 = tf.Variable(tf.random_normal([n[1]]))
        #隐含层一的输出
        y1 = tf.nn.sigmoid(tf.matmul(x0, W0) + b0)
        #声明从隐含层一到隐含层二的权和偏置
        W1 = tf.Variable(tf.random_normal([n[1], n[2]]))
        b1 = tf.Variable(tf.random_normal([n[2]]))
        #隐含层二的输出
        y2 = tf.nn.sigmoid(tf.matmul(y1, W1) + b1)
        # 声明从隐含层二到输出层的权和偏置
        W2 = tf.Variable(tf.random_normal([n[2], n[3]]))
        b2 = tf.Variable(tf.random_normal([n[3]]))
        #输出层的输出
        y = tf.matmul(y2, W2) + b2
    elif len(n) == 5:                                           #如果设置为5层神经网络
        #输入层
        x0 = tf.placeholder(tf.float32, [None, n[0]])
        #声明从输入层到隐含层一的权重和偏置
        W0 = tf.Variable(tf.random_normal([n[0], n[1]]))
        b0 = tf.Variable(tf.random_normal([n[1]]))
        #隐含层一的输出
        y1 = tf.nn.sigmoid(tf.matmul(x0, W0) + b0)
        #声明从隐含层一到隐含层二的权和偏置
        W1 = tf.Variable(tf.random_normal([n[1], n[2]]))
        b1 = tf.Variable(tf.random_normal([n[2]]))
        #隐含层二的输出
        y2 = tf.nn.sigmoid(tf.matmul(y1, W1) + b1)
        #声明从隐含层二到隐含层三的权和偏置
        W2 = tf.Variable(tf.random_normal([n[2], n[3]]))
        b2 = tf.Variable(tf.random_normal([n[3]]))
        #隐含层三的输出
        y3 = tf.nn.sigmoid(tf.matmul(y2, W2) + b2)
        #声明从隐含层三到输出层的权和偏置
        W3 = tf.Variable(tf.random_normal([n[3], n[4]]))
        b3 = tf.Variable(tf.random_normal([n[4]]))
        #输出层的输出
        y = tf.matmul(y3, W3) + b3
    else:
        print('神经网络层数设置不正确！请注意:此深度学习库只支持选择 3 ~ 5 层深的神经网络.')


    #定义损失函数
    t_loss = ImplicitSchemeOfEquation(x0, y) ** 2                   #隐式常微分方程等号左端F的平方
    #loss = tf.reduce_mean(t_loss + (y[0]+1)**2 + (y[-1]-1)**2)
    #n阶微分方程需要n个边界条件(此处只考虑狄利克雷边界条件)               #支持n阶微分方程,即支持n个边界条件
    BClossList = []                                                 #储存边界损失
    for xi in DirichletBCPoint:
        ui = realSolution(xi)
        BClossList.append((y[x_train.tolist().index(xi)]-ui) ** 2)
    loss = tf.reduce_mean(t_loss + sum(BClossList))                 #损失函数:F的平方加上边界损失的平方,再取均值


    #判断训练或直接读取模型参数文件,使用python异常处理
    try:                                                            #尝试执行读取参数文件操作
        model_file1 = open('ckpt/nn.ckpt-' + str(numBatches), 'w')  #模型参数文件保存路径
        model_file1.close()
        #当numBatches改变时,再用打开ckpt/nn.ckpt-numBatches.index'判断其是否存在,不存在则异常,执行except模块
        model_file2 = open('ckpt/nn.ckpt-'+str(numBatches)+'.index', 'r')
        model_file2.close()
    except:                                                         #如果遇到异常则执行
        #定义训练操作
        train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)    #使用自适应学习率的Adam优化器最小化损失函数
        init = tf.global_variables_initializer().run()              #执行初始化全局参数
        for i in range(numBatches):                                 #训练numBatches轮,每轮mini_batch输入len(x_t)个数据点
            sess.run(train_step, feed_dict={x0: x_t})               #每轮输入len(x_t)个点,即每输入len(x_t)个样本计算一次损失
                                                                    #反向传播一次,更新一次权,相当于每次将所有len(x_t)个离散点全部输入
            if i%100 == 0:                                          #每100轮打印一次
                total_loss = sess.run(loss, feed_dict={x0: x_t})    #填充数据,每mini-bach轮输入len(x_t)个点,即输入整个离散的x_t
                print("第%d轮的损失为:%g"%(i, total_loss))           #每训练100轮打印一次损失
                for xi in DirichletBCPoint:                         #打印边界点的预测值
                    print(sess.run(y[x_train.tolist().index(xi)], feed_dict={x0: x_t}))
                print()
        #模型参数保存
        saver = tf.train.Saver(max_to_keep=1)                       #建立一个Saver对象
        saver.save(sess, 'ckpt/nn.ckpt', global_step = numBatches)
    else:                                                           #如果未出现异常则执行
        #模型参数恢复
        #再次使用try进行异常处理是为了排除当训练完成并保存参数模型至cktp一次后
        #如果继续更改神经网络结构进行下一次训练,会出现读取保存的模型参数shape不匹配问题的情况
        try:                                                        #如果模型参数shape适合,则直接读取使用
            saver = tf.train.Saver(max_to_keep=1)
            numBatchesStr = str(numBatches)
            model_file = 'ckpt/nn.ckpt-' + numBatchesStr            #模型参数文件保存路径
            saver.restore(sess, model_file)
        except:                                                     #如果更改了神经网络结构导致上次保存的模型参数shape不符合,则重新训练
            #定义训练操作
            train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
            init = tf.global_variables_initializer().run()
            for i in range(numBatches):
                sess.run(train_step, feed_dict={x0: x_t})
                if i % 100 == 0:
                    total_loss = sess.run(loss, feed_dict={x0: x_t})
                    print("第%d轮的损失为:%g" % (i, total_loss))
                    for xi in DirichletBCPoint:
                        print(sess.run(y[x_train.tolist().index(xi)], feed_dict={x0: x_t}))
                    print()
            try:
                #模型参数保存
                #如果出现'Fail rename; Input/output error'异常,请删除上次保存的模型参数文件'ckpt',重新进行训练
                saver = tf.train.Saver(max_to_keep=1)  # 建立一个Saver对象
                saver.save(sess, 'ckpt/nn.ckpt', global_step=numBatches)
            except:
                print("如果出现'Fail rename; Input/output error'异常,请删除上次保存的模型参数文件'ckpt',重新进行训练")


    #画图
    output_y = sess.run(y, feed_dict={x0: x_t})                     #返回离散的预测值
    output_f = sess.run(t_loss, feed_dict={x0: x_t})                #返回离散的隐式方程左端F的平方
    y_output = x_train.copy()                                       #浅复制(目的是声明y_output形状与x_train相同)
    f_output = x_train.copy()

    for i in range(len(x_train)):
        y_output[i] = output_y[i]                                   #即离散的预测值
        f_output[i] = output_f[i]                                   #即离散的隐式方程左端F的平方

    fig = plt.figure("Prediction curve and Real curve")
    plt.plot(x_train, y_real, 'r-')
    plt.plot(x_train, y_output, 'k.')

    fig2 = plt.figure("y_real - y_output")                          #可视化误差
    plt.plot(x_train, abs(y_real - y_output))

    fig3 = plt.figure("Loss")                                       #可视化损失函数
    plt.plot(x_train, f_output + (y_output[0]+1)**2 + (y_output[-1]-1)**2)
    plt.show()
    return y_output.reshape([1,-1])[0]                              #返回离散的预测的数值解(重塑为行向量);[0]是为了去一层列表


#测试
#if __name__ == "__main__":
#    pass
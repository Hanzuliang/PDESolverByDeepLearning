import PDESolverByDeepLearning.hanzuliang as PDESolver
import tensorflow as tf

'''
The input and output parameters description of PDESolver function:

PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation, DirichletBC, numBatches)
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



'''
example1: First order differential equation
real solution: u(x)= 5*x**3 + x**2 + 2*x + 1
u'(x) = 15*x**2 + 2*x + 2;  u(-1) = -4;  x∈[-1,1]
'''

domain = [-1, 1]                                                                    #Domain
realSolution = lambda x: 5*x**3 + x**2 + 2*x + 1                                    #Real solution
n = 100                                                                             #Divide the domain into n sample points
#如果出现'Fail rename; Input/output error'异常,请删除上次保存的模型参数文件'ckpt',重新进行训练
StructureOfNeuralNetwork = [1, 10, 1]                                               #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(u, x)[0] - 15*x**2 - 2*x - 2   #It must be the implicit scheme of the equation,
                                                                                    #Where u is the predicted numerical solution we want to get
DirichletBCPoint = [-1]                                                             #Dirichlet boundary conditions
numBatches = 30000                                                                  #Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork,
                               ImplicitSchemeOfEquation, DirichletBCPoint, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)
print(y_output[0])

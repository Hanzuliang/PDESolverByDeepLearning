1. This operator is suitable for solving the problem of one-dimensional n-order differential equation with Dirichlet boundary conditions.


2. Input parameter description:
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation, DirichletBCPoint, numBatches):
    	:param domain: The domain of the definition of the equation.
    	:param n: Discretize the domain into n grid points.
    	:param realSolution: The true solution of the equation.
    	:param StructureOfNeuralNetwork = [n1,...,ni,...,no]:
            		Number of layers of neural network is len(StructureOfNeuralNetwork)
            		n1: Number of neurons in input layer, whose value is equal to the number of variables in the equation
            		ni: The number of neurons in the ith hidden layer, whose value is selected according to the complexity and oscillation of the equation.
           		no: The number of neurons in the output layer must be 1.
    	:param ImplicitSchemeOfEquation: Implicit scheme of differential equation.
    	:param DirichletBCPoint = [x1,x2,x3,x4,x5]: There are at most five Dirichlet boundary conditions,that is, at most five order differential equations are supported.
    	:param numBatches: Number of training iterations.
    	:return y_output: The numerical solution predicted by Deep Learning is returned in the form of row vector.
 

3. Case: 
Case1.   First order differential equation
             real solution: u(x)= 5*x**3 + x**2 + 2*x + 1
             u'(x) = 15*x**2 + 2*x + 2;   u(-1) = -4;   x¡Ê[-1,1]
#Code of Case1
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

domain = [-1, 1]                                                                    	     	#Domain
realSolution = lambda x: 5*x**3 + x**2 + 2*x + 1                                    		#Real solution
n = 100                                                                             		#Divide the domain into n sample points
#If appear 'Fail rename; Input/output error', please delete the last saved model parameter file 'ckpt' and training again.
StructureOfNeuralNetwork = [1, 10, 1]                                               	        #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(u, x)[0] - 15*x**2 - 2*x - 2   	        #It must be the implicit scheme of the equation 
DirichletBCPoint = [-1]                                                             		Dirichlet boundary conditions
numBatches = 30000                                                                  		#Number of iterations
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation, DirichletBCPoint, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)


Case2.     Second order differential equation
	real solution: u(x)=x**5
	u''(x)=20*x**3; u(-1)=-1; u(1)=1; x¡Ê[-1,1]
#Code of Case2
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

domain = [-1, 1]                                                                              	 #Domain
realSolution = lambda x: x**5                                                                 	 #Real solution
n = 100                                                                                       	 #Divide the domain into n sample points
#If appear 'Fail rename; Input/output error', please delete the last saved model parameter file 'ckpt' and training again.
StructureOfNeuralNetwork = [1, 10, 5, 2, 1]                                                   	 #Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(tf.gradients(u, x)[0], x)[0] - 20*x**3    
DirichletBCPoint = [-1, 1]                                                                    	 #Number of iterations
numBatches = 30000
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation, DirichletBCPoint, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)


Case3.	Third order differential equation
	real solution: u(x)=x**7 + 2*x**5 + 3*x**3 + x**2
	u'''(x) = 210*x**4 + 120*x**2 + 18; u(-1)=-5; u(0)=0; u(1)=7; x¡Ê[-1,1]
#Code of Case3
import tensorflow as tf
import PDESolverByDeepLearning.hanzuliang as PDESolver

domain = [-1, 1]                                                                              	#Domain
realSolution = lambda x: x**7 + 2*x**5 + 3*x**3 + x**2                                        	#Real solution
n = 100                                                                                      	#Divide the domain into n sample points
#If appear 'Fail rename; Input/output error', please delete the last saved model parameter file 'ckpt' and training again.
StructureOfNeuralNetwork = [1, 50, 20, 5, 1]                                                  	#Neural network structure
ImplicitSchemeOfEquation = lambda x, u: tf.gradients(tf.gradients(tf.gradients(u, x)[0], x)[0], x)[0]  - 210*x**4 - 120*x**2 - 18
DirichletBCPoint = [-1, 0, 1]                                                                 	#Number of iterations
numBatches = 50000
y_output = PDESolver.PDESolver(domain, n, realSolution, StructureOfNeuralNetwork, ImplicitSchemeOfEquation, DirichletBCPoint, numBatches)
print('The discrete solution predicted by Deep Learning is:')
print(y_output)



from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))        
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros((1, num_classes))
      
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        X, cache2 = affine_forward(X, self.params['W2'], self.params['b2'])
        scores = X
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dx = softmax_loss(X, y)

        loss += 0.5 * self.reg * np.sum(np.sum(self.params['W1'] * self.params['W1']))
        loss += 0.5 * self.reg * np.sum(np.sum(self.params['W2'] * self.params['W2']))
        
        dx, dw, db = affine_backward(dx, cache2)
        grads['W2'] = dw + self.reg * self.params['W2']
        grads['b2'] = db
        dx, dw, db = affine_relu_backward(dx, cache1)
        grads['W1'] = dw + self.reg * self.params['W1']
        grads['b1'] = db
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float32 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        weight_bound = 1.7321*weight_scale
        if(len(hidden_dims) > 0):
            weights = [np.random.random(-weight_bound, weight_bound, (input_dim, hidden_dims[0])).astype(np.float32)]
            biases = [np.zeros(1, hidden_dims[0], dtype=np.float32)]
            
            scales = [np.ones(input_dim, hidden_dims[0], dtype=np.float32)]
            shifts = [np.zeros(1, hidden_dims[0], dtype=np.float32)]

            if(len(hidden_dims) > 1):
                for layer_index in range(1, self.num_layers-1):
                    weights.append(np.random.random(-weight_bound, weight_bound, (hidden_dims[layer_index-1], hidden_dims[layer_index])).astype(np.float32))
                    biases.append(np.zeros(1, hidden_dims[layer_index], dtype=np.float32))
                    
                    scales.append(np.ones(input_dim, hidden_dims[0], dtype=np.float32))
                    shifts.append(np.zeros(1, hidden_dims[0], dtype=np.float32))
                    
            weights.append(np.random.random(-weight_bound, weight_bound, (hidden_dims[-1], num_classes)).astype(np.float32))
            biases.append(np.zeros(1, num_classes, dtype=np.float32))
            
            # scales.append(np.ones(hidden_dims[-1], num_classes, dtype=np.float32))
            # shifts.append(np.zeros(1, num_classes, dtype=np.float32))
        
        else:
            weights = [np.random.random(-weight_bound, weight_bound, (input_dim, num_classes), dtype=np.float32)]
            biases = [np.zeros(1, num_classes, dtype=np.float32)]
            
            # scales = [np.ones(input_dim, num_classes, dtype=np.float32)]
            # shifts = [np.zeros(1, num_classes, dtype=np.float32)]
            
        self.params['weights'] = weights
        self.params['biases'] = biases
    
    def loss(self, x, y = None):
        cache = []
        
        for layer_index in range(self.num_layers):
            x, cache_temp = affine_forward(x=x, w=self.params['weight'][layer_index], b=self.params['biases'][layer_index])
            cache_temp.append(cache_temp)
            
        
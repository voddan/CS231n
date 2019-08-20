from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0]   # N
    loss = 0.0
    for i in range(num_train): #N
        scores = X[i].dot(W) # C
        correct_class_score = scores[y[i]]
        for j in range(num_classes): #C
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train): #N
        scores = X[i].dot(W) # C
        correct_class_score = scores[y[i]]
        for j in range(num_classes): #C
            # if j == y[i]:
            #     continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                if j != y[i]:
                    dW[:, j] += X[i]
                    dW[:, y[i]] -= X[i]

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    C = W.shape[1]  # C
    N = X.shape[0]  # N

    scores1 = X.dot(W)  # (N, C)

    correct_scores = scores1[range(N), y]
    assert correct_scores.shape == (N, )

    scores0 = scores1.T

    margin2 = scores0 - correct_scores
    margin1 = margin2 + 1
    margin0 = margin1.T

    marginClipped = np.clip(margin0, a_min=0, a_max=None)

    loss3 = np.sum(marginClipped)
    loss2 = loss3 - N
    loss1 = loss2 / N

    regw2 = W * W
    regw1 = np.sum(regw2)
    regw0 = reg * regw1

    loss0 = loss1 + regw0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #backprop loss0 = loss1 + regw0
    dloss1 = 1
    dregw0 = 1
    #backprop regw0 = reg * regw1
    dregw1 = reg * dregw0
    #backprop regw1 = np.sum(regw2)
    dregw2 = np.ones(regw2.shape) * dregw1
    #backprop regw2 = W * W
    dW_0 = 2 * W * dregw2


    #backprop loss1 = loss2 / N
    dloss2 = 1/N * dloss1
    #backprop loss2 = loss3 - N
    dloss3 = 1 * dloss2
    #backprop loss3 = np.sum(marginClipped)
    dmarginClipped = np.ones(marginClipped.shape) * dloss3
    #backprop marginClipped = np.clip(margin0, a_min=0, a_max=None)
    dmargin0 = dmarginClipped
    dmargin0[np.nonzero(margin0 <= 0)] = 0
    #backprop margin0 = margin1.T
    dmargin1 = dmargin0.T
    #backprop margin1 = margin2 + 1
    dmargin2 = dmargin1
    #backprop margin2 = scores0 - correct_scores
    dscores0 = 1 * dmargin2
    dcorrect_scores = -1 * dmargin2
    #backprop scores0 = scores1.T
    dscores1_1 = dscores0.T
    #backprop correct_scores = scores1[range(N), y]
    t = np.ones(dcorrect_scores.shape)
    t[y, range(N)] = 0
    dscores1_2 = dcorrect_scores
    dscores1_2[np.nonzero(t)] = 0

    dscores1 = dscores1_1 + dscores1_2.T
    #backprop scores1 = X.dot(W)
    dW_1 = X.T.dot(dscores1)

    dW = dW_0 + dW_1


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss0, dW

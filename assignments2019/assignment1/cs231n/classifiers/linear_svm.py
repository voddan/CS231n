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

    num_classes = W.shape[1]  # C
    num_train = X.shape[0]  # N
    loss = 0.0
    scores = X.dot(W)  # (N, C)

    correct_scores = scores[range(num_train), y]
    assert correct_scores.shape == (num_train, )

    margin = (scores.T - correct_scores + 1).T  # (C)

    marginClipped = np.clip(margin, a_min=0, a_max=None)
    loss += np.sum(marginClipped) - num_train

    loss /= num_train
    loss += reg * np.sum(W * W)

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
    num_dim = X.shape[1]

    scores = X.dot(W)  # (N, C)
    correct_scores = scores[range(num_train), y]
    margin = (scores.T - correct_scores + 1).T  # (N, C)
    assert margin.shape == (num_train, num_classes)

    # dScores = X.T.dot(margin > 0)
    # dScores = X  #(N, D)
    # dCorrect_scores = dScores[range(num_train), y]
    #
    # dMargin = (dScores.T - dCorrect_scores).T
    #
    # dMarginClipped = dMargin * (margin > 0) #todo
    # dW += np.sum(dMarginClipped)




    dScore = X.T.dot(margin > 0)
    dW += dScore

    # sum = X.T.dot(np.sum(margin > 0, axis=1).reshape(num_train, 1))

    for i in range(num_train): #N
        sum = X.T * np.sum(margin > 0, axis=1)[i]
        assert sum.shape == (num_dim, num_train), sum.shape
        dW[:, y[i]] -= sum[:, i]

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

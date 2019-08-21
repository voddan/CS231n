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

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    import autograd.numpy as np
    from autograd import grad, jacobian, elementwise_grad, holomorphic_grad

    C = W.shape[1]  # C
    N = X.shape[0]  # N

    def scores1F(W):
        scores1 = np.dot(X, W)  # (N, C)
        return scores1

    def correct_scoresF(W):
        correct_scores = scores1F(W)[range(N), y]
        return correct_scores

    def scoresF(W):
        scores0 = scores1F(W).T
        return scores0

    def marginF(W):
        margin2 = scoresF(W) - correct_scoresF(W)
        margin1 = margin2 + 1
        margin0 = margin1.T
        return margin0

    def marginClippedF(W):
        marginClipped = np.clip(marginF(W), a_min=0, a_max=None)
        return marginClipped

    def sumMarginClippedF(W):
        loss3 = np.sum(marginClippedF(W))
        return loss3

    def lossF(W):
        loss3 = sumMarginClippedF(W)
        loss2 = loss3 / N
        loss1 = loss2 - 1
        return loss1

    def regwF(W):
        regw2 = W * W
        regw1 = np.sum(regw2)
        regw0 = reg * regw1
        return regw0

    def svm_loss(W):
        loss0 = lossF(W) + regwF(W)
        return loss0


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

    def dmarginF(W):
        return elementwise_grad(marginF)(W)

    def dmarginClippedF(W):
        # mc = marginClippedF(W)
        # d = dmarginF(W)
        # d[np.nonzero(mc <= 0)] = 0
        # return d
        return elementwise_grad(marginClippedF)(W)

    def dsumMarginClippedF(W):
        return dmarginClippedF(W)

    def dlossF(W):
        return 1/N * dsumMarginClippedF(W)

    def dregwF(W):
        return 2 * reg * W

    def dsvm_loss(W):
        return dlossF(W) + dregwF(W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return svm_loss(W), dsvm_loss(W)

# ADMM-for-Logistic-Regression
This is the Python package for ADMM for Logistic Regression.

## Description
ADMM (Alternating Direction Method of Multipliers) is a popular approach for convex optimization which could be useful for separable target function with regularization. Moreover, existed work have already shown that this method could solve logistic regression problem with various regularization terms. However, when sample size becomes explosive, it is extremely time-consuming even if one utilizes parallel computing techniques, since it requires one pass of all data samples at each iteration. Our goal here is to develop a computational method to reduce its time complexity for logistic regression with regularization. One specific goal is to combine existing algorithm packages such as CVXPy with state-of-the-art computational tool in PyTorch/Tensorflow

## Stage 1
At this stage, we focus on writing codes of consensus ADMM for logistic regression with regularization. We focus on the case where the regularization term is the L1 norm of parameter vector, and we applied parallel computing method for updating coefficients.

## Reference
Stephen Boyd, Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers,  http://stanford.edu/~boyd/papers/pdf/admm\_distr\_stats.pdf

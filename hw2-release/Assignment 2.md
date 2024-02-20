## Assignment 2

#### Problem 1

Laplace formula:

$L(\omega, \xi, \alpha, \beta) = \frac{1}{2} \|\omega\|^2 + C \sum_{i=1}^{n} \xi_i + \sum_{i=1}^{n} \alpha_i (1 - y_i \langle \omega, x_i \rangle - \xi_i) + \sum_{i=1}^{n} \beta_i (-\xi_i)$

Then we try to get dual problem:

$$
\min_{\omega, \xi}[\max_{\alpha>=0 ,\beta>=0} [\frac{1}{2} \|\omega\|^2 + C \sum_{i=1}^{n} \xi_i + \sum_{i=1}^{n} \alpha_i (1 - y_i \langle \omega, x_i \rangle - \xi_i) + \sum_{i=1}^{n} \beta_i (-\xi_i)]]
$$
equal to:
$$
\max_{\alpha>=0 ,\beta>=0}[\min_{\omega, \xi}[\frac{1}{2} \|\omega\|^2 + C \sum_{i=1}^{n} \xi_i + \sum_{i=1}^{n} \alpha_i (1 - y_i \langle \omega, x_i \rangle - \xi_i) + \sum_{i=1}^{n} \beta_i (-\xi_i)]]
$$
Fix $\alpha, \beta$
$$
\frac{\partial L}{\partial \omega} = 0 \Rightarrow \omega^* = \sum_{i=1}^{n} \alpha_i y_i x_i \quad  
$$

$$
\frac{\partial L}{\partial \xi_i} = 0 \Rightarrow C + \alpha_i (-1) + \beta_i (-1) = 0 \Rightarrow \alpha_i + \beta_i = C
$$

Substituting w into Lagrangian；
$$
\max_{\alpha \geq0, \beta \geq0, \alpha+\beta = C} \alpha^T 1-\frac{1}{2}\alpha^Ty^Tx^Txy\alpha
$$

#### Problem 2

To address each of these points, let's break them down one by one:

### 1. Prediction Function for Hard-Margin SVM

Given the optimal dual solution $(\hat{\alpha}_i)_{i=1}^N$ for a hard-margin SVM, the prediction function \( f(x) \) for a new input \( x \) can be written as follows, based on the dual formulation of SVM:

$ f(x) = \sum_{i=1}^N \hat{\alpha}_i y_i \langle x_i, x \rangle + b $

### 2. Prediction Function Using RBF Kernel

When applying the kernel trick with the RBF (Radial Basis Function) kernel, the prediction function $f_\sigma(x)$  changes to incorporate the kernel function \( $\kappa(x_i, x)$ \), which replaces the dot product \( $\langle x_i, x \rangle$ \). The RBF kernel is defined as:

 $\kappa(x_i, x) = \exp\left(-\frac{\|x_i - x\|^2}{2\sigma^2}\right)$ 

Thus, the prediction function with the RBF kernel becomes:

$f_\sigma(x) = \sum_{i=1}^N \hat{\alpha}_i y_i \kappa(x_i, x) + b$ 

### 3. Limit

To show that 

$$
\lim_{\sigma \rightarrow 0} \frac{f_\sigma(x)}{\exp\left(-\frac{\rho^2}{2\sigma^2}\right)} = \sum_{i \in T} \hat{\alpha}_i y_i
$$
we consider the contribution of the support vectors in \( T \) (the set of closest support vectors to \( x \)) and those in  $S \setminus T$  separately.

For $i \in T $, as $\sigma \rightarrow 0 $,  $\kappa(x_i, x)$  approaches 1 for the closest support vectors because $ \|x - x_i\|^2 = \rho^2 $, which is the smallest distance.

For  $i \in S \setminus T$ , the kernel values  $\kappa(x_i, x)$  approach 0 faster than $\exp\left(-\frac{\rho^2}{2\sigma^2}\right) $ because \($\|x - x_i\|^2 > \rho^2$.

Therefore, in the limit as  $\sigma \rightarrow 0$ , the sum over  $i \in S \setminus T$  becomes negligible compared to the sum over $i \in T$ , leading to the expression:

$ \lim_{\sigma \rightarrow 0} \frac{f_\sigma(x)}{\exp\left(-\frac{\rho^2}{2\sigma^2}\right)} = \sum_{i \in T} \hat{\alpha}_i y_i $

This demonstrates that as \( \sigma \) becomes very small, the RBF kernel SVM prediction at \( x \) is dominated by the contribution of the nearest support vectors, making it akin to a 1-nearest neighbor predictor with support vectors as the training set.



#### Problem 3

With the data points and their labels provided, we can now proceed with the calculations:

### 1. Sample Entropy of \( D \)

Given that we have 3 instances of the label -1 and 3 instances of the label 1, the dataset is balanced. Thus, the sample entropy of \( D \) is:

$H(D) = -\left(\frac{3}{6} \log_2\left(\frac{3}{6}\right) + \frac{3}{6} \log_2\left(\frac{3}{6}\right)\right) $
$H(D) = -2 \times \left(\frac{1}{2} \log_2\left(\frac{1}{2}\right)\right) $
$ H(D) = -2 \times \left(\frac{1}{2} \times -1\right) $
$ H(D) = 1 $

### 2. Maximum Information Gain for the First Split

x1 >=5

$I = H(D) - \sum\frac{S_i}{S}H(S_i)$

$H(D) = 1$

$S_1 = -\frac{3}{4}log(\frac{3}{4})-\frac{1}{4}log(\frac{1}{4})$

$S_2= 0$

$I=1-\frac{4}{6}S_1=0.625$

### 3. Further Splits Based on Maximum Information Gain

##### first split: 

​	$I=0$

##### second split:

 	$x2<=2$

​	$I = 0.625-0 = 0.625$

### AdaBoost with Decision Stumps

For the AdaBoost part, since the dataset is small, let's perform the calculations manually for each iteration \( t = 1, 2 \).

#### Initial Weights

For \( t=1 \), all weights are initialized to \( 1/6 \) because there are 6 data points.

#### Iteration \( t = 1 \)

1. **Choose the decision stump** that minimizes the weighted error rate \( \epsilon_t \). We would check each feature for the best split.
2. **Calculate the weighted error rate** for this stump.
3. **Calculate the weight of the decision stump** using \( \alpha_t = \frac{1}{2}\log\left(\frac{1-\epsilon_t}{\epsilon_t}\right) \).
4. **Update the weights** for each sample based on the misclassification.

#### Iteration \( t = 2 \)

1. **Repeat the process** with the updated weights from iteration 1.


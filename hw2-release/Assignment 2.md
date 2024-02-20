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





#### Problem 4

##### 1. Generalization Bound

Given:
- True error rate \( $R(h) = p$ \)
- Empirical error rate \( $\hat{R}_S(h) = \hat{p}$ \)
- Confidence level \( $\delta = 0.05$ \) 
- Accuracy requirement \( $\epsilon = 0.05$ \)

The Hoeffding Inequality states: $P(|\hat{p} - p| > \epsilon) \leq 2e^{-2n\epsilon^2}$ 

to ensure \( $P(|\hat{p} - p| > \epsilon) \leq \delta$ \), so we set: $2e^{-2n\epsilon^2} \leq \delta $

Solving for \( n \), the number of samples, gives us: $n \geq \frac{\log(2/\delta)}{2\epsilon^2} $

n >= 738

##### 2. VC Dimensions

(a) 

For the 1D affine classifier,  $F_{\text{affine}} = \{1\{wx + w_0 \geq 0\} : X \rightarrow R | w, w_0 \in R\} $, _

we need to find the VC dimension $V_C(F{\text{affine}}) $

**Proof Strategy:**

- **Shattering 1 point:** It's straightforward to see that any single point in \( $R$ \) can be shattered by choosing appropriate values of \( $w$ \) and \( $w_0$ \). For example, for a point \( $x_1$ \), we can choose \( w \) and \( w_0 \) such that \( $wx_1 + w_0 \geq 0$ \) for the label 1, and \( $wx_1 + w_0 < 0$ \) for the label 0.
- **Shattering 2 points:** For two distinct points \( $x_1$ \) and \( $x_2$ \), we can find \( $w$ \) and \( $w_0$ \) that correctly classify any labeling of these points. This can be done by positioning the decision boundary (the point where \( $wx + w_0 = 0$ \)) between \( $x_1$ \) and \( $x_2$ \) for different labelings.
- **Failure at 3 points:** To show that 3 points cannot be shattered, consider three points  $x_1 < x_2 < x_3 $ and a labeling that requires \( x_1 \) and \( x_3 \) to be classified as 1 while \( x_2 \) as 0. No single line (affine function) can separate \( x_2 \) from \( x_1 \) and \( x_3 \) in this way.

Hence, \( $V_C(F_{\text{affine}}) = 2$ \).

 (b) 

For a general affine classifier in \( k \) dimensions, $ F^k_{\text{affine}} = \{1\{w^Tx + w_0 \geq 0\} : X \rightarrow R^k | w \in R^k, w_0 \in R\} $, 

the VC dimension is related to the number of parameters defining the decision boundary, which is \( k + 1 \) (including  $w_0$ ).

**Proof Strategy:**

##### Shattering \( k \) Points

Let's consider \( k \) points in \( $\mathbb{R}^k$ \). We want to show that for any labeling of these points, there exists a hyperplane defined by \( w \) and \( w_0 \) that can separate the points according to their labels.

1. **Choice of Points**: Choose k  points in  $\mathbb{R}^k$  such that they are linearly independent. A simple way to ensure linear independence is to choose points that lie along the axes of  $\mathbb{R}^k$ ,

2. **Labeling**: Consider an arbitrary labeling of these points. Each point  $x_i$  is assigned a label  $y_i$  where $y_i \in \{+1, -1\}$ .

3. **Constructing the Hyperplane**: We need to find $w$ and  $w_0$ such that for each \( i \), $sign(w^Tx_i + w_0) = y_i$ . This can be written as a system of linear inequalities:
   $w^Tx_i + w_0 > 0, \text{ if } y_i = +1 $
   $w^Tx_i + w_0 < 0, \text{ if } y_i = -1$

4. **Solving for \( w \) and \( w_0 \)**: Since the points $x_i$ are linearly independent, the system of equations (or inequalities) has a solution. This is a consequence of the properties of linearly independent vectors and the capability of a hyperplane to separate them in $\mathbb{R}^k$.

##### Failure at \( k + 1 \) Points

Now, let's prove that \( k + 1 \) points in \( $\mathbb{R}^k$ \) cannot be shattered by such classifiers.

1. **Choosing \( k + 1 \) Points**: In \( $\mathbb{R}^k$ \), choose \( k + 1 \) points. Unlike the case with \( k \) points, these \( k + 1 \) points must be linearly dependent since there are more points than dimensions.

2. **Pigeonhole Principle**: The pigeonhole principle suggests that with \( k + 1 \) points and \( k \) dimensions, there must be some redundancy or dependency among the points, implying that they cannot all be separated linearly by a hyperplane.

3. **Configuration of Labels**: Consider a configuration where the points are labeled in such a way that the point which is a linear combination of the others has a different label from the rest. Due to the linear dependency, there's no hyperplane that can separate this point from the others while correctly classifying all points.

No hyperplane can correctly classify \( x_{k+1} \) without misclassifying at least one of the other points, due to the linear dependence.

This demonstrates that for any set of \( k + 1 \) points in  $\mathbb{R}^k$ , there exists a labeling configuration that cannot be separated by a hyperplane, proving that k + 1 points cannot be shattered and thereby establishing the VC dimension of general affine classifiers in $\mathbb{R}^k$  as  k .

Therefore, \( $V_C(F^k_{\text{affine}}) = k + 1$ \).

(c)

Only 1

##### Shattering 1 points:

You can always find c to shatter

##### Shattering 2 points:

let $x_1 = -x_2$

Therefore,

$\cos(cx_1) = \cos(cx_2)$

and $x_1$, $x_2$ cannot be seperated.

$V_C(F_{\text{cos}}) =  1$ 

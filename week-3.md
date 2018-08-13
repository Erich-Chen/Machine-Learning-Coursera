# Week-3

## 6a. Classification and Representation

### 6.1 Classification

To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification problem** in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y∈{0,1}. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the label for the training example.

### 6.2 Hypothesis Representation

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for $h_\theta (x)$ to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the form for our hypotheses $h_\theta (x)$ to satisfy $0 \leq h_\theta (x) \leq 1$. This is accomplished by plugging $\theta^Tx$ into the Logistic Function.

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

$\quad \displaystyle h_\theta(x) = g(\theta^Tx)$

$\quad \displaystyle z = \theta^Tx$

$\quad \displaystyle g(z) = \frac{1}{1 + e^{-z}}$

The following image shows us what the sigmoid function looks like:

![](pictures/Logistic_function.png)

The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

$h_\theta(x)$ will give us the **probability** that our output is 1. For example, $h_\theta(x)=0.7$ gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

$h_\theta(x) = P(y=1|x;\theta) = 1 - P(y=0|x;\theta)$

$P(y=0|x;θ)+P(y=1|x;θ)=1$

### 6.3 Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

$h_\theta(x) \geq 0.5 \rightarrow y=1$

$h_\theta(x) \lt  0.5 \rightarrow y=0$

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

$g(z) \geq 0.5$ when $z \geq 0$

Remember.

$z = 0,e^0=1 \Rightarrow g(z)= \frac{1}{2}$

$z \rightarrow \infty, e^{-\infty} \rightarrow 0 \Rightarrow g(z)=1$

$z \rightarrow -\infty, e^\infty \rightarrow 0 \Rightarrow g(z)=0$

So if our input to g is $\theta^T X$, then that means:

$h_\theta(x)=g(\theta^Tx)≥0.5$ when $\theta^Tx≥0$

From these statements we can now say:

$\theta^Tx \geq 0 \Rightarrow y=1$

$\theta^Tx \lt 0 \Rightarrow y=0$

The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

**Example**:

$\theta = \left[\begin{matrix} 1 \\  2 \\ 3 \end{matrix}\right]$

$y = 1 \space\space\space\space if \space 5 + (-1)x_1 + 0x_2 \geq 0$

$5 - x_1 \geq 0$

$- x_1 \geq -5$

$x_1 \leq 5$

In this case, our decision boundary is a straight vertical line placed on the graph where $x_1 = 5$, and everything to the left of that denotes $y = 1$, while everything to the right denotes $y = 0$.

Again, the input to the sigmoid function $g(z)$ (e.g. $\theta^T X$) doesn't need to be linear, and could be a function that describes a circle (e.g. $z = \theta_0 + \theta_1 x_1^2 +\theta_2 x_2^2$​) or any shape to fit our data.

## 6b. Logistic Regression Model

### 6.4 Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:

$\quad \displaystyle J(\theta) = \frac{1}{m} \sum_{i=1}^m Cost(h_\theta(x^{(i)}, y^{(i)})$

$\quad \displaystyle Cost(h_\theta(x^{(i)}, y^{(i)}) = -log(h_\theta(x)) \quad \quad \quad if \ y=1$

$\quad \displaystyle Cost(h_\theta(x^{(i)}, y^{(i)}) = -log(1 - h_\theta(x)) \quad \quad \quad if \ y=0$

When $y = 1$, we get the following plot for $J(\theta)$ vs $h_\theta (x)$:

![](pictures/Logistic_regression_cost_function_positive_class.png)

Similarly, when $y = 0$, we get the following plot for $J(\theta)$ vs $h_\theta (x)$:

![](pictures/Logistic_regression_cost_function_negative_class.png)

$\quad Cost(h_\theta(x),y) = 0 \space if \space h_\theta(x)=y$

$\quad Cost(h_\theta(x),y) → \infty \space if \space y=0 \space and \space h_\theta(x)→1$

$\quad Cost(h_\theta(x),y) → \infty \space if \space y=1 \space and \space h_\theta(x)→0$

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that $J(\theta)$ is convex for logistic regression.

### 6.5 Simplified Cost Function and Gradient Descent

**Note:** [6:53 - the gradient descent equation should have a 1/m factor]

We can compress our cost function's two conditional cases into one case:

$\quad \mathrm{Cost}(h_\theta(x), y) = -y \; log(h_\theta(x)) - (1-y) \; log(1-h_\theta(x))$

Notice that when y is equal to 1, then the second term $(1-y)\log(1-h_\theta(x))$ will be zero and will not affect the result. If y is equal to 0, then the first term $-y \log(h_\theta(x))$ will be zero and will not affect the result.

We can fully write out our entire cost function as follows:

$\quad \displaystyle J(\theta) = - \frac{1}{m} \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$

A vectorized implementation is: 

$\quad h = g(Xθ)$

$\quad J(\theta) = \frac{1}{m} \cdot (-y^Tlog(h) - (1-y)^Tlog(1-h))$

**Gradient Descent**

Remember that the general form of gradient descent is: 

`Repeat{`

$\quad \displaystyle \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)$

`}`

We can work out the derivative part using calculus to get:

`Repeat{`

$\quad \displaystyle \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m(h_\theta(x^{(i)} - y^{(i)})x_j^{(i)}$

`}`

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

$\quad \theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$

### 6.6 Advanced Optimization

**Note:** [7:35 - '100' should be 100 instead. The value provided should be an integer and not a character string.]

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value

$\quad \theta$

$\quad J(\theta)$

$\quad \displaystyle \frac{\partial}{\partial \theta_j}$

We can write a single function that returns both of these:

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use octave's `fminunc()` optimization algorithm along with the `optimset()` function that creates an object containing the options we want to send to `fminunc()`. (Note: the value for MaxIter should be an integer, not a character string - errata in the video at 7:30)

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function `fminunc()` our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

### 6.7 Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

$\quad y \in \{0, 1 \dots n\}$

$\quad h_\theta^{(0)}(x) = P(y=0|x; \theta)$

$\quad h_\theta^{(1)}(x) = P(y=1|x; \theta)$

$\quad \cdots$

$\quad h_\theta^{(n)}(x) = P(y=n|x; \theta)$

$\quad \text{prediction} = \max\limits_i(h_\theta^{(i)}(x))$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

The following image shows how one could classify 3 classes:

![](pictures/Screenshot-2016-11-13-10.52.29.png)

**To summarize:**

Train a logistic regression classifier $h_\theta(x)$ for each class￼ to predict the probability that ￼ ￼$y = i￼$￼.

To make a prediction on a new $x$, pick the class ￼that maximizes $h_\theta (x)$.

## 7. Solving the Problem of Overfitting
### 7.1 The Problem of Overfitting

Consider the problem of predicting $y$ from $x \in \mathbb{R}$. The leftmost figure below shows the result of fitting a $y = \theta_0 + \theta_1x$ to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.

![](pictures/Screenshot-2016-11-15-00.23.30.png)

Instead, if we had added an extra feature $x^2$ , and fit $y = \theta_0 + \theta_1x + \theta_2x^2$ , then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a $5^{th}$ order polynomial $y = \sum_{j=0} ^5 \theta_j x^j$. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of **underfitting** — in which the data clearly shows structure not captured by the model—and the figure on the right is an example of **overfitting**.

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1. Reduce the number of features:
-  Manually select which features to keep.
-  Use a model selection algorithm (studied later in the course).

2. Regularization
-  Keep all the features, but reduce the magnitude of parameters $\theta_j$​.
-  Regularization works well when we have a lot of slightly useful features.

### 7.2 Cost Function

**Note:** [5:18 - There is a typo. It should be $\sum_{j=1}^{n} \theta _j ^2$  instead of $\sum_{i=1}^{n} \theta _j ^2$]

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

$\quad \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:

$\quad \displaystyle \min_\theta\ \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\cdot\theta_3^2 + 1000\cdot\theta_4^2$

We've added two extra terms at the end to inflate the cost of $\theta_3$ and $\theta_4$​. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near zero. This will in turn greatly reduce the values of $\theta_3x^3$ and $\theta_4x^4$ in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms $\theta_3x^3$ and $\theta_4x^4$.

![](pictures/Screenshot-2016-11-15-08.53.32.png)

We could also regularize all of our theta parameters in a single summation as:

$\quad \displaystyle \min_\theta\ \frac{1}{2m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2$

The λ, or lambda, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if $\lambda = 0$ or is too small ?


### 7.3 Regularized Linear Regression

**Note:** [8:43 - It is said that X is non-invertible if $m \leq n$. The correct statement should be that X is non-invertible if $m < n$, and may be non-invertible if $m = n$.]

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

 **Gradient Descent**

We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

`Repeat {`

$\quad \displaystyle \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)}) x_0^{(i)}$

$\quad \displaystyle \theta_j := \theta_j - \alpha \left[ \left(\frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] \quad j \in \{1,2\dots n\}$

`}`

The term $\frac{\lambda}{m}\theta_j$ performs our regularization. With some manipulation our update rule can also be represented as:

$\quad \displaystyle \theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$

The first term in the above equation, $1 - \alpha\frac{\lambda}{m}$ will always be less than 1. Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update. Notice that the second term is now exactly the same as it was before.

**Normal Equation**

Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

$\quad \theta = (X^TX + \lambda \cdot L)^{-1}X^Ty$

$\quad \quad \quad \quad where \ L =
  \begin{bmatrix}
    0   &   &   &   & \\
    &   1   &   &   & \\
    &   &   1   &   & \\
    &   &   & \ddots & \\
    &   &   &   &   1
  \end{bmatrix}$

$L$ is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including $x_0$​), multiplied with a single real number λ.

Recall that if $m < n$, then $X^TX$ is non-invertible. However, when we add the term λ⋅L, then $X^TX + λ \cdot L$ becomes invertible.

### 7.4 Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:

![](pictures/Screenshot-2016-11-22-09.31.21.png)

**Cost Function**

Recall that our cost function for logistic regression was:

$\quad \displaystyle J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) ]$

We can regularize this equation by adding a term to the end:

$\quad \displaystyle \begin{aligned}
 J(\theta) = & - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] \\
             & + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
 \end{aligned}$

The second sum, $\sum_{j=1}^n \theta_j^2$​ **means to explicitly exclude** the bias term, $\theta_0$. i.e. the θ vector is indexed from 0 to n (holding n+1 values, $\theta_0$ through $\theta_n$​), and this sum explicitly skips $\theta_0$​, by running from 1 to n, skipping 0. Thus, when computing the equation, we should continuously update the two following equations:

![](pictures/Screen-Shot-2016-12-07-at-10.49.02-PM.png)

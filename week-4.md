# week-4

## 8. Neural Networks: Representation

### 8.1 Motivations: Non-linear Hypotheses

### 8.2 Motivations: Neurons and the Brain

### 8.3 Neural Networks: Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features $x_1\cdots x_n$, and the output is the result of our hypothesis function. In this model our $x_0$ input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, $\displaystyle \frac{1}{1 + e^{-\theta^Tx}}$, yet we sometimes call it a sigmoid (logistic) activation function. In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like: 

$\quad \displaystyle \begin{bmatrix} x_0 \\ x_1 \\x_2 \end{bmatrix} \rightarrow [\quad] \rightarrow h_\theta(x)$

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes $a^2_0 \cdots a^2_n$ and call them "activation units."

$\quad a_i^{(j)}$    = "activation" of unit $i$ in layer $j$ 

$\quad \Theta^{(j)}$ = matrix of weights controlling function mapping from layer j to layer j+1

If we had one hidden layer, it would look like:

$\quad \displaystyle \begin{bmatrix} x_0 \\ x_1 \\x_2 \end{bmatrix} \rightarrow \begin{bmatrix} a_1^{(2)} \\ a_2^{(2)} \\a_3^{(2)} \end{bmatrix} \rightarrow h_\theta(x)$

The values for each of the "activation" nodes is obtained as follows:

$\displaystyle \quad \quad \quad \quad \quad 
a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)$

$\displaystyle \quad \quad \quad \quad \quad 
a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3)$

$\displaystyle \quad \quad \quad \quad \quad 
a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3)$

$\displaystyle \quad 
h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)})$

This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes. 

Each layer gets its own matrix of weights, $\Theta^{(j)}$. 

The dimensions of these matrices of weights is determined as follows:

> If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$. 

The +1 comes from the addition in $\Theta^{(j)}$ of the "bias nodes", $x_0$ and $\Theta_0^{(j)}$. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:

![](pictures/Screenshot-2016-11-22-10.08.51.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of $\Theta^{(1)}$ is going to be 4×3 where $s_j = 2$ and $s_{j+1}$ = 4$, so $s_{j+1} \times (s_j + 1) = 4 \times 3.

### 8.4 Neural Networks: Model Representation II

To re-iterate, the following is an example of a neural network:

$\displaystyle \quad \quad \quad \quad \quad 
a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)$

$\displaystyle \quad \quad \quad \quad \quad 
a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3)$

$\displaystyle \quad \quad \quad \quad \quad 
a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3)$

$\displaystyle \quad 
h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)})$

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $z_k^{(j)}$ that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get:

$\quad a_1^{(2)} = g(z_1^{(2)})$

$\quad a_2^{(2)} = g(z_2^{(2)})$

$\quad a_3^{(2)} = g(z_3^{(2)})$

In other words, for layer $j=2$ and node $k$, the variable $z$ will be:

$\quad z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n$

The vector representation of $x$ and $z^{j}$ is:

$\quad x = \begin{bmatrix} x_0 \\ x_1 \\ \cdots \\ x_n\end{bmatrix}, 
 z^{(j)} = \begin{bmatrix} z_1^{(j)} \\ z_2^{(j)} \\ \cdots \\ z_n^{(j)} \\ \end{bmatrix}$

Setting $x = a^{(1)}$, we can rewrite the equation as:

$\quad z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$ 
 
We are multiplying our matrix $\Theta^{(j-1)}$ with dimensions $s_j \times (n+1)$ (where $s_j$ is the number of our activation nodes) by our vector $a^{(j-1)}$ with height $(n+1)$. This gives us our vector $z^{(j)}$ with height $s_j$. Now we can get a vector of our activation nodes for layer $j$ as follows:

$\quad a^{(j)} = g(z^{(j)})$

Where our function g can be applied element-wise to our vector $z^{(j)}$.

We can then add a bias unit (equal to 1) to layer *j* after we have computed $a^{(j)}$. This will be element $a_0^{(j)}$ and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:

$\quad z^{(j+1)} = \Theta^{(j)}a^{(j)}$ 

We get this final $z$ vector by multiplying the next theta matrix after $\Theta^{(j-1)}$ with the values of all the activation nodes we just got. This last theta matrix $\Theta^{(j)}$ will have only **one row** which is multiplied by one column $a^{(j)}$ so that our result is a single number. We then get our final result with:

$\quad h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})$

Notice that in this **last step**, between layer j and layer j+1, we are doing **exactly the same thing** as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.


### 8.5 Applications: Examples and Intuitions I 

A simple example of applying neural networks is by predicting $x_1$ AND $x_2$, which is the logical 'and' operator and is only true if both $x_1$ and $x_2$ are 1.

The graph of our functions will look like:

$\displaystyle \quad
 \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} 
 \rightarrow \begin{bmatrix} g(z^{(2)}) \end{bmatrix} 
 \rightarrow h_\Theta(x)
$

Remember that $x_0$ is our bias variable and is always 1.

Let's set our first theta matrix as: 

$\quad \Theta^{(1)} = \begin{bmatrix} -30 & 20 & 20 \end{bmatrix}$

This will cause the output of our hypothesis to only be positive if both $x_1$ and $x_2$ are 1. In other words:

$\quad h_\Theta(x) = g(-30 + 20x_1 + 2-x_2)$

$\quad x_1=0 \ and \ x_2=0 \ then \ g(-30) \approx 0$

$\quad x_1=0 \ and \ x_2=1 \ then \ g(-10) \approx 0$

$\quad x_1=1 \ and \ x_2=0 \ then \ g(-10) \approx 0$

$\quad x_1=1 \ and \ x_2=1 \ then \ g( 10) \approx 1$

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates. The following is an example of the logical operator 'OR', meaning either $x_1$ is true or $x_2$ is true, or both:

![](pictures/Screenshot-2016-11-23-10.03.48.png)

Where g(z) is the following:

![](pictures/Screenshot-2016-11-23-10.07.24.png)


### 8.6 Applications: Examples and Intuitions II 

The $\Theta^{(1)}$matrices for AND, NOR, and OR are:

$\quad AND:Θ(1) = [−30 \quad 20 \quad 20]\\
 \quad NOR:Θ(1) = [10 \quad −20 \quad −20]\\
 \quad OR: Θ(1) = [−10 \quad 20 \quad 20]$


We can combine these to get the XNOR logical operator (which gives 1 if $x_1$ and $x_2$ are both 0 or both 1).

$\quad \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix}
 \rightarrow \begin{bmatrix} a_1^{(2)} \\ a_2^{(2)} \end{bmatrix}
 \rightarrow \begin{bmatrix} a^{(3)} \end{bmatrix}
 \rightarrow h_\Theta(x)
$

For the transition between the first and second layer, we'll use a $Θ^{(1)}$ matrix that combines the values for AND and NOR:

$\quad \Theta^{(1)} =
          \begin{bmatrix} -30 & 20 & 20 \\ 10 & -20 & -20\end{bmatrix}$

For the transition between the second and third layer, we'll use a $\Theta^{(2)}$ matrix that uses the value for OR:

$\quad \Theta^{(2)} =
          \begin{bmatrix} -10 & 20 & 20 \end{bmatrix}$

Let's write out the values for all our nodes:

$\quad a^{(2)} = g(\Theta^{(1)} \cdot x)\\
 \quad a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)})\\
 \quad h_\Theta(x) = a^{(3)}$

And there we have the XNOR operator using a hidden layer with two nodes! The following summarizes the above algorithm:

![](pictures/Screenshot-2016-11-23-10.28.41.png)


### 8.7 Applications: Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly:

![](pictures/Screenshot-2016-11-23-10.49.05.png)

We can define our set of resulting classes as y:

$\quad y^{(i)} =
   \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, 
   \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}, 
   \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, 
   \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}
$

Each $y^{(i)}$ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like:

$\quad \displaystyle \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ \cdots \\ x_n \end{bmatrix}
   \rightarrow \begin{bmatrix} a_0^{(2)} \\ a_1^{(2)} \\ a_2^{(2)} \\ \cdots \end{bmatrix}
   \rightarrow \begin{bmatrix} a_0^{(3)} \\ a_1^{(3)} \\ a_2^{(3)} \\ \cdots \end{bmatrix}
   \rightarrow \cdots
   \rightarrow \begin{bmatrix} h_\Theta(x)_1 \\ h_\Theta(x)_2 \\ h_\Theta(x)_3 \\ h_\Theta(x)_4 \end{bmatrix}
$

Our resulting hypothesis for one set of inputs may look like:

$h_\Theta(x) = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}$

In which case our resulting class is the third one down, or $h_\Theta(x)_3$, which represents the motorcycle.
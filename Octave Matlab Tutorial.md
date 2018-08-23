# Octave / Matlab Tutorial

## Submit Assignment
1. Download programming assignment zip file (machine-learning-exN.zip)
2. Access https://matlab.mathworks.com/
3. Drag zip file to webpage to upload it. 
4. Run `unzip machine-learning-exN.zip`. Then delete zip file on web side. 
4. Right-click the ‘machine-learning-exn’ folder, and select 'Remove from Path -> Selected Folder and Subfolders'.
5. right-click the folder 'exN' and select 'Open'. Start coding. 
6. Run `submit` with account (email) and assignment password (shown on assignment page). 
7. Pass the assignment. 

## Tutorial

### Vectorization Example

**1. for hypothesis representation:**

$\displaystyle h_\theta(x) = \sum_{j=0}^m \theta_j x_j = \theta^T X$

$where \quad 
    \theta =    \begin{bmatrix} 
                    \theta_0 \\ \theta_1 \\ \cdots \\ \theta_n 
                \end{bmatrix}$

$\quad \quad \quad \quad 
    x =         \begin{bmatrix} 
                    x_0 \\ x_1 \\ \cdots \\ x_n
                \end{bmatrix}$

```matlab
% remember the word 'hypothesis' == 'prediction'
% I would prefer 'h' or 'hypothesis' instead of 'prediction' as variable 

% unvectorized implementation
prediction = 0.0;
for j = 1:n+1,
  prediction = prediction + theta(j) * x(j)
end;

% Better and Faster: Vectorized implementation
prediction = theta' * x;
```


**2. for Gradient Descent:**

$\quad \displaystyle \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m \left(h(\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$

expand it into: 

$\quad \displaystyle \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m \left(h(\theta(x^{(i)}) - y^{(i)} \right) x_0^{(i)}$

$\quad \displaystyle \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m \left(h(\theta(x^{(i)}) - y^{(i)} \right) x_1^{(i)}$

$\quad \cdots \quad \cdots$

$\quad \displaystyle \theta_n := \theta_n - \alpha \frac{1}{m} \sum_{i=1}^m \left(h(\theta(x^{(i)}) - y^{(i)} \right) x_n^{(i)}$

thus, vectorized implementation is: 

$\quad \displaystyle \theta :=\theta - \alpha * \delta$

$\quad \quad \displaystyle where:$

$\quad \quad \quad \quad \displaystyle 
    \theta = \begin{bmatrix} 
            \theta_0 \\ \theta_1 \\ \cdots \\ \theta_n
        \end{bmatrix}$

$\quad \quad \quad \quad \displaystyle 
    \alpha \in \mathbb{R}$

$\quad \quad \quad \quad \displaystyle 
    \delta = \frac{1}{m} \sum_{i=1}^m \left(h(\theta(x^{(i)}) - y^{(i)} \right) X^{(i)}$

Noticed the facts that: 
* $\theta$ is a vector $\mathbb{R}^{n+1}$: (n+1) rows * 1 column;
  
* $\displaystyle \frac{1}{m}$ is a real number ($\mathbb{R}$);
  
* $\displaystyle \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)} \right)$ is also a real number ($\mathbb{R}$), 

  but both $h = h_\theta(x)$ and $y$ (NOT $h_\theta(x^{(i)})$ or $y^{(i)})$ are vectors $\mathbb{R}^m$ hence $h-y$ is vector $\mathbb{R}^m$
  
  then the interesting part is: $\sum ...$ can be caculated by matrix multiplication $\begin{bmatrix} 1 & 1 & \cdots & 1 \end{bmatrix} _{1 \times m} * (h-y)$
  
* $X$ is a matrix $\mathbb{R}^{n+1}$;

???
```matlab
% vectorized implementation
m = length(y)
h = theta' * X;
delta = (1/m) * (ones(1,m) * (h-y)) * x;
theta = theta - alpha .* delta;
```


**3. Cost Function J:** 

$\quad \displaystyle J(\theta) = \frac{1}{2m} \sum _{i=1}^m  (h_\theta (x_{i}) - y_{i} )^2$

the interesting part is: $\sum(h-y)^2$ can be caculated by matrix multiplication: 
$(h-y)^T * (h-y)$

```matlab
h = X * theta;
delta = h - y;

% delta' * delta = sum(delta .^2)
J = 1/(2*m) * (delta' * delta);
```

### Basic
```matlab
5+6
3-2
5*8
1/2
2^6

1 == 2
1 ~= 2   % NOT EQUAL

1 && 0    % AND
1 || 0    % OR
xor(1,0)  % XOR

PS1('>> ')

a = 3
a = 3; b = 'hi';   % semicolon supressing output
a = pi;
disp(a);
disp(sprintf('2 decimals: %0.2f', a));   % print a in 2 decimals
format long;    % set long number digits, default is short

A = [1 2; 3 4; 5 6]   % generate matrix 3 * 2
V = [1 2 3]
V = [1; 2; 3]
V = 1:0.1:1.5    % [1.0 1.1 1.2 1.3 1.4 1.5]
ones(2,3)        % generate all-1 matrix
zeros(1, 3)
C = 2*ones(2,3)

W = rand(3, 3)   % random numbers between 0 and 1
W = randn(1, 3)  % random numbers between -1 and 1

hist(W)          % draw history W's

eye(4)           $ identity matrix (单位矩阵，对角线为1其它全0)

help eye
help help
```

### Moving Data Around
```matlab
A = [1 2; 3 4; 5 6]
sz = size(A)        % return a matrix contains #row and #column of A - [3 2]
V = [1; 2; 3; 4; 5]
length(V)

pwd
cd 'C:\users\erich\Desktop'
ls

load featuresX.dat   % featuresX.dat is a file name
load('featuresX.dat')

who                 % show variables (just names) in current work space
whos                % more details than who
size(featuresX)
clear featuresX

V = priceY(1:10)    % take first 10 lines
save hello.mat V;   % save V to hello.mat
save hello.txt V -ascii   % save V as text (ASCII) for human reading

clear               % clear work space
load hello.mat      % load

A(3,2)              % create matrix A with 3 rows and 2 columns
A(1,:)              % get the 1st row
A(:,2)              % get 2nd column
A([1 3], :)         % get 1st and 3rd row with all columns
A(:,2) = [11; 12; 13]   % replace
A = [A, [101; 102; 103]]  % append another cloumn
A(:)                % all items into a vector
C = [A B]           % put B on left of A; same as [A, B]
C = [A; B]          % put B below A
```

### Computing on data
```matlab
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [1 1; 2 2]
V = [1; 2; 3]

A * C               % matrix multiplication
A .*C               % element-wise multiplication

A .^2
1 ./ A
log(V)
exp(V)
abs(V)
-V
V + ones(length(V), 1)
V + 1

A'                  % Transpose of A

a = [1 2 5 3 8 -1]
[val, ind] = max(a)   % get max value of A and its index

a < 3               % return values
find(a < 3)         % return indexes

A = magic(9)        % help magic for more

sum(a)
prod(a)             % product
ceil(a)

max(A, [], 1)       % max per column
max(A, [], 2)       % max per row

sum(A, 1)
sum(A, 2)

sum(sum(A .* eye(9)))
flipud(eye(9))      % 翻转对角矩阵，形成了反对角矩阵

pinv(A)             % pseudeo-inverse 求逆（伪）
```

### Plotting Data
```matlab
t = [0:0.01:0.98]
y1 = sin(2*pi*4*t)
y2 = cos(2*pi*4*t)

hold on;            % plot new figure on top of old ones
plot(t, y1)
plot(t, y2)
xlabel('time')
ylabel('value)
lagend('sin', 'cos')
title('my plot')
print -dpng 'myplot.png'
help plot
close

figure(1); plot(t, y1)
figure(2); plot(t, y2)

% draw two figures side-by-side 
subplot(1, 2, 1)
plot(t, y1)
subplot(1, 2, 2)
plot(t, y2)

axis([0.5 1 -1 1])
clf             % clear the figure

A = magic(5)
imagesc(A)      % different colors
imagesc(A), colorbar, colormap gray;
```

### Controls statements (for, while, if and functions)
```matlab
v = zeros(10, 1)
indicies = 1:10;
for i = indicies, 
    v(i) = 1^i;
end;

i = 1; 
while i >=5, 
    v(i) = 100;
    i = i+1;
end;

i = 1;
while true,
    v(i) = 999;
    i = i+1;
    if i == 6, 
        break;
    end;
end;

if v(1) == 1, 
    disp('the value is one');
elseif v(1) == 2,
    disp('the value is two');
else
    disp('the value is not one or two');
end;
```

```matlab
% this file (main.m) call function'
addpath('C:\path\to\file_func_m\')      % if not in same folder
squareThisNumber(5)
squareAndCubeThisNumber(5)
```

```matlab
% the file (fuc.m) defines function 
fucntion y = squareThisNumber(x)
y = x^2;
```

```matlab
function y = squareAndCubeThisNumber(x)
y = x^2^3
```

```matlab
function J = costFunctionJ(X, y, theta)
% X is the "design matrix" containing our training examples. 
% y is the class labels

m = size(X, 1);     % number of training examples
predictions = X ** theta;   % predictions of hypothesis on all m examples
sqrErrors = (predictions -y) .^2;   % squuared errors

J = 1/(2*m) * sum(sqrErrors);
```

## Referrence
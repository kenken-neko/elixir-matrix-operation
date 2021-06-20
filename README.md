# MatrixOperation
*MatrixOperation* is a linear algebra library in Elixir language. For example, this library can be used to solve eigenvalue equations and singular value decompositions. There are several other functions that can be used to solve some of these problems. You can refer to the online documentation at https://hexdocs.pm/matrix_operation/MatrixOperation.html#content and mathematical description at 'docs/latex_out/numerical_formula.pdf' in this package.    
Moreover, several patterns of functions are implemented as algorithms for solving each problem. The functions are methods that QR decomposition techniques to solve eigenvalue equations of arbitrary dimensions, or algebraic techniques that are limited in the number of dimensions but provide exact solutions. There is also function of the Jacobi method, which is a method for solving eigenvalue equations of real symmetric matrices.  

## Notice
A matrix of any dimension can be created.
```
iex> MatrixOperation.even_matrix(3, 2, 1)
[[1, 1], [1, 1], [1, 1]]
```
The first argument is the number of row number, the second argument is the number of column number and the third argument is value of the elements.  
Matrix indices of a row and column is an integer starting from 1 (not from 0).
<img src="https://user-images.githubusercontent.com/42142120/82437767-ed1afd00-9ad2-11ea-8ff0-223eb8f0b1d9.jpg" width="500">  
For example,
```
iex> rand_matrix = MatrixOperation.random_matrix(2, 3, 0, 1, "real")
[[0.3823, 0.8708, 0.9712], [0.4284, 0.0067, 0.0604]]

iex> MatrixOperation.get_one_element(rand_matrix, [1, 2])
0.8708
```

## Installation
You can install this package by adding this code to dependencies in your mix.exs file:
```elixir
def deps do
  [
    {:matrix_operation, "~> 0.3.9"}
  ]
end
```

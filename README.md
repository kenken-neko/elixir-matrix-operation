# elixir-matrix-operation
Matrix operation library in Elixir.  
The brief explanation (explanation.pdf) for a mathematical description is described.

## Operations:
* Transpose
* Trace
* Determinant
* Cramer's rule (to extract a specific solution of linear_equations)
* Constant multiple
* Inverse matrix
* Product
* Addition
* Subtraction
* Hadamard product
* Hadamard division
* Hadamard power
* Tensor product
* Eigenvalue (Algebratic method for 2×2 or 3×3 matrix)
* Eigenvalue and eigenvector (Iteration method for n×n matrix)
* Variance covariance matrix
* LU decomposition
* Direct method (to solve linear_equations)

    

## Sub Operations：
* Numbers of row and column of a matrix are informed. 
* A n-th unit matrix is generated.
* A n×m even matrix is generated.
* An element of a matrix is got. 
* A row/column of a matrix is extracted. 
* A row/column of a matrix is deleted.
* A row/column of a matrix is exchanged.


## Installation
You can install this package by adding this code to dependencies in your mix.exs file:
```elixir
def deps do
  [
    {:matrix_operation, "~> 0.1.1"}
  ]
end
```

## Notice
The column and row numbers are specified by a integer from 1 (not from 0).
<img src="https://user-images.githubusercontent.com/42142120/82437767-ed1afd00-9ad2-11ea-8ff0-223eb8f0b1d9.jpg" width="600">  
For example,
```
MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], [1, 1])
1
```

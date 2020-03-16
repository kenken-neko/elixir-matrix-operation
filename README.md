# elixir-matrix-operation
Matrix operation library in Elixir.

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
* Eigenvalue (2×2 or 3×3 matrix by algebratic method)
* Eigenvalue and eigenvector (n×n matrix by iteration method)
* Variance covariance matrix
* LU decomposition
* Direct method (to solve linear_equations)

    

## Sub Operations：
* Numbers of row and column of a matrix are informed. 
* A n-th unit matrix is generated.
* A n×m even matrix is generated.
* A element of a matrix is got. 
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

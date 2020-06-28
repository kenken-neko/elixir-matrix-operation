# elixir-matrix-operation
Matrix operation library in Elixir.  
It is described the brief explanation (explanation.pdf) for a mathematical description.

## Notice
The column and row numbers are specified by a integer from 1 (not from 0).
<img src="https://user-images.githubusercontent.com/42142120/82437767-ed1afd00-9ad2-11ea-8ff0-223eb8f0b1d9.jpg" width="500">  
For example,
```
MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], [1, 1])
1
```

## Installation
You can install this package by adding this code to dependencies in your mix.exs file:
```elixir
def deps do
  [
    {:matrix_operation, "~> 0.2.0"}
  ]
end
```

## Operations:
* Transpose
```elixir
MatrixOperation.transpose([[1.0, 2.0], [3.0, 4.0]])
[[1.0, 3.0], [2.0, 4.0]]
```
* Trace
```elixir
MatrixOperation.trace([[1.0, 2.0], [3.0, 4.0]])
5.0
```
* Determinant
```elixir
MatrixOperation.determinant([[1, 2, 1], [2, 1, 0], [1, 1, 2]])
-5
```
* Cramer's rule (to extract a specific solution of linear_equations)
```elixir
MatrixOperation.cramer([[0, -2, 1], [-1, 1, -4], [3, 3, 1]], [[3], [-7], [4]], 1)
2.0

MatrixOperation.linear_equations_cramer([[0, -2, 1], [-1, 1, -4], [3, 3, 1]], [[3], [-7], [4]])
[2.0, -1.0, 1.0]
```
* LU decomposition
```elixir
MatrixOperation.lu_decomposition([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
[
    L: [[1, 0, 0, 0], [2.0, 1, 0, 0], [3.0, 4.0, 1, 0], [-1.0, -3.0, 0.0, 1]],
    U: [[1, 1, 0, 3], [0, -1.0, -1.0, -5.0], [0, 0, 3.0, 13.0], [0, 0, 0, -13.0]]
]
```
* Direct method (to solve linear_equations)
```elixir
MatrixOperation.linear_equations_direct([[0, -2, 1], [-1, 1, -4], [3, 3, 1]], [[3], [-7], [4]])
[2.0, -1.0, 1.0]
```
* Constant multiple
```elixir
MatrixOperation.const_multiple(2, [[1, 2, 3], [2, 2, 2], [3, 8, 9]])
[[2, 4, 6], [4, 4, 4], [6, 16, 18]]
```
* Constant addition
```elixir
MatrixOperation.const_addition(1, [1.0, 2.0, 3.0])
[2.0, 3.0, 4.0]
```
* Inverse matrix
```elixir
MatrixOperation.inverse_matrix([[1, 1, -1], [-2, -1, 1], [-1, -2, 1]])
[[-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [-3.0, -1.0, -1.0]]
```
* Product
```elixir
MatrixOperation.product([[3, 2, 3], [2, 1, 2]], [[2, 3], [2, 1], [3, 5]])
[[19, 26], [12, 17]]
```
* Addition
```elixir
MatrixOperation.add([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
[[5, 5, 4], [5, 3, 4]]
```
* Subtraction
```elixir
MatrixOperation.subtract([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
[[1, -1, 2], [-1, -1, 0]]
```
* Hadamard product
```elixir
MatrixOperation.hadamard_product([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
[[6, 6, 3], [6, 2, 4]]
```
* Hadamard division
```elixir
MatrixOperation.hadamard_division([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
[[1.5, 0.6666666666666666, 3.0], [0.6666666666666666, 0.5, 1.0]]
```
* Hadamard power
```elixir
MatrixOperation.hadamard_power([[3, 2, 3], [2, 1, 2]], 2)
[[9.0, 4.0, 9.0], [4.0, 1.0, 4.0]]
```
* Tensor product
```elixir
MatrixOperation.tensor_product([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [2, 1, 2], [3, 5, 3]])
[
    [
        [[6, 9, 3], [6, 3, 6], [9, 15, 9]],
        [[4, 6, 2], [4, 2, 4], [6, 10, 6]],
        [[6, 9, 3], [6, 3, 6], [9, 15, 9]]
    ],
    [
        [[4, 6, 2], [4, 2, 4], [6, 10, 6]],
        [[2, 3, 1], [2, 1, 2], [3, 5, 3]],
        [[4, 6, 2], [4, 2, 4], [6, 10, 6]]
    ]
]
```
* Eigenvalue (Algebraic method for 2×2 or 3×3 matrix)
```elixir
MatrixOperation.eigenvalue([[6, -3], [4, -1]])
[3.0, 2.0]
```
* diagonalization (2×2 or 3×3 matrix)
```elixir
MatrixOperation.diagonalization([[1, 3], [4, 2]])
[[5.0, 0], [0, -2.0]]
```
* Eigenvalue and eigenvector (Power iteration method for n×n matrix)
```elixir
MatrixOperation.power_iteration([[1, 1, 2], [0, 2, -1], [0, 0, 3]], 100)
[3.0, [1.0, -2.0, 2.0]]
```
* Variance covariance matrix
```elixir
MatrixOperation.variance_covariance_matrix([[40, 80], [80, 90], [90, 100]])
[
    [466.66666666666663, 166.66666666666666],
    [166.66666666666666, 66.66666666666666]
]
```

    

## Sub Operations：
* Numbers of row and column of a matrix are informed.
```elixir
MatrixOperation.row_column_matrix([[3, 2, 3], [2, 1, 2]])
[2, 3]
```
* A n-th unit matrix is generated.
```elixir
MatrixOperation.unit_matrix(3)
[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```
* A n×m same element matrix is generated.
```elixir
MatrixOperation.even_matrix(3, 2, 1)
[[1, 1], [1, 1], [1, 1]]
```
* An element of a matrix is got. 
```elixir
MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], [1, 1])
1
```
* A row/column of a matrix is extracted. 
```elixir
MatrixOperation.get_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
[1, 2, 3]

MatrixOperation.get_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
[1, 4, 7]
```
* A row/column of a matrix is deleted.
```elixir
MatrixOperation.delete_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
[[1, 2, 3], [4, 5, 6]]

MatrixOperation.delete_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
[[1, 3], [4, 6], [7, 9]]
```
* A row/column of a matrix is exchanged.
```elixir
MatrixOperation.exchange_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, [1, 1, 1])
[[1, 2, 3], [4, 5, 6], [1, 1, 1]]

MatrixOperation.exchange_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, [1, 1, 1])
[[1, 1, 3], [4, 1, 6], [7, 1, 9]]
```

# elixir-matrix-operation
Matrix operation library in Elixir.  
It is described the brief explanation (numerical_formula.pdf) for a mathematical description.

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
    {:matrix_operation, "~> 0.3.4"}
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
    [[1, 0, 0, 0], [2.0, 1, 0, 0], [3.0, 4.0, 1, 0], [-1.0, -3.0, 0.0, 1]],
    [[1, 1, 0, 3], [0, -1.0, -1.0, -5.0], [0, 0, 3.0, 13.0], [0, 0, 0, -13.0]]
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
* Singular value (2×n or n×2 or 3×n or n×3 matrix)
```elixir
MatrixOperation.singular_value([[0, 1], [1, 0], [1, 0]])
[1.4142135623730951, 1.0]
MatrixOperation.singular_value([[2, 2, 2, 2], [1, -1, 1, -1], [-1, 1, -1, 1]])
[4.0, 0.0, 2.8284271247461903]
```
* diagonalization (2×2 or 3×3 matrix)
```elixir
MatrixOperation.diagonalization([[1, 3], [4, 2]])
[[5.0, 0], [0, -2.0]]
```
* Jordan normal form (2×2 or 3×3 matrix)
```elixir
MatrixOperation.jordan_normal_form([[1, -1, 1], [0, 2, -2], [1, 1, 3]])
[[2.0, 1, 0], [0, 2.0, 1], [0, 0, 2.0]]
MatrixOperation.jordan_normal_form([[3, 0, 1], [-1, 2, -1], [-1, 0, 1]])
[[2.0, 1, 0], [0, 2.0, 0], [0, 0, 2.0]]
MatrixOperation.jordan_normal_form([[1, 0, -1], [0, 2, 0], [0, 1, 1]])
[[2.0, 0, 0], [0, 0.9999999999999999, 1], [0, 0, 0.9999999999999999]]
```
* Eigenvalue and eigenvector (Power iteration method to solve maximum eigenvalue and eigenvector of n-th eigen equation)
```elixir
MatrixOperation.power_iteration([[1, 1, 2], [0, 2, -1], [0, 0, 3]], 100)
[
  3.0,
  [1.0, -2.0, 2.0]
]
```
The second argument (ex. 100) is max iterate number.
* Eigenvalue and eigenvector (Jacobi method to solve n-th eigen equation)
```elixir
MatrixOperation.jacobi([[10, 3, 2], [3, 5, 1], [2, 1, 0]], 100)
[
  [11.827601656659317, 3.5956497715829547, -0.42325142824210527],
  [
    [0.8892872578006493, -0.42761854121982545, -0.16220529066103917],
    [0.4179466723082325, 0.9038581385545962, -0.09143874712126684],
    [0.1857114757355589, 0.013522151221627882, 0.982511271796136]
  ]
]
```
The second argument (ex. 100) is max iterate number.
* Singular Value Decomposition (by using Jacobi method)
```elixir
MatrixOperation.svd([[1, 0, 0], [0, 1, 1]], 100)
[
  [1.0, 1.4142135623730951],
  [[1.0, 0], [0, 1.0]],
  [
    [1.0, 0, 0],
    [0, 0.7071067458364744, -0.707106816536619],
    [0, 0.707106816536619, 0.7071067458364744]
  ]
]
```
The second argument (ex. 100) is max iterate number.
* Frobenius norm
```elixir
MatrixOperation.frobenius_norm([[2, 3], [1, 4], [2, 1]])
5.916079783099616
```
* L one norm (L_1 norm)
```elixir
MatrixOperation.one_norm([[2, 3], [1, 4], [2, 1]])
5
```
* L two norm (L_2 norm)
```elixir
MatrixOperation.two_norm([[2, 3], [1, 4], [2, 1]])
5.674983803488142
```
* Max norm
```elixir
MatrixOperation.max_norm([[2, 3], [1, 4], [2, 1]])
8
```
* Rank (by Jacobi method)
```elixir
MatrixOperation.max_norm([[2, 3], [1, 4], [2, 1]])
8
```
The second argument (ex. 100) is max iterate number for Jacobi method.
* Variance covariance matrix
```elixir
MatrixOperation.rank([[2, 3, 4, 2], [1, 4, 2, 3], [2, 1, 4, 4]], 100)
3
```

    

## Sub Operations：
* Numbers of row and column of a matrix are informed.
```elixir
MatrixOperation.row_column_matrix([[3, 2, 3], [2, 1, 2]])
[2, 3]
```
* n-th unit matrix is generated.
```elixir
MatrixOperation.unit_matrix(3)
[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```
* n×m same element matrix is generated.
```elixir
MatrixOperation.even_matrix(3, 2, 1)
[[1, 1], [1, 1], [1, 1]]
```
* Element of a matrix is got. 
```elixir
MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], [1, 1])
1
```
* Row/Column of a matrix is extracted. 
```elixir
MatrixOperation.get_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
[1, 2, 3]

MatrixOperation.get_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
[1, 4, 7]
```
* Row/Column of a matrix is deleted.
```elixir
MatrixOperation.delete_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
[[1, 2, 3], [4, 5, 6]]

MatrixOperation.delete_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
[[1, 3], [4, 6], [7, 9]]
```
* Row/Column of a matrix is exchanged.
```elixir
MatrixOperation.exchange_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, [1, 1, 1])
[[1, 2, 3], [4, 5, 6], [1, 1, 1]]

MatrixOperation.exchange_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, [1, 1, 1])
[[1, 1, 3], [4, 1, 6], [7, 1, 9]]
```

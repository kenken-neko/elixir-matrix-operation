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
    {:matrix_operation, "~> 0.3.0"}
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
* Jacobi method (Iteration method to solve n-th eigen equation)
```elixir
MatrixOperation.jacobi([[10, 3, 2], [3, 5, 1], [2, 1, 0]], 100)
[
  [11.827601654846498, 3.5956497715829547, -0.4232514264294592],
  [
    [0.8892913734834387, -0.41794841208075917, -0.1856878506717961],
    [0.4229077692904142, 0.9060645461356799, -0.014002032343986153],
    [0.17409730532592232, -0.06607694813719736, 0.982509015329186]
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

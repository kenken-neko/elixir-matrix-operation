# elixir-matrix-operation
*MatrixOperation* is a linear algebra library in Elixir language. For example, this library can be used to solve eigenvalue equations.  
You can refer to the online documentation at https://hexdocs.pm/matrix_operation/MatrixOperation.html#content and mathematical description at 'docs/latex_out/numerical_formula.pdf' in this package.  

## Notice
Matrix indices of a row and column is an integer starting from 1 (not from 0).
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
    {:matrix_operation, "~> 0.3.7"}
  ]
end
```

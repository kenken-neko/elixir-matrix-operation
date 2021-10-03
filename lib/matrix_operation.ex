defmodule MatrixOperation do
  @moduledoc """
    *MatrixOperation* is a linear algebra library in Elixir language.
    Matrix indices of a row and column is an integer starting from 1 (not from 0).
  """

  @doc """
    Numbers of rows and columns of a matrix are got.
    #### Argument
      - matrix: Target matrix for finding the numbers of rows and columns.
    #### Output
      {num_rows, num_cols}: Numbers of rows and columns of a matrix
    #### Example
        iex> MatrixOperation.size([[3, 2, 3], [2, 1, 2]])
        {2, 3}
    """
  def size(matrix) when is_list(hd(matrix)) do
    col_num = Enum.map(matrix, &size_sub(&1, 0))
    max_num = Enum.max(col_num)
    if(max_num == Enum.min(col_num), do: {length(matrix), max_num}, else: nil)
  end

  def size(_matrix) do
    nil
  end

  defp size_sub(row, i) when i != length(row) do
    if(is_number(Enum.at(row, i)), do: size_sub(row, i + 1), else: nil)
  end

  defp size_sub(row, i) when i == length(row) do
    i
  end

  @doc """
    A n-th unit matrix is got.
    #### Argument
      - n: Number of rows / columns in the unit matrix to output.
    #### Output
      A n-th unit matrix
    #### Example
        iex> MatrixOperation.unit_matrix(3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
  def unit_matrix(n) when n > 0 and is_integer(n) do
    idx_list = Enum.to_list(1..n)
    Enum.map(idx_list, fn x -> Enum.map(idx_list, &unit_matrix_sub(x, &1)) end)
  end

  defp unit_matrix_sub(i, j) when i == j do
    1
  end

  defp unit_matrix_sub(_i, _j) do
    0
  end

  @doc """
    A m×n matrix having even-elements is got.
    #### Argument
      - elem: Value of the common element of the matrix to output.
      - {row_num, col_num}: Size of the matrix to output.
    #### Output
      A row_num×col_num matrix having even elements
    #### Example
        iex> MatrixOperation.even_matrix(0, {2, 3})
        [[0, 0, 0], [0, 0, 0]]
        iex> MatrixOperation.even_matrix(1, {3, 2})
        [[1, 1], [1, 1], [1, 1]]
    """
  def even_matrix(elem, {row_num, col_num})
    when row_num > 0 and col_num > 0 and is_number(elem) do
    List.duplicate(elem, col_num)
    |> List.duplicate(row_num)
  end

  def even_matrix(_elem, _size) do
    nil
  end

  @doc """
    A m×n matrix having random elements is got.
    #### Argument
      - min_val: Minimum value of random number.
      - max_val: Maximum value of random number.
      - {row_num, col_num}: Size of the matrix to output.
      - type: Data type of elements. "int" or "real".
    #### Output
      A row_num×col_num matrix having random elements
    """
  def random_matrix(min_val, max_val, {row_num, col_num}, type \\ "int")
    when row_num > 0 and col_num > 0 and max_val > min_val do
    Enum.to_list(1..row_num)
    |> Enum.map(
        fn _ ->
          Enum.map(
            Enum.to_list(1..col_num), & &1 * 0 + random_element(min_val, max_val, type)
          )
        end
      )
  end

  def random_matrix(_min_val, _max_val, _size, _type) do
    nil
  end

  defp random_element(min_val, max_val, "int") do
    Enum.random(min_val..max_val)
  end

  defp random_element(min_val, max_val, "real") do
    const = 10000000
    min_val_real = min_val * const
    max_val_real = max_val * const
    Enum.random(min_val_real..max_val_real) / const
  end

  @doc """
    An element of a matrix is got.
    #### Argument
      - matrix: Target matrix from which to extract the element.
      - {row_idx, col_idx}: Index of row and column of the element to be extracted.
    #### Output
      An element of a matrix
    #### Example
        iex> MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], {1, 1})
        1
    """
  def get_one_element(matrix, {row_idx, col_idx}) do
    matrix
    |> Enum.at(row_idx - 1)
    |> Enum.at(col_idx - 1)
  end

  @doc """
    A row of a matrix is got.
    #### Argument
      - matrix: Target matrix from which to extract the row.
      - row_idx: Index of the row to be extracted.
    #### Output
      A row of a matrix
    #### Example
        iex> MatrixOperation.get_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
        [1, 2, 3]
    """
  def get_one_row(matrix, row_idx) do
    matrix
    |> Enum.at(row_idx - 1)
  end

  @doc """
    A column of a matrix is got.
    #### Argument
      - matrix: Target matrix from which to extract the column.
      - col_idx: Index of the column to be extracted.
    #### Output
      A column of a matrix
    #### Example
        iex> MatrixOperation.get_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
        [1, 4, 7]
    """
  def get_one_column(matrix, col_idx) do
    matrix
    |> transpose()
    |> Enum.at(col_idx - 1)
  end

  @doc """
    A row of a matrix is deleted.
    #### Argument
      - matrix: Target matrix from which to delete the row.
      - del_idx: Index of the row to be deleted.
    #### Output
      The matrix from which the specified row was deleted.
    #### Example
        iex> MatrixOperation.delete_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
        [[1, 2, 3], [4, 5, 6]]
    """
  def delete_one_row(matrix, del_idx) do
    matrix
    |> Enum.with_index()
    |> Enum.reject(fn {_x, idx} -> idx == del_idx - 1 end)
    |> Enum.map(fn {x, _idx} -> x end)
  end

  @doc """
    A column of a matrix is deleted.
    #### Argument
      - matrix: Target matrix from which to delete the column.
      - del_idx: Index of the column to be deleted.
    #### Output
      The matrix from which the specified column was deleted.
    #### Example
        iex> MatrixOperation.delete_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
        [[1, 3], [4, 6], [7, 9]]
    """
  def delete_one_column(matrix, del_idx) do
    matrix
    |> transpose()
    |> Enum.with_index()
    |> Enum.reject(fn {_x, idx} -> idx == del_idx - 1 end)
    |> Enum.map(fn {x, _idx} -> x end)
    |> transpose()
  end

  @doc """
    A row of a matrix is exchanged.
    #### Argument
      - matrix: Target matrix from which to exchange the row.
      - exchange_idx: Index of the row to be exchanged.
      - exchange_list: List of the row to be exchanged.
    #### Output
      The matrix from which the specified row was exchanged.
    #### Example
        iex> MatrixOperation.exchange_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, [1, 1, 1])
        [[1, 2, 3], [4, 5, 6], [1, 1, 1]]
    """
  def exchange_one_row(matrix, exchange_idx, exchange_list) do
    matrix
    |> Enum.with_index()
    |> Enum.map(fn {x, idx} -> if(idx == exchange_idx - 1, do: exchange_list, else: x) end)
  end

  @doc """
    A column of a matrix is exchanged.
    #### Argument
      - matrix: Target matrix from which to exchange the column.
      - exchange_idx: Index of the column to be exchanged.
      - exchange_list: List of the column to be exchanged.
    #### Output
      The matrix from which the specified column was exchanged.
    #### Example
        iex> MatrixOperation.exchange_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, [1, 1, 1])
        [[1, 1, 3], [4, 1, 6], [7, 1, 9]]
    """
  def exchange_one_column(matrix, exchange_idx, exchange_list) do
    matrix
    |> transpose
    |> Enum.with_index()
    |> Enum.map(fn {x, idx} -> if(idx == exchange_idx - 1, do: exchange_list, else: x) end)
    |> transpose()
  end

  @doc """
    Transpose of a matrix
    #### Argument
      - matrix: Target matrix to transpose.
    #### Output
      Transposed matrix
    #### Example
        iex> MatrixOperation.transpose([[1.0, 2.0], [3.0, 4.0]])
        [[1.0, 3.0], [2.0, 4.0]]
    """
  def transpose(matrix) do
    Enum.zip(matrix)
    |> Enum.map(&Tuple.to_list(&1))
  end

  @doc """
    Trace of a matrix
    #### Argument
      - matrix: Target matrix to output trace.
    #### Output
      Trance of the matrix
    #### Example
        iex> MatrixOperation.trace([[1.0, 2.0], [3.0, 4.0]])
        5.0
    """
  def trace(matrix) do
    {row_num, col_num} = size(matrix)
    matrix_with_idx = add_index(matrix)

    Enum.map(matrix_with_idx, &trace_sub(&1, row_num, col_num))
    |> Enum.sum()
  end

  defp trace_sub(_, row_num, col_num) when row_num != col_num do
    nil
  end

  defp trace_sub([idx, row], _row_num, _col_num) do
    Enum.at(row, idx - 1)
  end

  @doc """
    A determinant of a n×n square matrix is got.
    #### Argument
      - matrix: Target matrix to output determinant.
    #### Output
      Determinant of the matrix
    #### Example
        iex> MatrixOperation.determinant([[1, 2, 1], [2, 1, 0], [1, 1, 2]])
        -5
        iex> MatrixOperation.determinant([[1, 2, 1, 1], [2, 1, 0, 1], [1, 1, 2, 1], [1, 2, 3, 4]])
        -13
        iex> MatrixOperation.determinant([ [3,1,1,2,1], [5,1,3,4,1], [2,0,1,0,1], [1,3,2,1,1], [1,1,1,1,1] ])
        -14
    """
  def determinant(matrix) do
    determinant_sub(1, matrix)
  end

  # 1×1 matrix
  defp determinant_sub(_, matrix) when length(matrix) == 1 do
    Enum.at(matrix, 0)
    |> Enum.at(0)
  end

  # 2×2 matrix
  defp determinant_sub(co, [[a11, a12], [a21, a22]]) do
    co * (a11 * a22 - a12 * a21)
  end

  # 3×3 or over matrix
  defp determinant_sub(co, matrix) do
    matrix_with_idx = add_index(matrix)

    Enum.map(
      matrix_with_idx,
      &determinant_sub(
        (-1 + 2 * rem(hd(&1), 2)) * co * hd(Enum.at(&1, 1)),
        minor_matrix(matrix_with_idx, &1)
      )
    )
    |> Enum.sum()
  end

  defp minor_matrix(matrix_with_idx, row) do
    (matrix_with_idx -- [row])
    |> Enum.map(&Enum.at(&1, 1))
    |> Enum.map(&Enum.drop(&1, 1))
  end

  # add index
  defp add_index(matrix) do
    Stream.iterate(1, &(&1 + 1))
    |> Enum.zip(matrix)
    |> Enum.map(&(&1 |> Tuple.to_list()))
  end

  @doc """
    Cramer's rule
    #### Argument
      - matrix: Target matrix to perform Cramer's rule.
      - vertical_vec: Vertical vector to perform Cramer's rule.
      - select_idx: Index of the target to perform Cramer's rule.
    #### Output
      Solution to the linear equation when Cramer's rule is applied.
    #### Example
        iex> MatrixOperation.cramer([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1], [0], [0]], 1)
        1.0
        iex> MatrixOperation.cramer([[0, -2, 1], [-1, 1, -4], [3, 3, 1]], [[3], [-7], [4]], 1)
        2.0
    """
  def cramer(matrix, vertical_vec, select_idx) do
    [t] = transpose(vertical_vec)
    det = determinant(matrix)
    cramer_sub(matrix, t, select_idx - 1, det)
  end

  defp cramer_sub(_, _, _, nil), do: nil
  defp cramer_sub(_, _, _, 0), do: nil

  defp cramer_sub(a, t, select_idx, det) do
    rep_det = transpose(a) |> replace_element_in_list(select_idx, t, 0, []) |> determinant
    rep_det / det
  end

  defp replace_element_in_list(list, i, replace_element, i, output) when i < length(list) do
    replace_element_in_list(list, i, replace_element, i + 1, output ++ [replace_element])
  end

  defp replace_element_in_list(list, select_idx, replace_element, i, output)
       when i < length(list) do
    replace_element_in_list(
      list,
      select_idx,
      replace_element,
      i + 1,
      output ++ [Enum.at(list, i)]
    )
  end

  defp replace_element_in_list(list, _select_idx, _replace_element, i, output)
       when i == length(list),
       do: output

  @doc """
    Leading principal minor is generetaed.
    #### Argument
      - matrix: Target matrix to find leading principal minor.
      - idx: Index of a row and column to find leading principal minor.
    #### Output
      Leading principal minor
    #### Example
        iex> MatrixOperation.leading_principal_minor([[1, 3, 2], [2, 5, 1], [3, 4, 5]], 2)
        [[1, 3], [2, 5]]
    """
  def leading_principal_minor(matrix, idx) do
    matrix
    |> Enum.slice(0, idx)
    |> Enum.map(& Enum.slice(&1, 0, idx))
  end

  @doc """
    LU decomposition
    #### Argument
      - matrix: Target matrix to solve LU decomposition.
    #### Output
      {L, U}. L(U) is L(U)-matrix of LU decomposition.
    #### Example
        iex> MatrixOperation.lu_decomposition([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
        {
          [[1, 0, 0, 0], [2.0, 1, 0, 0], [3.0, 4.0, 1, 0], [-1.0, -3.0, 0.0, 1]],
          [[1, 1, 0, 3], [0, -1.0, -1.0, -5.0], [0, 0, 3.0, 13.0], [0, 0, 0, -13.0]]
        }
    """
  def lu_decomposition(matrix) do
    {row_num, col_num} = size(matrix)
    # check the setupufficient condition
    check_num = lu_decomposition_check(matrix, row_num, col_num)
    if(check_num == 0, do: nil, else: lu_decomposition_sub(matrix, 0, length(matrix), [], []))
  end

  defp lu_decomposition_check(_matrix, row_num, col_num) when row_num != col_num do
    nil
  end

  defp lu_decomposition_check(matrix, row_num, _col_num) do
    Enum.to_list(1..row_num)
    |> Enum.map(& leading_principal_minor(matrix, &1) |> determinant)
    |> Enum.reduce(fn x, acc -> x * acc end)
  end

  defp lu_decomposition_sub(matrix, k, matrix_len, _l_matrix, _u_matrix) when k == 0 do
    u_matrix = even_matrix(0, {matrix_len, matrix_len})
               |> exchange_one_row(1, hd(matrix))
    inverce_u11 = 1.0 / hd(hd(u_matrix))
    factor = matrix
    |> transpose()
    |> get_one_row(1)
    |> Enum.slice(1, matrix_len)
    l_row = [1] ++ hd(const_multiple(inverce_u11, [factor]))
    l_matrix = even_matrix(0, {matrix_len, matrix_len})
               |> exchange_one_row(1, l_row)
    lu_decomposition_sub(matrix, k + 1, matrix_len, l_matrix, u_matrix)
  end

  defp lu_decomposition_sub(matrix, k, matrix_len, l_matrix, u_matrix) when k != matrix_len do
    t_matrix = transpose(matrix)
    u_solve = u_cal(matrix, k, matrix_len, l_matrix, u_matrix)
    u_matrix_2 = exchange_one_row(u_matrix, k + 1, u_solve)
    l_solve = l_cal(t_matrix, k, matrix_len, l_matrix, u_matrix_2)
    l_matrix_2 = exchange_one_row(l_matrix, k + 1, l_solve)
    lu_decomposition_sub(matrix, k + 1, matrix_len, l_matrix_2, u_matrix_2)
  end

  defp lu_decomposition_sub(_matrix, _k, _matrix_len, l_matrix, u_matrix) do
    {transpose(l_matrix), u_matrix}
  end

  defp l_cal(t_matrix, k, matrix_len, l_matrix, u_matrix) do
    factor = Enum.at(t_matrix, k) |> Enum.slice(k + 1, matrix_len)
    u_extract = transpose(u_matrix) |> Enum.at(k)
    l_row = transpose(l_matrix)
    |> Enum.slice(k + 1, matrix_len)
    |> Enum.map(& inner_product(&1, u_extract))
    |> Enum.zip(factor)
    |> Enum.map(fn {x, y} -> y - x end)

    inverce_uii = 1.0 / Enum.at(Enum.at(u_matrix, k), k)
    [l_row_2] = const_multiple(inverce_uii, [l_row])
    [1] ++ l_row_2
    |> add_zero_element(0, k)
  end

  defp u_cal(matrix, k, matrix_len, l_matrix, u_matrix) do
    factor = Enum.at(matrix, k) |> Enum.slice(k, matrix_len)
    l_extract = transpose(l_matrix) |> Enum.at(k)
    transpose(u_matrix)
    |> Enum.slice(k, matrix_len)
    |> Enum.map(& inner_product(&1, l_extract))
    |> Enum.zip(factor)
    |> Enum.map(fn {x, y} -> y - x end)
    |> add_zero_element(0, k)
  end

  defp add_zero_element(list, init, fin) when init != fin do
    add_zero_element([0] ++ list, init + 1, fin)
  end

  defp add_zero_element(list, _init, _fin) do
    list
  end

  @doc """
    Linear equations are solved by LU decomposition.
    #### Argument
      - matrix: Target matrix to solve simultaneous linear equations.
      - vertical_vec: Vertical vector to solve linear equations.
    #### Output
      Solutions of the linear equations
    #### Example
        iex> MatrixOperation.solve_sle([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1], [0], [0]])
        [1.0, 0.0, 0.0]
        iex> MatrixOperation.solve_sle([[4, 1, 1], [1, 3, 1], [2, 1, 5]], [[9], [10], [19]])
        [1.0, 2.0, 3.0]
    """
  def solve_sle(matrix, vertical_vec) do
    # check the setupufficient condition
    if determinant(matrix) == 0 do
      nil
    else
      [t] = transpose(vertical_vec)
      solve_sle_sub(matrix, t)
    end
  end

  defp solve_sle_sub(matrix, t) do
    {l_matrix, u_matrix} = lu_decomposition(matrix)
    dim = length(l_matrix)
    y = forward_substitution(l_matrix, t, [], 0, dim)
    backward_substitution(u_matrix, y, [], dim, dim)
  end

  defp forward_substitution(l_matrix, t, _y, k, dim) when k == 0 do
    forward_substitution(l_matrix, t, [hd(t)], k + 1, dim)
  end

  defp forward_substitution(l_matrix, t, y, k, dim) when k != dim do
    l_extract = Enum.at(l_matrix, k) |> Enum.slice(0, k)
    y_extract = y |> Enum.slice(0, k)
    ly = inner_product(l_extract, y_extract)
    t_ly = Enum.at(t, k) - ly
    forward_substitution(l_matrix, t, y ++ [t_ly], k + 1, dim)
  end

  defp forward_substitution(_l_matrix, _t, y, k, dim) when k == dim do
    y
  end

  defp backward_substitution(u_matrix, y, _b, k, dim) when k == dim do
    dim_1 = dim - 1
    y_n = Enum.at(y, dim_1)
    u_nn = Enum.at(Enum.at(u_matrix, dim_1), dim_1)
    backward_substitution(u_matrix, y, [y_n / u_nn], k - 1, dim)
  end

  defp backward_substitution(_, _, b, k, _) when k == 0 do
    b
  end

  defp backward_substitution(u_matrix, y, b, k, dim) when k != dim do
    k_1 = k - 1
    u_extract = Enum.at(u_matrix, k_1) |> Enum.slice(k, dim)
    lb = inner_product(u_extract, b)
    inverce_uii = Enum.at(Enum.at(u_matrix, k_1), k_1)
    t_lb = (Enum.at(y, k_1) - lb) / inverce_uii
    backward_substitution(u_matrix, y, [t_lb] ++ b, k_1, dim)
  end

  @doc """
    A matrix is multiplied by a constant.
    #### Argument
      - const: Constant to multiply the matrix.
      - matrix: Target vector/matrix to be multiplied by a constant.
    #### Output
      Vector/Matrix multiplied by the constant.
    #### Example
        iex> MatrixOperation.const_multiple(-1, [1.0, 2.0, 3.0])
        [-1.0, -2.0, -3.0]
        iex> MatrixOperation.const_multiple(2, [[1, 2, 3], [2, 2, 2], [3, 8, 9]])
        [[2, 4, 6], [4, 4, 4], [6, 16, 18]]
    """
  def const_multiple(const, x) when is_number(x) do
    const * x
  end

  def const_multiple(const, x) when is_list(x) do
    Enum.map(x, &const_multiple(const, &1))
  end

  @doc """
    A matrix is added by a constant.
    #### Argument
      - const: Constant to add the matrix.
      - matrix: Target vector/matrix to be added by a constant.
    #### Output
      Vector/Matrix multiplied by the constant.
    #### Example
        iex> MatrixOperation.const_addition(1, [1.0, 2.0, 3.0])
        [2.0, 3.0, 4.0]
        iex> MatrixOperation.const_addition(1, [[1, 2, 3], [2, 2, 2], [3, 8, 9]])
        [[2, 3, 4], [3, 3, 3], [4, 9, 10]]
    """
  def const_addition(const, x) when is_number(x) do
    const + x
  end

  def const_addition(const, x) when is_list(x) do
    Enum.map(x, &const_addition(const, &1))
  end

  @doc """
    Inverse Matrix
    #### Argument
      - matrix: Matrix to be inverse Matrix.
    #### Output
      Inverse Matrix
    #### Example
        iex> MatrixOperation.inverse_matrix([[1, 1, -1], [-2, -1, 1], [-1, -2, 1]])
        [[-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [-3.0, -1.0, -1.0]]
    """
  def inverse_matrix(matrix) when is_list(hd(matrix)) do
    det = determinant(matrix)

    create_index_matrix(matrix)
    |> Enum.map(&map_index_row(matrix, det, &1))
    |> transpose()
  end

  def inverse_matrix(_) do
    nil
  end

  defp create_index_matrix(matrix) do
    idx_list = Enum.to_list(1..length(matrix))
    Enum.map(idx_list, fn x -> Enum.map(idx_list, &[x, &1]) end)
  end

  defp map_index_row(_matrix, det, _row) when det == 0 do
    nil
  end

  defp map_index_row(matrix, det, row) do
    Enum.map(row, &minor_matrix(matrix, det, &1))
  end

  defp minor_matrix(matrix, det, [row_num, col_num]) do
    det_temp_matrix =
      delete_one_row(matrix, row_num)
      |> transpose
      |> delete_one_row(col_num)
      |> determinant

    if(rem(row_num + col_num, 2) == 0,
      do: det_temp_matrix / det,
      else: -1 * det_temp_matrix / det
    )
  end

  @doc """
    Matrix product
    #### Argument
      - a: Left side of the product of matrices.
      - b: Right side of the product of matrices.
    #### Output
      Product of two matrices
    #### Example
        iex> MatrixOperation.product([[3, 2, 3], [2, 1, 2]], [[2, 3], [2, 1], [3, 5]])
        [[19, 26], [12, 17]]
    """
  def product(a, b) do
    check_product(a, b)
  end

  defp check_product(a, b) do
    {_, col_num_a} = size(a)
    {row_num_b, _} = size(b)
    if(col_num_a == row_num_b, do: product_sub(a, b), else: nil)
  end

  defp product_sub(a, b) do
    Enum.map(a, fn row_a ->
      transpose(b)
      |> Enum.map(&inner_product(row_a, &1))
    end)
  end

  defp inner_product(row_a, col_b) do
    Enum.zip(row_a, col_b)
    |> Enum.map(&Tuple.to_list(&1))
    |> Enum.map(&Enum.reduce(&1, fn x, acc -> x * acc end))
    |> Enum.sum()
  end

  @doc """
    Matrix addition
    #### Argument
      - a: Left side of the addition of matrices.
      - b: Right side of the addition of matrices.
    #### Output
      Addition of two matrices
    #### Example
        iex> MatrixOperation.add([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[5, 5, 4], [5, 3, 4]]
    """
  def add(a, b) do
    check_add(a, b)
  end

  defp check_add(a, b) do
    size_a = size(a)
    size_b = size(b)
    if(size_a == size_b, do: add_sub(a, b), else: nil)
  end

  defp add_sub(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} ->
      Enum.zip(x, y)
      |> Enum.map(&Tuple.to_list(&1))
      |> Enum.map(&Enum.reduce(&1, fn x, acc -> x + acc end))
    end)
  end

  @doc """
    Matrix subtraction
    #### Argument
      - a: Left side of the subtraction of matrices.
      - b: Right side of the subtraction of matrices.
    #### Output
      Subtraction of two matrices
    #### Example
        iex> MatrixOperation.subtract([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[1, -1, 2], [-1, -1, 0]]
    """
  def subtract(a, b) do
    check_subtract(a, b)
  end

  defp check_subtract(a, b) do
    size_a = size(a)
    size_b = size(b)
    if(size_a == size_b, do: subtract_sub(a, b), else: nil)
  end

  defp subtract_sub(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} ->
      Enum.zip(x, y)
      |> Enum.map(&Tuple.to_list(&1))
      |> Enum.map(&Enum.reduce(&1, fn x, acc -> acc - x end))
    end)
  end

  @doc """
    Hadamard product
    #### Argument
      - a: Left side of the Hadamard production of matrices.
      - b: Right side of the Hadamard production of matrices.
    #### Output
      Hadamard production of two matrices
    #### Example
        iex> MatrixOperation.hadamard_product([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[6, 6, 3], [6, 2, 4]]
    """
  def hadamard_product(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> hadamard_product_sub(x, y) end)
  end

  defp hadamard_product_sub(row_a, row_b) do
    Enum.zip(row_a, row_b)
    |> Enum.map(&Tuple.to_list(&1))
    |> Enum.map(&Enum.reduce(&1, fn x, acc -> x * acc end))
  end

  @doc """
    Hadamard division
    #### Argument
      - a: Left side of the Hadamard division of matrices.
      - b: Right side of the Hadamard division of matrices.
    #### Output
      Hadamard division of two matrices
    #### Example
        iex> MatrixOperation.hadamard_division([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[1.5, 0.6666666666666666, 3.0], [0.6666666666666666, 0.5, 1.0]]
    """
  def hadamard_division(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> hadamard_division_sub(x, y) end)
  end

  defp hadamard_division_sub(row_a, row_b) do
    Enum.zip(row_a, row_b)
    |> Enum.map(&Tuple.to_list(&1))
    |> Enum.map(&Enum.reduce(&1, fn x, acc -> acc / x end))
  end

  @doc """
    Hadamard power
    #### Argument
      - matrix: Target matrix that elements are to be n-th powered.
      - n: Exponent of a power.
    #### Output
      Matrix that elements are to be n-th powered
    #### Example
        iex> MatrixOperation.hadamard_power([[3, 2, 3], [2, 1, 2]], 2)
        [[9.0, 4.0, 9.0], [4.0, 1.0, 4.0]]
    """
  def hadamard_power(matrix, n) do
    Enum.map(matrix, &Enum.map(&1, fn x -> :math.pow(x, n) end))
  end

  @doc """
    Tensor product
    #### Argument
      - a: Left side of the tensor production of matrices.
      - b: Right side of the tensor production of matrices.
    #### Output
      Tensor production of two matrices
    #### Example
        iex> MatrixOperation.tensor_product([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [2, 1, 2], [3, 5, 3]])
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
    """
  def tensor_product(a, b) when is_list(a) do
    Enum.map(a, &tensor_product(&1, b))
  end

  def tensor_product(a, b) when is_number(a) do
    const_multiple(a, b)
  end

  @doc """
    Calculate eigenvalue using algebra method [R^2×R^2/R^3×R^3 matrix]
    #### Argument
      - [[a11, a12], [a21, a22]] or [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]:
        R^2×R^2/R^3×R^3 matrix
    #### Output
      Eigenvalues which is a non-trivial value other than zero.
    #### Example
        iex> MatrixOperation.eigenvalue_algebra([[3, 1], [2, 2]])
        {4.0, 1.0}
        iex> MatrixOperation.eigenvalue_algebra([[6, -3], [4, -1]])
        {3.0, 2.0}
        iex> MatrixOperation.eigenvalue_algebra([[1, 1, 1], [1, 2, 1], [1, 2, 3]])
        {4.561552806429505, 0.43844714673139706, 1.0000000468390973}
        iex> MatrixOperation.eigenvalue_algebra([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
        {3.0000000027003626, 0.9999999918989121}
    """
  # 2×2 algebra method
  def eigenvalue_algebra([[a11, a12], [a21, a22]]) do
    quadratic_formula(1, -a11 - a22, a11 * a22 - a12 * a21)
    |> exclude_zero_eigenvalue()
    |> List.to_tuple()
  end

  # 3×3 algebratic method
  def eigenvalue_algebra([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]) do
    a = -1
    b = a11 + a22 + a33
    c = a21 * a12 + a13 * a31 + a32 * a23 - a11 * a22 - a11 * a33 - a22 * a33
    d =
      a11 * a22 * a33 + a12 * a23 * a31 + a13 * a32 * a21 - a11 * a32 * a23 - a22 * a31 * a13 -
        a33 * a21 * a12

    dis = -4 * a * c * c * c - 27 * a * a * d * d + b * b * c * c + 18 * a * b * c * d - 4 * b * b * b * d
    if(dis > 0, do: cubic_formula(a, b, c, d), else: nil)
    |> exclude_zero_eigenvalue()
    |> List.to_tuple()
  end

  def eigenvalue_algebra(_a) do
    "2×2 or 3×3 matrix only"
  end

  defp quadratic_formula(a, b, c) do
    quadratic_formula_sub(a, b, c)
  end

  defp quadratic_formula_sub(a, b, c) when b * b < 4 * a * c do
    nil
  end

  defp quadratic_formula_sub(a, b, c) do
    d = :math.sqrt(b * b - 4 * a * c)
    [0.5 * (-b + d) / a, 0.5 * (-b - d) / a]
  end

  defp cubic_formula(a, b, c, d)
      when -4 * a * c * c * c - 27 * a * a * d * d + b * b * c * c + 18 * a * b * c * d -
             4 * b * b * b * d < 0 do
    nil
  end

  defp cubic_formula(a, b, c, d) do
    ba = b / a
    ca = c / a
    da = d / a

    const1 = (27 * da + 2 * ba * ba * ba - 9 * ba * ca) / 54
    const2 = cubic_formula_sub(const1 * const1 + :math.pow((3 * ca - ba * ba) / 9, 3))
    const_plus = csqrt([-const1 + Enum.at(const2, 0), Enum.at(const2, 1)], 3)
    const_minus = csqrt([-const1 - Enum.at(const2, 0), -Enum.at(const2, 1)], 3)
    root3 = :math.sqrt(3)

    x1 = Enum.at(const_plus, 0) + Enum.at(const_minus, 0) - ba / 3

    x2 =
      -0.5 * Enum.at(const_plus, 0) - 0.5 * root3 * Enum.at(const_plus, 1) -
        0.5 * Enum.at(const_minus, 0) + 0.5 * root3 * Enum.at(const_minus, 1) - ba / 3

    x3 =
      -0.5 * Enum.at(const_plus, 0) + 0.5 * root3 * Enum.at(const_plus, 1) -
        0.5 * Enum.at(const_minus, 0) - 0.5 * root3 * Enum.at(const_minus, 1) - ba / 3

    [x1, x2, x3]
    |> Enum.map(& zero_approximation(&1))
  end

  defp cubic_formula_sub(x) when x < 0 do
    [0, :math.sqrt(-x)]
  end

  defp cubic_formula_sub(x) do
    [:math.sqrt(x), 0]
  end

  defp atan(x) when x < 0 do
    y = atan(-x)
    -1 * y
  end

  defp atan(x) do
    atan_sub(x, 0, 0)
  end

  defp atan_sub(x, z, s) when z < x do
    del = 0.0000001
    z = z + del
    s = s + del / (z * z + 1)
    atan_sub(x, z, s)
  end

  defp atan_sub(_, _, s) do
    s
  end

  defp csqrt([re, im], _n) when re == 0 and im == 0 do
    [0, 0]
  end

  defp csqrt([re, im], n) when re == 0 and im > 0 do
    r = :math.pow(im * im, 0.5 / n)
    re2 = r * :math.pow(3, 0.5) * 0.5
    im2 = r * 0.5
    [re2, im2]
  end

  defp csqrt([re, im], n) when re == 0 and im < 0 do
    r = :math.pow(im * im, 0.5 / n)
    re2 = r * :math.pow(3, 0.5) * 0.5
    im2 = -r * 0.5
    [re2, im2]
  end

  defp csqrt([re, im], n) when re < 0 do
    r = :math.pow(re * re + im * im, 0.5 / n)
    re2 = -r * :math.cos(atan(im / re) / n)
    im2 = r * :math.sin(atan(im / re) / n)
    [re2, im2]
  end

  defp csqrt([re, im], n) do
    r = :math.pow(re * re + im * im, 0.5 / n)
    re2 = r * :math.cos(atan(im / re) / n)
    im2 = r * :math.sin(atan(im / re) / n)
    [re2, im2]
  end

  # Due to a numerical calculation error
  defp zero_approximation(delta) when abs(delta) < 0.000001 do
    0
  end

  defp zero_approximation(delta) do
    delta
  end

  defp exclude_zero_eigenvalue(eigenvalues) do
    eigenvalues2 = Enum.map(eigenvalues, & zero_approximation(&1))
    len = length(eigenvalues)
    zero_list = Enum.to_list(1..len)
    |> Enum.map(& &1 * 0)
    eigenvalues2 -- zero_list
  end

  defp exclude_zero_eigenvalue(eigenvalues, eigenvectors) do
    Enum.map(eigenvalues, & zero_approximation(&1))
    |> Enum.zip(eigenvectors)
    |> Enum.map(fn {val, vec} -> if(val==0, do: nil, else: {val, vec}) end)
    |> Enum.filter(& !is_nil(&1))
    |> Enum.unzip()
  end

  """
    Matrix diagonalization using algebra method [R^2×R^2/R^3×R^3 matrix]
    #### Argument
      - matrix: R^2×R^2/R^3×R^3 matrix. Target matrix to be diagonalized.
    #### Output
      Diagonalized matrix
    #### Example
        iex> MatrixOperation.diagonalization_algebra([[1, 3], [4, 2]])
        [[5.0, 0], [0, -2.0]]
        iex> MatrixOperation.diagonalization_algebra([[2, 1, -1], [1, 1, 5], [-1, 2, 1]])
        [[-2.6170355131217935, 0, 0], [0, 4.1017849347870765, 0], [0, 0, 2.515250578334717]]
        iex> MatrixOperation.diagonalization_algebra([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
        nil
    """
  defp diagonalization_algebra(matrix) do
    ev = matrix
    |> eigenvalue_algebra()
    |> Tuple.to_list()
    if(length(ev)==length(matrix), do: ev, else: nil)
    |> diagonalization_algebra_condition()
  end

  defp diagonalization_algebra_condition(matrix) when matrix == nil do
    nil
  end

  defp diagonalization_algebra_condition(matrix) do
    matrix
    |> Enum.with_index()
    |> Enum.map(& diagonalization_algebra_sub(&1, length(matrix), 0, []))
  end

  defp diagonalization_algebra_sub(_, dim, i, row) when i + 1 > dim do
    row
  end

  defp diagonalization_algebra_sub({ev, index}, dim, i, row) when i != index do
    diagonalization_algebra_sub({ev, index}, dim, i + 1, row ++ [0])
  end

  defp diagonalization_algebra_sub({ev, index}, dim, i, row) when i == index do
    diagonalization_algebra_sub({ev, index}, dim, i + 1, row ++ [ev])
  end

  @doc """
    Jordan_normal_form [R^2×R^2/R^3×R^3 matrix]
    #### Argument
      - matrix: R^2×R^2/R^3×R^3 matrix. Target matrix to be Jordan normal form.
    #### Output
      Jordan normal form matrix
    #### Example
        iex> MatrixOperation.jordan_normal_form([[1, 3], [4, 2]])
        [[5.0, 0], [0, -2.0]]
        iex> MatrixOperation.jordan_normal_form([[7, 2], [-2, 3]])
        [[5.0, 1], [0, 5.0]]
        iex> MatrixOperation.jordan_normal_form([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
        nil
        iex> MatrixOperation.jordan_normal_form([[1, -1, 1], [0, 2, -2], [1, 1, 3]])
        [[2.0, 1, 0], [0, 2.0, 1], [0, 0, 2.0]]
        iex> MatrixOperation.jordan_normal_form([[3, 0, 1], [-1, 2, -1], [-1, 0, 1]])
        [[2.0, 1, 0], [0, 2.0, 0], [0, 0, 2.0]]
        iex> MatrixOperation.jordan_normal_form([[1, 0, -1], [0, 2, 0], [0, 1, 1]])
        [[2.0, 0, 0], [0, 0.9999999999999999, 1], [0, 0, 0.9999999999999999]]
        iex> MatrixOperation.jordan_normal_form([[6, 2, 3], [-3, 0, -2], [-4, -2, -1]])
        [[1.0, 0, 0], [0, 2.0, 1], [0, 0, 2.0]]
    """
  # R^2×R^2 matrix
  def jordan_normal_form([[m11, m12], [m21, m22]]) do
    b = -m11 - m22
    c = m11 * m22 - m12 * m21
    jordan_R2R2(b, c, [[m11, m12], [m21, m22]])
  end
  # R^3×R^3 matrix
  def jordan_normal_form([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]) do
    b = m11 + m22 + m33
    c = m21 * m12 + m13 * m31 + m32 * m23 - m11 * m22 - m11 * m33 - m22 * m33
    d =
      m11 * m22 * m33 + m12 * m23 * m31 + m13 * m32 * m21 - m11 * m32 * m23 - m22 * m31 * m13 -
        m33 * m21 * m12
    jordan_R3R3(b, c, d, [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
  end

  def jordan_normal_form(_) do
    nil
  end

  defp jordan_R2R2(b, c, m) when (b * b > 4 * c) do
    diagonalization_algebra(m)
  end

  defp jordan_R2R2(b, c, m) when b * b == 4 * c do
    m_lambda = subtract(m, [[-b * 0.5, 0], [0, -b * 0.5]])
    max_jordan_dim = jordan_R2R2_sub(m_lambda, 1)
    jordan_R2R2_sub2(b, max_jordan_dim)
  end

  defp jordan_R2R2(_, _, _) do
    nil
  end

  defp jordan_R2R2_sub(ml, n) when ml != [[0, 0], [0, 0]] and n <= 2 do
    product(ml, ml)
    |> jordan_R2R2_sub(n + 1)
  end

  defp jordan_R2R2_sub(_, n) when n > 2 do
    nil
  end

  defp jordan_R2R2_sub(_, n) do
    n
  end

  defp jordan_R2R2_sub2(b, mjd) when mjd == 2 do
    [[-b * 0.5, 1], [0, -b * 0.5]]
  end

  defp jordan_R2R2_sub2(b, mjd) when mjd == 1 do
    [[-b * 0.5, 0], [0, -b * 0.5]]
  end

  defp jordan_R2R2_sub2(_, _) do
    nil
  end

  defp jordan_R3R3(b, c, d, m)
    when 4 * c * c * c - 27 * d * d + b * b * c * c - 18 * b * c * d -
           4 * b * b * b * d > 0 do
    diagonalization_algebra(m)
  end
  # Triple root
  defp jordan_R3R3(b, c, d, m)
    when (4 * c * c * c - 27 * d * d + b * b * c * c - 18 * b * c * d -
           4 * b * b * b * d == 0) and (b * b == -3 * c and b * b * b == 27 * d) do
    m_lambda = subtract(m, [[b/3, 0, 0], [0, b/3, 0], [0, 0, b/3]])
    max_jordan_dim = jordan_R3R3_sub(m_lambda, 1)
    jordan_R3R3_sub2(b, max_jordan_dim)
  end
  # Double root
  defp jordan_R3R3(b, c, d, _)
    when (4 * c * c * c - 27 * d * d + b * b * c * c - 18 * b * c * d -
           4 * b * b * b * d == 0) do
    lambda = cubic_formula(-1, b, c, d)
    jordan_R3R3_sub3(lambda)
  end

  defp jordan_R3R3(_, _, _, _) do
    nil
  end

  defp jordan_R3R3_sub(ml, n) when ml != [[0, 0, 0], [0, 0, 0], [0, 0, 0]] and n < 3 do
    product(ml, ml)
    |> Enum.map(& Enum.map(&1, fn x -> zero_approximation(x) end))
    |> jordan_R3R3_sub(n + 1)
  end

  defp jordan_R3R3_sub(_, n) when n > 3 do
    nil
  end

  defp jordan_R3R3_sub(_, n) do
    n
  end

  defp jordan_R3R3_sub2(b, mjd) when mjd == 3 do
    [[b/3, 1, 0], [0, b/3, 1], [0, 0, b/3]]
  end

  defp jordan_R3R3_sub2(b, mjd) when mjd == 2 do
    [[b/3, 1, 0], [0, b/3, 0], [0, 0, b/3]]
  end

  defp jordan_R3R3_sub2(b, mjd) when mjd == 1 do
    [[b/3, 0, 0], [0, b/3, 0], [0, 0, b/3]]
  end

  defp jordan_R3R3_sub2(_, _) do
    nil
  end

  defp jordan_R3R3_sub3([l1, l2, l3]) when l1 == l2 do
    [[l1, 1, 0], [0, l2, 0], [0, 0, l3]]
  end

  defp jordan_R3R3_sub3([l1, l2, l3]) when l2 == l3 do
    [[l1, 0, 0], [0, l2, 1], [0, 0, l3]]
  end

  defp jordan_R3R3_sub3([l1, l2, l3]) when l1 == l3 do
    [[l1, 1, 0], [0, l3, 0], [0, 0, l2]]
  end

  defp jordan_R3R3_sub3(_) do
    nil
  end

  @doc """
    Power iteration method (maximum eigen value and eigen vector)
    #### Argument
      - matrix: Matrix to adapt the power iteration method.
      - iter_max: iteration number of the power iteration method. The default value is 1000.
    #### Output
      Maximum eigenvalue and normalized eigenvector corresponding to the maximum eigenvalue
    #### Example
        iex> MatrixOperation.power_iteration([[3, 1], [2, 2]])
        {
          4.0,
          [0.7071067811865476, 0.7071067811865476]
        }
        iex> MatrixOperation.power_iteration([[1, 1, 2], [0, 2, -1], [0, 0, 3]])
        {
          3.0,
          [0.3333333333333333, -0.6666666666666666, 0.6666666666666666]
        }
    """
  def power_iteration(matrix, iter_max \\ 1000) do
    init_vec = random_column(length(matrix))
    xk_pre = power_iteration_sub(matrix, init_vec, iter_max)
    # eigen vector
    [xk_vec] = product(matrix, xk_pre) |> transpose
    [xk_pre_vec] = transpose(xk_pre)
    # eigen value
    eigen_value = inner_product(xk_vec, xk_vec) / inner_product(xk_vec, xk_pre_vec)
    norm_xk_vec = :math.sqrt(inner_product(xk_vec, xk_vec))
    normalized_eigen_vec = Enum.map(xk_vec, & &1/norm_xk_vec)
    {eigen_value, normalized_eigen_vec}
  end

  defp random_column(num) when num > 1 do
    tmp = Enum.reduce(1..num, [], fn _, acc -> [Enum.random(0..50000) / 10000 | acc] end)
    transpose([tmp])
  end

  defp random_column(_num) do
    nil
  end

  defp power_iteration_sub(matrix, v, iter_max) do
    # Normarization is for overflow suppression
    Enum.reduce(1..iter_max, v, fn _, acc ->
      vp = product(matrix, acc)
      [vpt] = transpose(vp)
      const_multiple(1 / :math.sqrt(inner_product(vpt, vpt)), vp)
    end)
  end

  @doc """
    Calculate eigenvalues and eigenvectors by using Jacobi method
    #### Argument
      - matrix: Matrix to adapt the power iteration method.
      - iter_max: iteration number of the power iteration method. The default value is 1000.
    #### Output
      [Eigenvalues list, Eigenvectors list]: Eigenvalues and eigenvectors
    #### Example
        iex> MatrixOperation.jacobi([[10, 3, 2], [3, 5, 1], [2, 1, 0]])
        {
          [11.827601656660915, 3.5956497715829547, -0.42325142824210527],
          [
            [0.8892872578006493, -0.42761854121985043, -0.16220529066103917],
            [0.4179466723082575, 0.9038581385546461, -0.09143874712126684],
            [0.1857114757355714, 0.013522151221627882, 0.982511271796136]
          ]
        }
    """
  def jacobi(matrix, iter_max \\ 1000) do
    [pap, p] = jacobi_iteration(matrix, iter_max, 0, unit_matrix(length(matrix)))
    p_rnd = Enum.map(p, & Enum.map(&1, fn x -> zero_approximation(x) end))

    pap
    |> Enum.with_index()
    |> Enum.map(& jacobi_sub4(&1))
    |> Enum.map(& zero_approximation(&1))
    |> exclude_zero_eigenvalue(p_rnd)
  end

  defp jacobi_iteration(matrix, iter_max, l, p_pre) when l != iter_max do
    {row_num, col_num} = size(matrix)
    odts = off_diagonal_terms(matrix, row_num, col_num, 0, 0, [])
    |> Enum.map(& abs(&1))

    max_odt = Enum.max(odts)
    [max_i, max_j] = Enum.with_index(odts)
    |> jocobi_sub(max_odt, 0)
    |> jocobi_sub2(col_num, 0)

    a_ij = get_one_element(matrix, {max_i + 1, max_j + 1})
    a_ii = get_one_element(matrix, {max_i + 1, max_i + 1})
    a_jj = get_one_element(matrix, {max_j + 1, max_j + 1})
    phi = phi_if(a_ii - a_jj, a_ij)

    p = jacobi_sub3(phi, col_num, max_i, max_j, 0, 0, [], [])
    p_pi = product(p_pre, p)
    p
    |> transpose()
    |> product(matrix)
    |> product(p)
    |> jacobi_iteration(iter_max, l + 1, p_pi)
  end

  defp jacobi_iteration(matrix, _, _, p) do
    [matrix, p]
  end

  defp phi_if(denominator, a_ij) when denominator < 0.0000001 and a_ij > 0 do
    -0.78539816339 # -pi/2
  end

  defp phi_if(denominator, a_ij) when denominator < 0.0000001 and a_ij < 0 do
    0.78539816339 # -pi/2
  end

  defp phi_if(denominator, a_ij) do
    atan(-2 * a_ij / denominator) * 0.5
  end

  defp off_diagonal_terms(m, row_num, col_num, i, j, output) when i < j and row_num >= i and col_num > j do
    off_diagonal_terms(m, row_num, col_num, i, j + 1, output ++ [get_one_element(m, {i + 1, j + 1})])
  end

  defp off_diagonal_terms(m, row_num, col_num, i, j, output) when i < j and row_num > i and col_num == j do
    off_diagonal_terms(m, row_num, col_num, i + 1, 0, output)
  end

  defp off_diagonal_terms(_, row_num, col_num, i, j, output) when row_num == i and col_num == j do
    output
  end

  defp off_diagonal_terms(m, row_num, col_num, i, j, output) do
    off_diagonal_terms(m, row_num, col_num, i, j + 1, output)
  end

  defp jocobi_sub(element_idx_list, target_element, i) when hd(element_idx_list) == {target_element, i} do
    i
  end

  defp jocobi_sub(element_idx_list, target_element, i) do
    [_|tail] = element_idx_list
    jocobi_sub(tail, target_element, i + 1)
  end

  defp jocobi_sub2(idx, col_num, i) when idx < (i + 1) * col_num - ((i + 1) * (2 + i) * 0.5) do
    [max_i, max_j] = [i, idx - i * (2 * col_num - i - 1) * 0.5 + i + 1]
    [max_i, round(max_j)]
  end

  defp jocobi_sub2(idx, col_num, i) do
    jocobi_sub2(idx, col_num, i + 1)
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) when i == j and ( i == target_i or j == target_j) do
    jacobi_sub3(phi, col_num, target_i, target_j, i, j + 1, o_row ++ [:math.cos(phi)], output)
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) when i == target_i and j == target_j and j != col_num do
    jacobi_sub3(phi, col_num, target_i, target_j, i, j + 1, o_row ++ [:math.sin(phi)], output)
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) when i == target_i and j == target_j and j == col_num do
    jacobi_sub3(phi, col_num, target_i, target_j, i + 1, 0, [] , output ++ [o_row ++ [:math.sin(phi)]])
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) when i == target_j and j == target_i do
    jacobi_sub3(phi, col_num, target_i, target_j, i, j + 1, o_row ++ [:math.sin(-phi)], output)
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) when (i != target_i or j != target_j) and i == j and j != col_num do
    jacobi_sub3(phi, col_num, target_i, target_j, i, j + 1, o_row ++ [1], output)
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) when (i != target_i or j != target_j) and i != j and j == col_num do
    jacobi_sub3(phi, col_num, target_i, target_j, i + 1, 0, [], output ++ [o_row])
  end

  defp jacobi_sub3(_, col_num, _, _, i, j, _, output) when i == j and j == col_num do
    output
  end

  defp jacobi_sub3(phi, col_num, target_i, target_j, i, j, o_row, output) do
    jacobi_sub3(phi, col_num, target_i, target_j, i, j + 1, o_row ++ [0], output)
  end

  defp jacobi_sub4({list, index}) do
    Enum.at(list, index)
  end

  @doc """
    Singular Value Decomposition (SVD) using Jacobi method.
    #### Argument
      - matrix: Matrix to adapt the SVD by using the QR decomposition method.
    #### Output
      [Singular values, U-matrix, V-matrix]:
        Singular values, U-matrix and V-matrix.
        Singular value is a non-trivial value other than zero.
    #### Example
        iex> MatrixOperation.svd([[1, 0, 0], [0, 1, 1]])
        {
          [1.0, 1.4142135623730951],
          [
            [1.0, 0.0],
            [0.0, 1.0]
          ],
          [
            [1.0, 0.0, 0.0],
            [0.0, 0.7071067811865475, 0.7071067811865475]
          ]
        }
        iex> MatrixOperation.svd([[1, 1], [1, -1], [1, 0]])
        {
          [1.7320508075688772, 1.4142135623730951],
          [
            [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
            [-0.7071067811865476, 0.7071067811865476, 0.0]
          ],
          [[1.0, 0.0], [0.0, 1.0]]
        }
        iex> MatrixOperation.svd([[1, 1], [1, 1]])
        {
          [1.9999999999999998],
          [
            [0.7071067811865476, 0.7071067811865476]
          ],
          [
            [0.7071067811865476, 0.7071067811865476]
          ]
        }
    """
  def svd(a) do
    a_t = transpose(a)
    svd_sub(a, a_t)
  end

  def svd_sub(a, a_t) when length(a) <= length(a_t) do
    # U matrix
    aat = product(a, a_t)
    {sv_sq, u} = eigen(aat)
    # V matirx
    ata = product(a_t, a)
    {_, v} = eigen(ata)
    # Singular value
    s = Enum.map(sv_sq, & :math.sqrt(&1))
    # A = USV^t
    {s, u, v}
  end

  def svd_sub(a, a_t) do
    # U matrix
    aat = product(a, a_t)
    {_, u} = eigen(aat)
    # V matirx
    ata = product(a_t, a)
    {sv_sq, v} = eigen(ata)
    # Singular value
    s = Enum.map(sv_sq, & :math.sqrt(&1))
    # A = USV^t
    {s, u, v}
  end

  @doc """
    Calculate eigenvalues and eigenvectors by using QR decomposition.
    #### Argument
      - a: Matrix to calculate eigenvalues and eigenvectors by using the QR decomposition.
    #### Output
      [Eigenvalues list, Eigenvectors list]: Eigenvalues and eigenvectors.
      Eigenvalue is a non-trivial value other than zero, and complex numbers are not supported.
    #### Example
        iex> MatrixOperation.eigen([[1, 4, 5], [4, 2, 6], [5, 6, 3]])
        {
          [12.175971065046914, -3.6686830979532736, -2.507287967093643],
          [
            [0.4965997845461912, 0.5773502691896258, 0.6481167492476514],
            [0.3129856771935595, 0.5773502691896258, -0.7541264035547063],
            [-0.8095854617397507, 0.577350269189626, 0.10600965430705471]
          ]
        }
    """
  def eigen(a) do
    delta = 0.0001 # avoid division by zero
    evals = eigenvalue(a)
    evecs = evals
    |> Enum.map(
      & eigenvalue_shift(a, -&1+delta)
      |> inverse_matrix()
      |> power_iteration()
      |> extract_second()
    )
    {evals, evecs}
  end

  defp eigenvalue(a) do
    # Set the number of iterations according to the number of dimensions.
    # Refer to the LAPACK (ex. dlahqr).
    iter_max = 30 * Enum.max([10, length(a)])
    matrix_len = length(a)
    u = unit_matrix(matrix_len)
    a
    |> hessenberg(matrix_len, u, 1)
    |> eigenvalue_sub(matrix_len, u, 0, iter_max)
    |> exclude_zero_eigenvalue()
  end

  defp eigenvalue_sub(a, matrix_len, u, count, iter_max) when count != iter_max do
    q_n = qr_for_ev(a, u, matrix_len, u, 1)
    a_k = q_n
    |> transpose()
    |> product(a)
    |> product(q_n)
    eigenvalue_sub(a_k, matrix_len, u, count+1, iter_max)
  end

  defp eigenvalue_sub(a_k, _, _,  _, _) do
    a_k
    |> Enum.with_index()
    |> Enum.map(fn {x, i} -> Enum.at(x, i) end)
  end

  defp qr_for_ev(a, q, matrix_len, u, num) when matrix_len != num do
    h = a
    |> get_one_column(num)
    |> replace_zero(num-1)
    |> householder(num-1, u)

    a_n = product(h, a)
    q_n = product(q, h)

    qr_for_ev(a_n, q_n, matrix_len, u, num+1)
  end

  defp qr_for_ev(_, q_n, _, _, _) do
    q_n
  end

  defp hessenberg(a, matrix_len, u, num) when matrix_len != num + 1 do
    q = a
    |> get_one_column(num)
    |> replace_zero(num)
    |> householder(num, u)

    qt = transpose(q)
    hess = q
    |> product(a)
    |> product(qt)

    hessenberg(hess, matrix_len, u, num+1)
  end

  defp hessenberg(hess, _, _, _) do
    hess
  end

  defp replace_zero(list, thresh_num) do
    list
    |> Enum.with_index()
    |> Enum.map(fn {x, i} -> if(i < thresh_num, do: 0, else: x) end)
  end

  defp householder(col, index, u) do
    col_norm = col
    |> Enum.map(& &1*&1)
    |> Enum.sum()
    |> :math.sqrt()

    top = Enum.at(col, index)
    top_cn = if(top >= 0, do: top + col_norm, else: top - col_norm)
    v = List.replace_at(col, index, top_cn)

    cn_top = if(top >= 0, do: col_norm + top, else: col_norm - top)
    vtv = [v]
    |> transpose
    |> product([v])

    # avoid division by zero
    norm = if(
      col_norm * cn_top == 0,
      do: 0.0001,
      else: col_norm * cn_top
    )
    m = const_multiple(1/norm, vtv)
    subtract(u, m)
  end

  defp eigenvalue_shift(a, ev) do
    unit = a
    |> length
    |> unit_matrix()
    b = const_multiple(ev, unit)
    add(a, b)
  end

  defp extract_second({_first, second}) do
    second
  end

  @doc """
    Matrix diagonalization using the QR decomposition.
    #### Argument
      - a: Matrix to be diagonalized by using the QR decomposition.
    #### Output
      Diagonalized matrix
    #### Example
        iex> MatrixOperation.diagonalization([[1, 3], [4, 2]])
        [[5.000000000000018, 0], [0, -1.999999999999997]]
        iex> MatrixOperation.diagonalization([[2, 1, -1], [1, 1, 5], [-1, 2, 1]])
        [[4.101784906061095, 0, 0], [0, -2.6170329440542233, 0], [0, 0, 2.515248037993127]]
        iex> MatrixOperation.diagonalization([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
        nil
        iex> MatrixOperation.diagonalization([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
        nil
        iex> MatrixOperation.diagonalization([[16, -1, 1, 2, 3], [2, 12, 1, 5, 6], [1, 3, -24, 8, 9], [3, 4, 9, 1, 23], [5, 3, 1, 2, 1]])
        [
          [-26.608939298557207, 0, 0, 0, 0],
          [0, 20.42436493500135, 0, 0, 0],
          [0, 0, 14.665793374162678, 0, 0],
          [0, 0, 0, -3.5477665464080044, 0],
          [0, 0, 0, 0, 1.0665475358009446]
        ]
    """
  def diagonalization(a) do
    ev = eigenvalue(a)
    if(length(ev)==length(a), do: ev, else: nil)
    |> diagonalization_condition()
  end

  defp diagonalization_condition(a) when a == nil do
    nil
  end

  defp diagonalization_condition(a) do
    a
    |> Enum.with_index()
    |> Enum.map(& diagonalization_sub(&1, length(a), 0, []))
    |> Enum.map(& Enum.map(&1, fn x -> zero_approximation(x) end))
  end

  defp diagonalization_sub(_, dim, i, row) when i + 1 > dim do
    row
  end

  defp diagonalization_sub({ev, index}, dim, i, row) when i != index do
    diagonalization_sub({ev, index}, dim, i + 1, row ++ [0])
  end

  defp diagonalization_sub({ev, index}, dim, i, row) when i == index do
    diagonalization_sub({ev, index}, dim, i + 1, row ++ [ev])
  end

  @doc """
    Calculate singular Value by using QR decomposition.
    #### Argument
      - a: Matrix to calculate singular values.
    #### Output
      Singular values list. Singular value is a non-trivial value other than zero.
    #### Example
        iex> MatrixOperation.singular_value([[1, 2, 3, 1], [2, 4, 1, 5], [3, 3, 10, 8]])
        {14.9121726205599, 4.23646340778201, 1.6369134152873912}
    """
  def singular_value(a) do
    a
    |> transpose()
    |> product(a)
    |> eigenvalue()
    |> Enum.map(& :math.sqrt(&1))
    |> List.to_tuple()
  end

  @doc """
    Calculate the rank of a matrix by using QR decomposition
    #### Example
        iex> MatrixOperation.rank([[2, 3, 4], [1, 4, 2], [2, 1, 4]])
        2
        iex> MatrixOperation.rank([[2, 3, 4, 2], [1, 4, 2, 3], [2, 1, 4, 4]])
        3
        iex> input = [[2, 3, 4, 3], [1, 42, 2, 11], [2, 1, 4, 4], [3, 7, 2, 2], [35, 6, 4, 6], [7, 23, 5, 2]]
        iex> MatrixOperation.rank(input)
        4
    """
  def rank(matrix) do
    matrix
    |> singular_value()
    |> Tuple.to_list()
    |> length()
  end

  @doc """
    Frobenius norm
    #### Argument
      - a: Matrix to calculate Frobenius norm.
    #### Output
      Frobenius norm
    #### Example
        iex> MatrixOperation.frobenius_norm([[2, 3], [1, 4], [2, 1]])
        5.916079783099616
        iex> MatrixOperation.frobenius_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
        7.54983443527075
    """
  def frobenius_norm(a) do
    a
    |> Enum.map(& Enum.map(&1, fn x -> x * x end))
    |> Enum.map(& Enum.sum(&1))
    |> Enum.sum()
    |> :math.sqrt()
  end

  @doc """
    The one norm
    #### Argument
      - a: Matrix to calculate the one norm.
    #### Output
      one norm
    #### Example
        iex> MatrixOperation.one_norm([[2, 3], [1, 4], [2, 1]])
        5
        iex> MatrixOperation.one_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
        7
    """
  def one_norm(a) do
    a
    |> Enum.map(& Enum.map(&1, fn x -> if(x > 0, do: x, else: -x) end))
    |> Enum.map(& Enum.sum(&1))
    |> Enum.max()
  end

  @doc """
    The two norm
    #### Argument
      - a: Matrix to calculate the two norm.
    #### Output
      The two norm
    #### Example
        iex> MatrixOperation.two_norm([[2, 3], [1, 4], [2, 1]])
        5.674983803488139
        iex> MatrixOperation.two_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
        7.329546646114923
    """
  def two_norm(a) do
    a
    |> singular_value()
    |> Tuple.to_list()
    |> Enum.max()
  end

  @doc """
    The max norm
    #### Argument
      - a: Matrix to calculate the max norm.
    #### Output
      The max norm
    #### Example
        iex> MatrixOperation.max_norm([[2, 3], [1, 4], [2, 1]])
        8
        iex> MatrixOperation.max_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
        10
    """
  def max_norm(a) do
    a
    |> transpose()
    |> Enum.map(& Enum.map(&1, fn x -> if(x > 0, do: x, else: -x) end))
    |> Enum.map(& Enum.sum(&1))
    |> Enum.max()
  end

  @doc """
    A variance-covariance matrix is generated.
    #### Argument
      - data: x and y coordinate lists ([[x_1, y_1], [x_2, y_2], ...]) to calculate variance-covariance matrix.
    #### Output
      Variance-covariance matrix
    #### Example
        iex> MatrixOperation.variance_covariance_matrix([[40, 80], [80, 90], [90, 100]])
        [
          [466.66666666666663, 166.66666666666666],
          [166.66666666666666, 66.66666666666666]
        ]
    """
  def variance_covariance_matrix(data) do
    x = data
    |> transpose()
    |> Enum.map(& Enum.map(&1, fn x -> x - Enum.sum(&1)/length(&1) end))
    xt  = transpose(x)
    xtx = product(x, xt)
    const_multiple(1/length(xt), xtx)
  end

end

defmodule MatrixOperation do
  @moduledoc """
  Documentation for Matrix operation library.
  """

  @doc """
  Numbers of row and column of a matrix are got.
  ## Examples
    iex> MatrixOperation.row_column_matrix([[3, 2, 3], [2, 1, 2]])
    [2, 3]
  """
  def row_column_matrix(matrix) when is_list(hd(matrix)) do
    col_num = Enum.map(matrix, &row_column_matrix_sub(&1, 0))
    max_num = Enum.max(col_num)
    if(max_num == Enum.min(col_num), do: [length(matrix), max_num], else: nil)
  end

  def row_column_matrix(_matrix) do
    nil
  end

  defp row_column_matrix_sub(row, i) when i != length(row) do
    if(is_number(Enum.at(row, i)), do: row_column_matrix_sub(row, i + 1), else: nil)
  end

  defp row_column_matrix_sub(row, i) when i == length(row) do
    i
  end

  @doc """
  A n-th unit matrix is got.
  ## Examples
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
    #### Examples
      iex> MatrixOperation.even_matrix(2, 3, 0)
      [[0, 0, 0], [0, 0, 0]]
      iex> MatrixOperation.even_matrix(3, 2, 1)
      [[1, 1], [1, 1], [1, 1]]
    """
  def even_matrix(m, n, s) when m > 0 and n > 0 and is_number(s) do
    Enum.to_list(1..m) |>
    Enum.map(fn _ -> Enum.map(Enum.to_list(1..n), & &1 * 0 + s) end)
  end

  def even_matrix(_m, _n, _s) do
    nil
  end

  @doc """
  A element of a matrix is got.
  ## Examples
    iex> MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], [1, 1])
    1
  """
  def get_one_element(matrix, [row_idx, col_idx]) do
    matrix
    |> Enum.at(row_idx - 1)
    |> Enum.at(col_idx - 1)
  end

  @doc """
  A row of a matrix is got.
  ## Examples
    iex> MatrixOperation.get_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
    [1, 2, 3]
  """
  def get_one_row(matrix, row_idx) do
    matrix
    |> Enum.at(row_idx - 1)
  end

  @doc """
  A column of a matrix is got.
  ## Examples
      iex> MatrixOperation.get_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
      [1, 4, 7]
  """
  def get_one_column(matrix, col_idx) do
    matrix
    |> transpose
    |> Enum.at(col_idx - 1)
  end

  @doc """
  A row of a matrix is deleted.
  ## Examples
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
  ## Examples
      iex> MatrixOperation.delete_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
      [[1, 3], [4, 6], [7, 9]]
  """
  def delete_one_column(matrix, del_idx) do
    matrix
    |> transpose
    |> Enum.with_index()
    |> Enum.reject(fn {_x, idx} -> idx == del_idx - 1 end)
    |> Enum.map(fn {x, _idx} -> x end)
    |> transpose
  end

  @doc """
  A row of a matrix is exchanged.
  ## Examples
      iex> MatrixOperation.exchange_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, [1, 1, 1])
      [[1, 2, 3], [4, 5, 6], [1, 1, 1]]
  """
  def exchange_one_row(matrix, exchange_idx, exchange_list) do
    matrix
    |> Enum.with_index()
    |> Enum.map(fn {x, idx} -> if(idx == exchange_idx - 1, do: exchange_list, else: x) end)
  end

  @doc """
  A row of a matrix is exchanged.
  ## Examples
      iex> MatrixOperation.exchange_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, [1, 1, 1])
      [[1, 1, 3], [4, 1, 6], [7, 1, 9]]
  """
  def exchange_one_column(matrix, exchange_idx, exchange_list) do
    matrix
    |> transpose
    |> Enum.with_index()
    |> Enum.map(fn {x, idx} -> if(idx == exchange_idx - 1, do: exchange_list, else: x) end)
    |> transpose
  end

  @doc """
  Transpose of a matrix
  ## Examples
      iex> MatrixOperation.transpose([[1.0, 2.0], [3.0, 4.0]])
      [[1.0, 3.0], [2.0, 4.0]]
  """
  def transpose(matrix) do
    Enum.zip(matrix)
    |> Enum.map(&Tuple.to_list(&1))
  end

  @doc """
  Trace of a matrix
  ## Examples
      iex> MatrixOperation.trace([[1.0, 2.0], [3.0, 4.0]])
      5.0
  """
  def trace(matrix) do
    [row_num, col_num] = row_column_matrix(matrix)
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
  ## Examples
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
  ## Examples
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
  Linear equations are solved by Cramer's rule.
  ## Examples
      iex> MatrixOperation.linear_equations_cramer([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1], [0], [0]])
      [1.0, 0.0, 0.0]
      iex> MatrixOperation.linear_equations_cramer([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1], [0], [0]])
      [1.0, 0.0, 0.0]
  """
  def linear_equations_cramer(matrix, vertical_vec) do
    # check the setupufficient condition
    if determinant(matrix) == 0 do
      nil
    else
      [t] = transpose(vertical_vec)
      linear_equations_cramer_sub(matrix, t, 0, [])
    end
  end

  defp linear_equations_cramer_sub(matrix, t, i, output) when i < length(matrix) do
    vertical_vec = transpose([t])
    linear_equations_cramer_sub(matrix, t, i + 1, output ++ [cramer(matrix, vertical_vec, i + 1)])
  end

  defp linear_equations_cramer_sub(matrix, _t, i, output) when i == length(matrix) do
    output
  end

  @doc """
    Leading principal minors are generetaed
    #### Examples
      iex> MatrixOperation.leading_principal_minor([[1, 3, 2], [2, 5, 1], [3, 4, 5]], 2)
      [[1, 3], [2, 5]]
    """
  def leading_principal_minor(matrix, k) do
    Enum.slice(matrix, 0, k)
    |> Enum.map(& Enum.slice(&1, 0, k))
  end

  @doc """
    LU decomposition
    #### Examples
      iex> MatrixOperation.lu_decomposition([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
      [
        [[1, 0, 0, 0], [2.0, 1, 0, 0], [3.0, 4.0, 1, 0], [-1.0, -3.0, 0.0, 1]],
        [[1, 1, 0, 3], [0, -1.0, -1.0, -5.0], [0, 0, 3.0, 13.0], [0, 0, 0, -13.0]]
      ]
    """
  def lu_decomposition(matrix) do
    [row_num, col_num] = row_column_matrix(matrix)
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
    u_matrix = even_matrix(matrix_len, matrix_len, 0)
               |> exchange_one_row(1, hd(matrix))
    inverce_u11 = 1.0 / hd(hd(u_matrix))
    factor = transpose(matrix)
              |> get_one_row(1)
              |> Enum.slice(1, matrix_len)
    l_row = [1] ++ hd(const_multiple(inverce_u11, [factor]))
    l_matrix = even_matrix(matrix_len, matrix_len, 0)
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
    [transpose(l_matrix), u_matrix]
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
  ## Examples
      iex> MatrixOperation.linear_equations_direct([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1], [0], [0]])
      [1.0, 0.0, 0.0]
      iex> MatrixOperation.linear_equations_direct([[4, 1, 1], [1, 3, 1], [2, 1, 5]], [[9], [10], [19]])
      [1.0, 2.0, 3.0]
  """
  def linear_equations_direct(matrix, vertical_vec) do
    # check the setupufficient condition
    if determinant(matrix) == 0 do
      nil
    else
      [t] = transpose(vertical_vec)
      linear_equations_direct_sub(matrix, t)
    end
  end

  defp linear_equations_direct_sub(matrix, t) do
    [l_matrix, u_matrix] = lu_decomposition(matrix)
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
  ## Examples
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
  ## Examples
      iex> MatrixOperation.const_addition(1, [1.0, 2.0, 3.0])
      [2.0, 3.0, 4.0]
  """
  def const_addition(const, x) when is_number(x) do
    const + x
  end

  def const_addition(const, x) when is_list(x) do
    Enum.map(x, &const_addition(const, &1))
  end

  @doc """
  Inverse Matrix
  ## Examples
      iex> MatrixOperation.inverse_matrix([[1, 1, -1], [-2, -1, 1], [-1, -2, 1]])
      [[-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [-3.0, -1.0, -1.0]]
  """
  def inverse_matrix(matrix) when is_list(hd(matrix)) do
    det = determinant(matrix)

    create_index_matrix(matrix)
    |> Enum.map(&map_index_row(matrix, det, &1))
    |> transpose
  end

  def inverse_matrix(_) do
    nil
  end

  defp create_index_matrix(matrix) do
    idx_list = Enum.to_list(1..length(matrix))
    Enum.map(idx_list, fn x -> Enum.map(idx_list, &[x, &1]) end)
  end

  defp map_index_row(_matrix, 0, _row) do
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
  ## Examples
      iex> MatrixOperation.product([[3, 2, 3], [2, 1, 2]], [[2, 3], [2, 1], [3, 5]])
      [[19, 26], [12, 17]]
  """
  def product(a, b) do
    check_product(a, b)
  end

  defp check_product(a, b) do
    col_num_a = row_column_matrix(a) |> Enum.at(1)
    row_num_b = row_column_matrix(b) |> Enum.at(0)
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
  ## Examples
      iex> MatrixOperation.add([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
      [[5, 5, 4], [5, 3, 4]]
  """
  def add(a, b) do
    check_add(a, b)
  end

  defp check_add(a, b) do
    row_col_a = row_column_matrix(a)
    row_col_b = row_column_matrix(b)
    if(row_col_a == row_col_b, do: add_sub(a, b), else: nil)
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
  ## Examples
      iex> MatrixOperation.subtract([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
      [[1, -1, 2], [-1, -1, 0]]
  """
  def subtract(a, b) do
    check_subtract(a, b)
  end

  defp check_subtract(a, b) do
    row_col_a = row_column_matrix(a)
    row_col_b = row_column_matrix(b)
    if(row_col_a == row_col_b, do: subtract_sub(a, b), else: nil)
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
  ## Examples
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
  ## Examples
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
  ## Examples
      iex> MatrixOperation.hadamard_power([[3, 2, 3], [2, 1, 2]], 2)
      [[9.0, 4.0, 9.0], [4.0, 1.0, 4.0]]
  """
  def hadamard_power(matrix, n) do
    Enum.map(matrix, &Enum.map(&1, fn x -> :math.pow(x, n) end))
  end

  @doc """
  Tensor product
  ## Examples
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
  eigenvalue by the direct method [R^2×R^2/R^3×R^3 matrix]
  ## Examples
    iex> MatrixOperation.eigenvalue_direct([[3, 1], [2, 2]])
    [4.0, 1.0]
    iex> MatrixOperation.eigenvalue_direct([[6, -3], [4, -1]])
    [3.0, 2.0]
    iex> MatrixOperation.eigenvalue_direct([[1, 1, 1], [1, 2, 1], [1, 2, 3]])
    [4.561552806429505, 0.43844714673139706, 1.0000000468390973]
    iex> MatrixOperation.eigenvalue_direct([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
    [3.0000000027003626, 0, 0.9999999918989121]
  """
  # 2×2 algebra method
  def eigenvalue_direct([[a11, a12], [a21, a22]]) do
    quadratic_formula(1, -a11 - a22, a11 * a22 - a12 * a21)
  end

  # 3×3 algebratic method
  def eigenvalue_direct([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]) do
    a = -1
    b = a11 + a22 + a33
    c = a21 * a12 + a13 * a31 + a32 * a23 - a11 * a22 - a11 * a33 - a22 * a33
    d =
      a11 * a22 * a33 + a12 * a23 * a31 + a13 * a32 * a21 - a11 * a32 * a23 - a22 * a31 * a13 -
        a33 * a21 * a12

    dis = -4 * a * c * c * c - 27 * a * a * d * d + b * b * c * c + 18 * a * b * c * d - 4 * b * b * b * d
    if(dis > 0, do: cubic_formula(a, b, c, d), else: nil)
  end

  def eigenvalue_direct(_a) do
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

  @doc """
    Matrix diagonalization by algebra method [R^2×R^2/R^3×R^3 matrix]
    #### Examples
      iex> MatrixOperation.diagonalization_algebra([[1, 3], [4, 2]])
      [[5.0, 0], [0, -2.0]]
      iex> MatrixOperation.diagonalization_algebra([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
      [[3.0000000027003626, 0, 0], [0, 0, 0], [0, 0, 0.9999999918989121]]
    """
  def diagonalization_algebra(matrix) do
    eigenvalue_direct(matrix)
    |> diagonalization_algebra_condition
  end

  defp diagonalization_algebra_condition(matrix) when matrix == nil do
    nil
  end

  defp diagonalization_algebra_condition(matrix) do
    matrix
    |> Enum.with_index
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
    #### Examples
      iex> MatrixOperation.jordan_normal_form([[1, 3], [4, 2]])
      [[5.0, 0], [0, -2.0]]
      iex> MatrixOperation.jordan_normal_form([[7, 2], [-2, 3]])
      [[5.0, 1], [0, 5.0]]
      iex> MatrixOperation.jordan_normal_form([[2, 1, -1], [1, 1, 0], [-1, 0, 1]])
      [[3.0000000027003626, 0, 0], [0, 0, 0], [0, 0, 0.9999999918989121]]
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
  ## Examples
    iex> MatrixOperation.power_iteration([[3, 1], [2, 2]], 100)
    [
      4.0,
      [2.8284271247461903, 2.8284271247461903]
    ]
    iex> MatrixOperation.power_iteration([[1, 1, 2], [0, 2, -1], [0, 0, 3]], 100)
    [
      3.0,
      [1.0, -2.0, 2.0]
    ]
  """
  def power_iteration(matrix, iter_num) do
    init_vec = random_column(length(matrix))
    xk_pre = power_iteration_sub(matrix, init_vec, iter_num)
    # eigen vector
    [xk_vec] = product(matrix, xk_pre) |> transpose
    [xk_pre_vec] = transpose(xk_pre)
    # eigen value
    eigen_value = inner_product(xk_vec, xk_vec) / inner_product(xk_vec, xk_pre_vec)
    [eigen_value, xk_vec]
  end

  defp random_column(num) when num > 1 do
    tmp = Enum.reduce(1..num, [], fn _, acc -> [Enum.random(0..50000) / 10000 | acc] end)
    transpose([tmp])
  end

  defp random_column(_num) do
    nil
  end

  defp power_iteration_sub(matrix, v, iter_num) do
    # Normarization is for overflow suppression
    Enum.reduce(1..iter_num, v, fn _, acc ->
      vp = product(matrix, acc)
      [vpt] = transpose(vp)
      const_multiple(1 / :math.sqrt(inner_product(vpt, vpt)), vp)
    end)
  end

  @doc """
    Calculate eigenvalue and eigenvector by using Jacobi method
    #### Examples
      iex> MatrixOperation.jacobi([[10, 3, 2], [3, 5, 1], [2, 1, 0]], 100)
      [
        [11.827601656659317, 3.5956497715829547, -0.42325142824210527],
        [
          [0.8892872578006493, -0.42761854121982545, -0.16220529066103917],
          [0.4179466723082325, 0.9038581385545962, -0.09143874712126684],
          [0.1857114757355589, 0.013522151221627882, 0.982511271796136]
        ]
      ]
    """
  def jacobi(matrix, iter_num) do
    [pap, p] = jacobi_iteration(matrix, iter_num, 0, unit_matrix(length(matrix)))
    p_rnd = Enum.map(p, & Enum.map(&1, fn x -> zero_approximation(x) end))

    eigenvalue_list = pap
    |> Enum.with_index
    |> Enum.map(& jacobi_sub4(&1))
    |> Enum.map(& zero_approximation(&1))
    [eigenvalue_list, p_rnd]
  end

  defp jacobi_iteration(matrix, iter_num, l, p_pre) when l != iter_num do
    [row_num, col_num] = row_column_matrix(matrix)
    odts = off_diagonal_terms(matrix, row_num, col_num, 0, 0, [])
    |> Enum.map(& abs(&1))

    max_odt = Enum.max(odts)
    [max_i, max_j] = Enum.with_index(odts)
    |> jocobi_sub(max_odt, 0)
    |> jocobi_sub2(col_num, 0)

    a_ij = get_one_element(matrix, [max_i + 1, max_j + 1])
    a_ii = get_one_element(matrix, [max_i + 1, max_i + 1])
    a_jj = get_one_element(matrix, [max_j + 1, max_j + 1])
    phi = phi_if(a_ii - a_jj, a_ij)

    p = jacobi_sub3(phi, col_num, max_i, max_j, 0, 0, [], [])
    p_pi = product(p_pre, p)
    p
    |> transpose
    |> product(matrix)
    |> product(p)
    |> jacobi_iteration(iter_num, l + 1, p_pi)
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
    off_diagonal_terms(m, row_num, col_num, i, j + 1, output ++ [get_one_element(m, [i + 1, j + 1])])
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
    Singular Value Decomposition (SVD) by using Jacobi method
    #### Examples
      iex> MatrixOperation.svd([[1, 0, 0], [0, 1, 1]], 100)
      [
        [1.0, 1.4142135623730951],
        [[1.0, 0], [0, 1.0]],
        [
          [1.0, 0, 0],
          [0, 0.7071067458364744, -0.707106816536619],
          [0, 0.707106816536619, 0.7071067458364744]
        ]
      ]
    """
  def svd(a, iter_num) do
    a_t = transpose(a)
    svd_sub(a, a_t, iter_num)
  end

  def svd_sub(a, a_t, iter_num) when length(a) <= length(a_t) do
    # U matrix
    aat = product(a, a_t)
    [sv_sq, u] = jacobi(aat, iter_num)
    # V matirx
    ata = product(a_t, a)
    [_, v] = jacobi(ata, iter_num)
    # Singular value
    s = Enum.map(sv_sq, & :math.sqrt(&1))
    # A = USV^t
    [s, u, v]
  end

  def svd_sub(a, a_t, iter_num) do
    # U matrix
    aat = product(a, a_t)
    [_, u] = jacobi(aat, iter_num)
    # V matirx
    ata = product(a_t, a)
    [sv_sq, v] = jacobi(ata, iter_num)
    # Singular value
    s = Enum.map(sv_sq, & :math.sqrt(&1))
    # A = USV^t
    [s, u, v]
  end

  @doc """
    Calculate eigenvalue by using QR decomposition
    #### Examples
      iex> MatrixOperation.eigenvalue([[1, 4, 5], [4, 2, 6], [5, 6, 3]], 100)
      [12.17597106504691, -3.6686830979532696, -2.5072879670936357]
      iex> MatrixOperation.eigenvalue([[6, 1, 1, 1], [1, 7, 1, 1], [1, 1, 8, 1], [1, 1, 1, 9]], 100)
      [10.803886359051251, 7.507748705362773, 6.39227529027387, 5.296089645312106]
    """
  def eigenvalue(a, iter_num) do
    eigenvalue_sub(a, 0, iter_num)
  end

  defp eigenvalue_sub(a, count, iter_num) when count != iter_num do
    matrix_len = length(a)
    u = unit_matrix(matrix_len)
    q_n = qr_for_ev(a, u, matrix_len, u, 1)
    a_k = q_n
    |> transpose
    |> product(a)
    |> product(q_n)
    eigenvalue_sub(a_k, count+1, iter_num)
  end

  defp eigenvalue_sub(a_k, _, _) do
    a_k
    |> Enum.with_index
    |> Enum.map(fn {x, i} -> Enum.at(x, i) end)
  end

  defp qr_for_ev(a, q, matrix_len, u, num) when matrix_len != num do
    h = get_one_column(a, num)
    |> replace_zero(num-1)
    |> householder_for_qr(num-1, u)
    
    a_n = product(h, a)
    q_n = product(q, h)
    
    qr_for_ev(a_n, q_n, matrix_len, u, num+1)
  end

  defp qr_for_ev(_, q_n, _, _, _) do
    q_n
  end

  defp replace_zero(list, thresh_num) do
    list
    |> Enum.with_index
    |> Enum.map(fn {x, i} -> if(i < thresh_num, do: 0, else: x) end)
  end

  defp householder_for_qr(col, index, u) do
    col_norm = col
    |> Enum.map(& &1*&1)
    |> Enum.sum
    |> :math.sqrt

    top = Enum.at(col, index)
    top_cn = if(top >= 0, do: top + col_norm, else: top - col_norm)
    v = List.replace_at(col, index, top_cn)

    cn_top = if(top >= 0, do: col_norm + top, else: col_norm - top)
    vtv = [v]
    |> transpose
    |> product([v])
    m = const_multiple(1/(col_norm * cn_top), vtv)

    subtract(u, m)  
  end

  @doc """
    Matrix diagonalization
    #### Examples
      iex> MatrixOperation.diagonalization([[1, 3], [4, 2]], 100)
      [[5.000000000000018, 0], [0, -1.999999999999997]]
      iex> MatrixOperation.diagonalization([[2, 1, -1], [1, 1, 0], [-1, 0, 1]], 100)
      [[3.000000000000001, 0, 0], [0, 1.0, 0], [0, 0, 0]]
    """
  def diagonalization(a, iter_num) do
    eigenvalue(a, iter_num)
    |> diagonalization_condition
    |> Enum.map(& Enum.map(&1, fn x -> zero_approximation(x) end))
  end

  defp diagonalization_condition(a) when a == nil do
    nil
  end

  defp diagonalization_condition(a) do
    a
    |> Enum.with_index
    |> Enum.map(& diagonalization_sub(&1, length(a), 0, []))
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
    Calculate singular Value by using QR decomposition
    #### Examples
      iex> MatrixOperation.singular_value([[1, 2, 3, 1], [2, 4, 1, 5], [3, 3, 10, 8]], 100)
      [14.912172620559879, 4.236463407782015, 1.6369134152873956, 0.0]
    """
  def singular_value(a, iter_num) do
    a
    |> transpose
    |> product(a)
    |> eigenvalue(iter_num)
    |> Enum.map(& zero_approximation(&1))
    |> Enum.map(& :math.sqrt(&1))
  end

  @doc """
  Calculate the rank of a matrix by using QR decomposition
  ## Examples
      iex> MatrixOperation.rank([[2, 3, 4], [1, 4, 2], [2, 1, 4]], 100)
      2
      iex> MatrixOperation.rank([[2, 3, 4, 2], [1, 4, 2, 3], [2, 1, 4, 4]], 100)
      3
  """
  def rank(matrix, iter_num) do
    singular_value(matrix, iter_num)
    |> count_finite_values
  end

  defp count_finite_values(x) when is_list(x) do
    x
    |> Enum.map(&count_finite_values(&1))
    |> Enum.sum
  end

  defp count_finite_values(x) when is_number(x) and x == 0 do
    0
  end

  defp count_finite_values(x) when is_number(x) do
    1
  end

  @doc """
    Frobenius norm
    #### Examples
      iex> MatrixOperation.frobenius_norm([[2, 3], [1, 4], [2, 1]])
      5.916079783099616
      iex> MatrixOperation.frobenius_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
      7.54983443527075
    """
  def frobenius_norm(a) do
    a
    |> Enum.map(& Enum.map(&1, fn x -> x * x end))
    |> Enum.map(& Enum.sum(&1))
    |> Enum.sum
    |> :math.pow(0.5)
  end

  @doc """
    one norm
    #### Examples
      iex> MatrixOperation.one_norm([[2, 3], [1, 4], [2, 1]])
      5
      iex> MatrixOperation.one_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
      7
    """
  def one_norm(a) do
    a
    |> Enum.map(& Enum.map(&1, fn x -> if(x > 0, do: x, else: -x) end))
    |> Enum.map(& Enum.sum(&1))
    |> Enum.max
  end

  @doc """
    two norm
    #### Examples
      iex> MatrixOperation.two_norm([[2, 3], [1, 4], [2, 1]])
      5.674983803488139
      iex> MatrixOperation.two_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
      7.329546646114924
    """
  def two_norm(a) do
    a
    |> singular_value(100)
    |> Enum.max
  end

  @doc """
    max norm
    #### Examples
      iex> MatrixOperation.max_norm([[2, 3], [1, 4], [2, 1]])
      8
      iex> MatrixOperation.max_norm([[1, 3, 3], [2, 4, 1], [2, 3, 2]])
      10
    """
  def max_norm(a) do
    transpose(a)
    |> Enum.map(& Enum.map(&1, fn x -> if(x > 0, do: x, else: -x) end))
    |> Enum.map(& Enum.sum(&1))
    |> Enum.max
  end

  @doc """
    A variance-covariance matrix is generated
    #### Examples
      iex> MatrixOperation.variance_covariance_matrix([[40, 80], [80, 90], [90, 100]])
      [
        [466.66666666666663, 166.66666666666666],
        [166.66666666666666, 66.66666666666666]
      ]
    """
  def variance_covariance_matrix(data) do
    x = data
    |> transpose
    |> Enum.map(& Enum.map(&1, fn x -> x - Enum.sum(&1)/length(&1) end))
    xt  = transpose(x)
    xtx = product(x, xt)
    const_multiple(1/length(xt), xtx)
  end

end

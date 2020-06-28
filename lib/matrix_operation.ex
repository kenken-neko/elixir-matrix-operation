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
  def row_column_matrix(a) when is_list(hd(a)) do
    columns_number = Enum.map(a, &row_column_matrix_sub(&1, 0))
    max_number = Enum.max(columns_number)
    if(max_number == Enum.min(columns_number), do: [length(a), max_number], else: nil)
  end

  def row_column_matrix(_) do
    nil
  end

  defp row_column_matrix_sub(row_a, i) when i != length(row_a) do
    if(is_number(Enum.at(row_a, i)), do: row_column_matrix_sub(row_a, i + 1), else: nil)
  end

  defp row_column_matrix_sub(row_a, i) when i == length(row_a) do
    i
  end

  @doc """
  A n-th unit matrix is got.
  ## Examples
    iex> MatrixOperation.unit_matrix(3)
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  """
  def unit_matrix(n) when n > 0 and is_integer(n) do
    index_list = Enum.to_list(1..n)
    Enum.map(index_list, fn x -> Enum.map(index_list, &unit_matrix_sub(x, &1)) end)
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

  def even_matrix(_, _, _) do
    nil
  end

  @doc """
  A element of a matrix is got.
  ## Examples
    iex> MatrixOperation.get_one_element([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], [1, 1])
    1
  """
  def get_one_element(matrix, [row_index, column_index]) do
    matrix
    |> Enum.at(row_index - 1)
    |> Enum.at(column_index - 1)
  end

  @doc """
  A row of a matrix is got.
  ## Examples
    iex> MatrixOperation.get_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
    [1, 2, 3]
  """
  def get_one_row(matrix, row_index) do
    matrix
    |> Enum.at(row_index - 1)
  end

  @doc """
  A column of a matrix is got.
  ## Examples
      iex> MatrixOperation.get_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
      [1, 4, 7]
  """
  def get_one_column(matrix, column_index) do
    matrix
    |> transpose
    |> Enum.at(column_index - 1)
  end

  @doc """
  A row of a matrix is deleted.
  ## Examples
      iex> MatrixOperation.delete_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
      [[1, 2, 3], [4, 5, 6]]
  """
  def delete_one_row(matrix, delete_index) do
    matrix
    |> Enum.with_index()
    |> Enum.reject(fn {_, i} -> i == delete_index - 1 end)
    |> Enum.map(fn {x, _} -> x end)
  end

  @doc """
  A column of a matrix is deleted.
  ## Examples
      iex> MatrixOperation.delete_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)
      [[1, 3], [4, 6], [7, 9]]
  """
  def delete_one_column(matrix, delete_index) do
    matrix
    |> transpose
    |> Enum.with_index()
    |> Enum.reject(fn {_, i} -> i == delete_index - 1 end)
    |> Enum.map(fn {x, _} -> x end)
    |> transpose
  end

  @doc """
  A row of a matrix is exchanged.
  ## Examples
      iex> MatrixOperation.exchange_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, [1, 1, 1])
      [[1, 2, 3], [4, 5, 6], [1, 1, 1]]
  """
  def exchange_one_row(matrix, exchange_index, exchange_list) do
    matrix
    |> Enum.with_index()
    |> Enum.map(fn {x, i} -> if(i == exchange_index - 1, do: exchange_list, else: x) end)
  end

  @doc """
  A row of a matrix is exchanged.
  ## Examples
      iex> MatrixOperation.exchange_one_column([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, [1, 1, 1])
      [[1, 1, 3], [4, 1, 6], [7, 1, 9]]
  """
  def exchange_one_column(matrix, exchange_index, exchange_list) do
    matrix
    |> transpose
    |> Enum.with_index()
    |> Enum.map(fn {x, i} -> if(i == exchange_index - 1, do: exchange_list, else: x) end)
    |> transpose
  end

  @doc """
  Transpose of a matrix
  ## Examples
      iex> MatrixOperation.transpose([[1.0, 2.0], [3.0, 4.0]])
      [[1.0, 3.0], [2.0, 4.0]]
  """
  def transpose(a) do
    Enum.zip(a)
    |> Enum.map(&Tuple.to_list(&1))
  end

  @doc """
  Trace of a matrix
  ## Examples
      iex> MatrixOperation.trace([[1.0, 2.0], [3.0, 4.0]])
      5.0
  """
  def trace(a) do
    [row, column] = row_column_matrix(a)
    a_index = add_index(a)

    Enum.map(a_index, &trace_sub(&1, row, column))
    |> Enum.sum()
  end

  defp trace_sub(_, row, column) when row != column do
    nil
  end

  defp trace_sub([index, row_list], _, _) do
    Enum.at(row_list, index - 1)
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
  def determinant(a) do
    determinant_sub(1, a)
  end

  # minor_matrix
  defp minor_matrix(a_with_index, row) do
    (a_with_index -- [row])
    |> Enum.map(&Enum.at(&1, 1))
    |> Enum.map(&Enum.drop(&1, 1))
  end

  # 1×1 matrix
  defp determinant_sub(_, a) when length(a) == 1 do
    Enum.at(a, 0)
    |> Enum.at(0)
  end

  # 2×2 matrix
  defp determinant_sub(co, [[a11, a12], [a21, a22]]) do
    co * (a11 * a22 - a12 * a21)
  end

  # 3×3 or over matrix
  defp determinant_sub(co, a) do
    a_with_index = add_index(a)

    Enum.map(
      a_with_index,
      &determinant_sub(
        (-1 + 2 * rem(hd(&1), 2)) * co * hd(Enum.at(&1, 1)),
        minor_matrix(a_with_index, &1)
      )
    )
    |> Enum.sum()
  end

  # add index
  defp add_index(a) do
    Stream.iterate(1, &(&1 + 1))
    |> Enum.zip(a)
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
  def cramer(a, vertical_vec, select_index) do
    [t] = transpose(vertical_vec)
    det_a = determinant(a)
    cramer_sub(a, t, select_index - 1, det_a)
  end

  defp cramer_sub(_, _, _, nil), do: nil
  defp cramer_sub(_, _, _, 0), do: nil

  defp cramer_sub(a, t, select_index, det_a) do
    rep_det_a = transpose(a) |> replace_element_in_list(select_index, t, 0, []) |> determinant
    rep_det_a / det_a
  end

  defp replace_element_in_list(list, i, replace_element, i, output) when i < length(list) do
    replace_element_in_list(list, i, replace_element, i + 1, output ++ [replace_element])
  end

  defp replace_element_in_list(list, select_index, replace_element, i, output)
       when i < length(list) do
    replace_element_in_list(
      list,
      select_index,
      replace_element,
      i + 1,
      output ++ [Enum.at(list, i)]
    )
  end

  defp replace_element_in_list(list, _select_index, _replace_element, i, output)
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
  def linear_equations_cramer(a, vertical_vec) do
    # check the setupufficient condition
    if determinant(a) == 0 do
      nil
    else
      [t] = transpose(vertical_vec)
      linear_equations_cramer_sub(a, t, 0, [])
    end
  end

  defp linear_equations_cramer_sub(a, t, i, output) when i < length(a) do
    vertical_vec = transpose([t])
    linear_equations_cramer_sub(a, t, i + 1, output ++ [cramer(a, vertical_vec, i + 1)])
  end

  defp linear_equations_cramer_sub(a, _t, i, output) when i == length(a) do
    output
  end

  @doc """
    Leading principal minors are generetaed
    #### Examples
      iex> MatrixOperation.leading_principal_minor([[1, 3, 2], [2, 5, 1], [3, 4, 5]], 2)
      [[1, 3], [2, 5]]
    """
  def leading_principal_minor(a, k) do
    Enum.slice(a, 0, k)
    |> Enum.map(& Enum.slice(&1, 0, k))
  end

  @doc """
    LU decomposition
    #### Examples
      iex> MatrixOperation.lu_decomposition([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
      [
        L: [[1, 0, 0, 0], [2.0, 1, 0, 0], [3.0, 4.0, 1, 0], [-1.0, -3.0, 0.0, 1]],
        U: [[1, 1, 0, 3], [0, -1.0, -1.0, -5.0], [0, 0, 3.0, 13.0], [0, 0, 0, -13.0]]
      ]
    """
  def lu_decomposition(a) do
    row_column = row_column_matrix(a)
    # check the setupufficient condition
    check_number = lu_decomposition_check(a, row_column)
    if(check_number == 0, do: nil, else: lu_decomposition_sub(a, 0, length(a), [], []))
  end

  defp lu_decomposition_check(_, [row_num, column_num]) when row_num != column_num do
    nil
  end

  defp lu_decomposition_check(a, [row_num, _]) do
    Enum.to_list(1..row_num)
    |> Enum.map(& leading_principal_minor(a, &1) |> determinant)
    |> Enum.reduce(fn x, acc -> x * acc end)
  end

  defp lu_decomposition_sub(a, k, len_a, _, _) when k == 0 do
    u_matrix = even_matrix(len_a, len_a, 0)
               |> exchange_one_row(1, hd(a))
    inverce_u11 = 1.0 / hd(hd(u_matrix))
    a_factor = transpose(a)
               |> get_one_row(1)
               |> Enum.slice(1, len_a)
    l_row = [1] ++ hd(const_multiple(inverce_u11, [a_factor]))
    l_matrix = even_matrix(len_a, len_a, 0)
               |> exchange_one_row(1, l_row)
    lu_decomposition_sub(a, k + 1, len_a, l_matrix, u_matrix)
  end

  defp lu_decomposition_sub(a, k, len_a, l_matrix, u_matrix) when k != len_a do
    a_t = transpose(a)
    u_solve = u_cal(a, k, len_a, l_matrix, u_matrix)
    u_matrix_2 = exchange_one_row(u_matrix, k + 1, u_solve)
    l_solve = l_cal(a_t, k, len_a, l_matrix, u_matrix_2)
    l_matrix_2 = exchange_one_row(l_matrix, k + 1, l_solve)
    lu_decomposition_sub(a, k + 1, len_a, l_matrix_2, u_matrix_2)
  end

  defp lu_decomposition_sub(_, _, _, l_matrix, u_matrix) do
    ["L": transpose(l_matrix), "U": u_matrix]
  end

  defp l_cal(a_t, k, len_a, l_matrix, u_matrix) do
    a_factor = Enum.at(a_t, k) |> Enum.slice(k + 1, len_a)
    u_extract = transpose(u_matrix) |> Enum.at(k)
    l_row = transpose(l_matrix)
    |> Enum.slice(k + 1, len_a)
    |> Enum.map(& inner_product(&1, u_extract))
    |> Enum.zip(a_factor)
    |> Enum.map(fn {x, y} -> y - x end)

    inverce_uii = 1.0 / Enum.at(Enum.at(u_matrix, k), k)
    [l_row_2] = const_multiple(inverce_uii, [l_row])
    [1] ++ l_row_2
    |> add_zero_element(0, k)
  end

  defp u_cal(a, k, len_a, l_matrix, u_matrix) do
    a_factor = Enum.at(a, k) |> Enum.slice(k, len_a)
    l_extract = transpose(l_matrix) |> Enum.at(k)
    transpose(u_matrix)
    |> Enum.slice(k, len_a)
    |> Enum.map(& inner_product(&1, l_extract))
    |> Enum.zip(a_factor)
    |> Enum.map(fn {x, y} -> y - x end)
    |> add_zero_element(0, k)
  end

  defp add_zero_element(list, init, fin) when init != fin do
    add_zero_element([0] ++ list, init + 1, fin)
  end

  defp add_zero_element(list, _, _) do
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
  def linear_equations_direct(a, vertical_vec) do
    # check the setupufficient condition
    if determinant(a) == 0 do
      nil
    else
      [t] = transpose(vertical_vec)
      lu_decomposition_sub(a, t)
    end
  end

  defp lu_decomposition_sub(a, t) do
    ["L": l_matrix, "U": u_matrix] = lu_decomposition(a)
    dim = length(l_matrix)
    y = forward_substitution(l_matrix, t, [], 0, dim)
    backward_substitution(u_matrix, y, [], dim, dim)
  end

  defp forward_substitution(l_matrix, t, _, k, dim) when k == 0 do
    forward_substitution(l_matrix, t, [hd(t)], k + 1, dim)
  end

  defp forward_substitution(l_matrix, t, y, k, dim) when k != dim do
    l_extract = Enum.at(l_matrix, k) |> Enum.slice(0, k)
    y_extract = y |> Enum.slice(0, k)
    ly = inner_product(l_extract, y_extract)
    t_ly = Enum.at(t, k) - ly
    forward_substitution(l_matrix, t, y ++ [t_ly], k + 1, dim)
  end

  defp forward_substitution(_, _, y, k, dim) when k == dim do
    y
  end

  defp backward_substitution(u_matrix, y, _, k, dim) when k == dim do
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
  def const_multiple(const, a) when is_number(a) do
    const * a
  end

  def const_multiple(const, a) when is_list(a) do
    Enum.map(a, &const_multiple(const, &1))
  end

  @doc """
  A matrix is added by a constant.
  ## Examples
      iex> MatrixOperation.const_addition(1, [1.0, 2.0, 3.0])
      [2.0, 3.0, 4.0]
  """
  def const_addition(const, a) when is_number(a) do
    const + a
  end

  def const_addition(const, a) when is_list(a) do
    Enum.map(a, &const_addition(const, &1))
  end

  @doc """
  Inverse Matrix
  ## Examples
      iex> MatrixOperation.inverse_matrix([[1, 1, -1], [-2, -1, 1], [-1, -2, 1]])
      [[-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [-3.0, -1.0, -1.0]]
  """
  def inverse_matrix(a) when is_list(hd(a)) do
    det_a = determinant(a)

    create_index_matrix(a)
    |> Enum.map(&map_index_row(a, det_a, &1))
    |> transpose
  end

  def inverse_matrix(_) do
    nil
  end

  defp create_index_matrix(a) do
    index_list = Enum.to_list(1..length(a))
    Enum.map(index_list, fn x -> Enum.map(index_list, &[x, &1]) end)
  end

  defp map_index_row(_, 0, _) do
    nil
  end

  defp map_index_row(a, det_a, row) do
    Enum.map(row, &minor_matrix(a, det_a, &1))
  end

  # minor_matrix
  defp minor_matrix(a, det_a, [row_number, column_number]) do
    det_temp_a =
      delete_one_row(a, row_number)
      |> transpose
      |> delete_one_row(column_number)
      |> determinant

    if(rem(row_number + column_number, 2) == 0,
      do: det_temp_a / det_a,
      else: -1 * det_temp_a / det_a
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
    column_number_a = row_column_matrix(a) |> Enum.at(1)
    row_number_b = row_column_matrix(b) |> Enum.at(0)
    if(column_number_a == row_number_b, do: product_sub(a, b), else: nil)
  end

  defp product_sub(a, b) do
    Enum.map(a, fn row_a ->
      transpose(b)
      |> Enum.map(&inner_product(row_a, &1))
    end)
  end

  defp inner_product(row_a, column_b) do
    Enum.zip(row_a, column_b)
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
    row_column_a = row_column_matrix(a)
    row_column_b = row_column_matrix(b)
    if(row_column_a == row_column_b, do: add_sub(a, b), else: nil)
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
    row_column_a = row_column_matrix(a)
    row_column_b = row_column_matrix(b)
    if(row_column_a == row_column_b, do: subtract_sub(a, b), else: nil)
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
  def hadamard_power(a, n) do
    Enum.map(a, &Enum.map(&1, fn x -> :math.pow(x, n) end))
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
  eigenvalue [R^2×R^2/R^3×R^3 matrix]
  ## Examples
    iex> MatrixOperation.eigenvalue([[3, 1], [2, 2]])
    [4.0, 1.0]
    iex> MatrixOperation.eigenvalue([[6, -3], [4, -1]])
    [3.0, 2.0]
    iex> MatrixOperation.eigenvalue([[1, 1, 1], [1, 2, 1], [1, 2, 3]])
    [4.561552806429505, 0.43844714673139706, 1.0000000468390973]
    iex> MatrixOperation.eigenvalue([[1, 2, 3], [2, 1, 3], [3, 2, 1]])
    [5.999999995559568, -2.000000031083018, -0.99999996447655]
  """
  # 2×2 algebra method
  def eigenvalue([[a11, a12], [a21, a22]]) do
    quadratic_formula(1, -a11 - a22, a11 * a22 - a12 * a21)
  end

  # 3×3 algebratic method
  def eigenvalue([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]) do
    a = -1
    b = a11 + a22 + a33
    c = a21 * a12 + a13 * a31 + a32 * a23 - a11 * a22 - a11 * a33 - a22 * a33

    d =
      a11 * a22 * a33 + a12 * a23 * a31 + a13 * a32 * a21 - a11 * a32 * a23 - a22 * a31 * a13 -
        a33 * a21 * a12

    cubic_formula(a, b, c, d)
  end

  def eigenvalue(_a) do
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

  def cubic_formula(a, b, c, d)
      when -4 * a * c * c * c - 27 * a * a * d * d + b * b * c * c + 18 * a * b * c * d -
             4 * b * b * b * d < 0 do
    nil
  end

  def cubic_formula(a, b, c, d) do
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
  end

  def cubic_formula_sub(x) when x < 0 do
    [0, :math.sqrt(-x)]
  end

  def cubic_formula_sub(x) do
    [:math.sqrt(x), 0]
  end

  def atan(x) when x < 0 do
    y = atan(-x)
    -1 * y
  end

  def atan(x) do
    atan_sub(x, 0, 0)
  end

  def atan_sub(x, z, s) when z < x do
    del = 0.0000001
    z = z + del
    s = s + del / (z * z + 1)
    atan_sub(x, z, s)
  end

  def atan_sub(_, _, s) do
    s
  end

  def csqrt([re, _im], _n) when re == 0 do
    nil
  end

  def csqrt([re, im], n) do
    r = :math.pow(re * re + im * im, 0.5 / n)
    re2 = r * :math.cos(atan(im / re) / n)
    im2 = r * :math.sin(atan(im / re) / n)
    [re2, im2]
  end

  @doc """
    Matrix diagonalization [R^2×R^2/R^3×R^3 matrix]
    #### Examples
      iex> MatrixOperation.diagonalization([[1, 3], [4, 2]])
      [[5.0, 0], [0, -2.0]]
      iex> MatrixOperation.diagonalization([[1, 2, 3], [2, 1, 3], [3, 2, 1]])
      [
        [5.999999995559568, 0, 0],
        [0, -2.000000031083018, 0],
        [0, 0, -0.99999996447655]
      ]
    """
  def diagonalization(a) do
    eigenvalue(a)
    |> diagonalization_condition
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
  Power iteration method (maximum eigen value and eigen vector)
  ## Examples
    iex> MatrixOperation.power_iteration([[3, 1], [2, 2]], 100)
    [4.0, [2.8284271247461903, 2.8284271247461903]]
    iex> MatrixOperation.power_iteration([[1, 1, 2], [0, 2, -1], [0, 0, 3]], 100)
    [3.0, [1.0, -2.0, 2.0]]
  """
  def power_iteration(a, max_k) do
    init_vec = random_column(length(a))
    xk_pre = power_iteration_sub(a, init_vec, max_k)
    # eigen vector
    [xk_vec] = product(a, xk_pre) |> transpose
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

  defp power_iteration_sub(a, v, max_k) do
    # Normarization is for overflow suppression
    Enum.reduce(1..max_k, v, fn _, acc ->
      vp = product(a, acc)
      [vpt] = transpose(vp)
      const_multiple(1 / :math.sqrt(inner_product(vpt, vpt)), vp)
    end)
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
    |> Enum.map(&Enum.map(&1, fn x -> x - Enum.sum(&1)/length(&1) end))
    xt  = transpose(x)
    xtx = product(x, xt)
    const_multiple(1/length(xt), xtx)
  end

end

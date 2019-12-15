defmodule MatrixOperation do
  @moduledoc """
  Documentation for Matrix operation library.
  """

  @doc """
    Numbers of row and column of a matrix are got.

    #### Examples
      iex> MatrixOperation.row_column_matrix([[3, 2, 3], [2, 1, 2]])
      [2, 3]
    """
  def row_column_matrix(a) when is_list(hd(a)) do
    columns_number = Enum.map(a, & row_column_matrix_sub(&1, 0))
    max_number = Enum.max(columns_number)
    if( max_number == Enum.min(columns_number), do: [length(a), max_number] , else: nil)
  end
  def row_column_matrix(a) do
    nil
  end
  defp row_column_matrix_sub(row_a, i) when i != length(row_a) do
    if(is_number(Enum.at(row_a, i)), do: row_column_matrix_sub(row_a, i + 1) , else: nil)
  end
  defp row_column_matrix_sub(row_a, i) when i == length(row_a) do
    i
  end


  @doc """
  	A element of a matrix is got.

    #### Examples
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

    #### Examples
      iex> MatrixOperation.get_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9] ], 1)
      [1, 2, 3]
    """
  def get_one_row(matrix, row_index) do
    matrix
    |> Enum.at(row_index - 1)
  end


  @doc """
    A column of a matrix is got.

    #### Examples
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

    #### Examples
        iex> MatrixOperation.delete_one_row([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3)
        [[1, 2, 3], [4, 5, 6]]
    """
  def delete_one_row(matrix, delete_index) do
    matrix
    |> Enum.with_index
    |> Enum.reject(fn {_, i} -> i == (delete_index - 1) end)
    |> Enum.map(fn {x, _} -> x end)
  end


  @doc """
  	Transpose of a matrix

    #### Examples
        iex> MatrixOperation.transpose([[1.0, 2.0], [3.0, 4.0]])
        [[1.0, 3.0], [2.0, 4.0]]
    """
  def transpose(a) do
  	Enum.zip(a)
  	|> Enum.map(& Tuple.to_list(&1))
  end

  @doc """
    Trace of a matrix

    #### Examples
        iex> MatrixOperation.trace([[1.0, 2.0], [3.0, 4.0]])
        5.0
    """
  def trace(a) do
    [row, column] = row_column_matrix(a)
    a_index = add_index(a)
    Enum.map(a_index, & trace_sub(&1, row, column))
    |> Enum.sum
  end
  defp trace_sub(_, row, column) when row != column do
    nil
  end
  defp trace_sub([index, row_list], _, _) do
    Enum.at(row_list, index-1)
  end


  @doc """
    A determinant of a n×n square matrix is got.

    #### Examples
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
    a_with_index -- [row]
    |> Enum.map(& Enum.at(&1, 1))
    |> Enum.map(& Enum.drop(&1, 1))
  end
  # 1×1 matrix
  defp determinant_sub(_, a) when length(a) == 1 do
    nil
  end
  # 2×2 matrix
  defp determinant_sub(co, [[a11, a12], [a21, a22]]) do
    co * ( a11 * a22 - a12 * a21 )
  end
  # 3×3 or over matrix
  defp determinant_sub(co, a) do
    a_with_index = add_index(a)
    Enum.map(a_with_index, & determinant_sub((-1 + 2 * rem(hd(&1), 2)) * co * hd(Enum.at(&1, 1)), minor_matrix(a_with_index, &1)))
    |> Enum.sum
  end
  # add index
  defp add_index(a) do
    Stream.iterate(1, & (&1 + 1))
    |> Enum.zip(a)
    |> Enum.map(& &1 |> Tuple.to_list)
  end


  @doc """
    Cramer's rule

    #### Examples
        iex> MatrixOperation.cramer([1, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0)
        1.0
    """
  def cramer(t, a, select_index) do
    det_a = determinant(a)
    cramer_sub(t, a, select_index, det_a)
  end

  defp cramer_sub(_, _, _, nil), do: nil
  defp cramer_sub(_, _, _, 0), do: nil
  defp cramer_sub(t, a, select_index, det_a) do
    rep_det_a = transpose(a) |> replace_element_in_list(select_index, t, 0, []) |> determinant
    rep_det_a / det_a
  end

  defp replace_element_in_list(list, i, replace_element, i, output) when i < length(list) do
    replace_element_in_list(list, i, replace_element, i + 1, output ++ [replace_element])
  end
  defp replace_element_in_list(list, select_index, replace_element, i, output) when i < length(list) do
    replace_element_in_list(list, select_index, replace_element, i + 1, output ++ [Enum.at(list, i)])
  end
  defp replace_element_in_list(list, select_index, replace_element, i, output) when i == length(list), do: output


  @doc """
    Liner equations are solved.

    #### Examples
        iex> MatrixOperation.linear_equations([1, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        [1.0, 0.0, 0.0]
        iex> MatrixOperation.linear_equations([3, -7, 4], [[0, -2, 1], [-1, 1, -4], [3, 3, 1]])
        [2.0, -1.0, 1.0]
    """
  def linear_equations(t, a) do
    det_a = determinant(a)
    condition_det(t, a, det_a)
  end
  defp condition_det(_, _, 0) do
    nil
  end
  defp condition_det(t, a, det_a) do
    linear_equations_sub(t, a, 0, [])
  end
  defp linear_equations_sub(t, a, i, output) when i < length(a) do
    linear_equations_sub(t, a, i + 1, output ++ [cramer(t, a, i)])
  end
  defp linear_equations_sub(t, a, i, output) when i == length(a) do
    output
  end


  @doc """
    A matrix is multiplied by a constant.

    #### Examples
        iex> MatrixOperation.const_multiple(-1, [1.0, 2.0, 3.0])
        [-1.0, -2.0, -3.0]
        iex> MatrixOperation.const_multiple(2, [[1, 2, 3], [2, 2, 2], [3, 8, 9]])
        [[2, 4, 6], [4, 4, 4], [6, 16, 18]]
    """
  def const_multiple(const, a) when is_number(a) do
    const * a
  end
  def const_multiple(const, a) when is_list(a) do
    Enum.map(a, & const_multiple(const, &1))
  end


  @doc """
    Inverse Matrix

    #### Examples
        iex> MatrixOperation.inverse_matrix([[1, 1, -1], [-2, -1, 1], [-1, -2, 1]])
        [[-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [-3.0, -1.0, -1.0]]
    """
  def inverse_matrix(a) when is_list(hd(a)) do
    det_a = determinant(a)
    create_index_matrix(a)
    |> Enum.map(& map_index_row(a, det_a, &1))
    |> transpose
  end
  def inverse_matrix(_) do
    nil
  end
  defp create_index_matrix(a) do
    index_list = Enum.to_list(1..length(a))
    Enum.map(index_list, fn x -> Enum.map(index_list, & [x, &1]) end)
  end
  defp map_index_row(_, 0, _) do
    nil
  end
  defp map_index_row(a, det_a, row) do
    Enum.map(row, & minor_matrix(a, det_a, &1))
  end
  # minor_matrix
  defp minor_matrix(a, det_a, [row_number, column_number]) do
    det_temp_a = delete_one_row(a, row_number)
                 |> transpose
                 |> delete_one_row(column_number)
                 |> determinant
    if(rem(row_number + column_number, 2) == 0, do: det_temp_a / det_a , else: -1 * det_temp_a / det_a)
  end


  @doc """
    Matrix product

    #### Examples
        iex> MatrixOperation.product([[3, 2, 3], [2, 1, 2]], [[2, 3], [2, 1], [3, 5]])
        [[19, 26], [12, 17]]
    """
  def product(a, b) do
    check_product(a, b)
  end
  defp check_product(a, b) do
    column_number_a = row_column_matrix(a) |> Enum.at(1)
    row_number_b = row_column_matrix(b) |> Enum.at(0)
    if( column_number_a == row_number_b, do: product_sub(a, b), else: nil)
  end
  defp product_sub(a, b) do
    Enum.map(a, fn row_a ->
      transpose(b)
      |> Enum.map(& inner_product(row_a, &1))
    end)
  end
  defp inner_product(row_a, column_b) do
    Enum.zip(row_a, column_b)
    |> Enum.map(& Tuple.to_list(&1))
    |> Enum.map(& Enum.reduce(&1, fn x, acc -> x * acc end))
    |> Enum.sum
  end


  @doc """
    Matrix addition

    #### Examples
        iex> MatrixOperation.add([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[5, 5, 4], [5, 3, 4]]
    """
  def add(a, b) do
    check_add(a, b)
  end
  defp check_add(a, b) do
    row_column_a = row_column_matrix(a)
    row_column_b = row_column_matrix(b)
    if( row_column_a == row_column_b, do: add_sub(a, b), else: nil)
  end
  defp add_sub(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} ->
        Enum.zip(x, y)
        |> Enum.map(& Tuple.to_list(&1))
        |> Enum.map(& Enum.reduce(&1, fn x, acc -> x + acc end))
      end)
  end


  @doc """
    Hadamard product

    #### Examples
        iex> MatrixOperation.hadamard_product([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[6, 6, 3], [6, 2, 4]]
    """
  def hadamard_product(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> hadamard_product_sub(x, y) end)
  end
  defp hadamard_product_sub(row_a, row_b) do
    Enum.zip(row_a, row_b)
    |> Enum.map(& Tuple.to_list(&1))
    |> Enum.map(& Enum.reduce(&1, fn x, acc -> x * acc end))
  end


  @doc """
    Matrix subtraction

    #### Examples
        iex> MatrixOperation.subtract([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[1, -1, 2], [-1, -1, 0]]
    """
  def subtract(a, b) do
    check_subtract(a, b)
  end
  defp check_subtract(a, b) do
    row_column_a = row_column_matrix(a)
    row_column_b = row_column_matrix(b)
    if( row_column_a == row_column_b, do: subtract_sub(a, b), else: nil)
  end
  defp subtract_sub(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} ->
        Enum.zip(x, y)
        |> Enum.map(& Tuple.to_list(&1))
        |> Enum.map(& Enum.reduce(&1, fn x, acc -> acc - x end))
      end)
  end


  @doc """
    Hadamard division

    #### Examples
        iex> MatrixOperation.hadamard_division([[3, 2, 3], [2, 1, 2]], [[2, 3, 1], [3, 2, 2]])
        [[1.5, 0.6666666666666666, 3.0], [0.6666666666666666, 0.5, 1.0]]
    """
  def hadamard_division(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> hadamard_division_sub(x, y) end)
  end
  defp hadamard_division_sub(row_a, row_b) do
    Enum.zip(row_a, row_b)
    |> Enum.map(& Tuple.to_list(&1))
    |> Enum.map(& Enum.reduce(&1, fn x, acc -> acc / x end))
  end


  @doc """
    Hadamard power

    #### Examples
        iex> MatrixOperation.hadamard_power([[3, 2, 3], [2, 1, 2]], 2)
        [[9.0, 4.0, 9.0], [4.0, 1.0, 4.0]]
    """
  def hadamard_power(a, n) do
    Enum.map(a, & Enum.map(&1, fn x -> :math.pow(x, n) end))
  end


  @doc """
    Tensor product

    #### Examples
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
    Enum.map(a, & tensor_product(&1, b))
  end
  def tensor_product(a, b) when is_number(a) do
    const_multiple(a, b)
  end

end

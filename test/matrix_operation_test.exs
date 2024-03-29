defmodule MatrixOperationTest do
  use ExUnit.Case

  doctest MatrixOperation
  import MatrixOperation

  describe "inverse_matrix" do
    test "property" do
      a = [[1, 1, -1], [-2, -1, 1], [-1, -2, 1]]
      result = a
      |> inverse_matrix()
      |> product(a)

      expect = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
      assert result == expect
    end
  end

  @tag :skip
  describe "mp_inverse_matrix" do
    test "property" do
      a = [[1, 1, -1], [-2, -1, 1], [-1, -2, 1]]
      result = a
      |> mp_inverse_matrix()
      |> product(a)

      expect = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
      # TODO: The Moore-Penrose general inverse matrix does not match the inverse matrix
      assert result  == expect
    end
  end

  describe "eigh" do
    test "property" do
      a = [[1, 4, 5], [4, 2, 6], [5, 6, 3]]
      {evals, evecs} = eigh(a)
      evals_matrix = [
        [Enum.at(evals, 0), 0, 0],
        [0, Enum.at(evals, 1), 0],
        [0, 0, Enum.at(evals, 2)]
      ]
      evecs_t = transpose(evecs)
      left = product(evals_matrix, evecs_t) |> Enum.map(& Enum.map(&1, fn x -> Float.round(x, 7) end))
      right = product(evecs_t, a) |> Enum.map(& Enum.map(&1, fn x -> Float.round(x, 7) end))
      assert left == right
    end
  end

  describe "eigen" do
    test "property" do
      a = [[1, 4, 5], [4, 2, 6], [5, 6, 3]]
      {evals, evecs} = eigen(a)
      evals_matrix = [
        [Enum.at(evals, 0), 0, 0],
        [0, Enum.at(evals, 1), 0],
        [0, 0, Enum.at(evals, 2)]
      ]
      left = product(evals_matrix, evecs) |> Enum.map(& Enum.map(&1, fn x -> Float.round(x, 7) end))
      right = product(evecs, a) |> Enum.map(& Enum.map(&1, fn x -> Float.round(x, 7) end))
      assert left == right
    end
  end

  describe "solve_sle" do
    test "property" do
      # Solve simultaneous linear equations: a.x = y
      a = [[4, 1, 1], [1, 3, 1], [2, 1, 5]]
      y = [[9], [10], [19]]
      insert_in_vec = fn x -> [x] end
      x = a
      |> solve_sle(y)
      |> insert_in_vec.()
      |> transpose()
      assert product(a, x) == y
    end
  end

  @tag :skip
  describe "svd" do
    test "property" do
      # Singular Value Decomposition: a = u.s.v_t
      a = [[4, 1, 2], [1, 3, 1], [2, 1, 5]]
      {sv, u, v} = svd(a)
      v_t = transpose(v)
      s = [
        [Enum.at(sv, 0), 0, 0],
        [0, Enum.at(sv, 1), 0],
        [0, 0, Enum.at(sv, 2)]
      ]
      result = u
      |> product(s)
      |> product(v_t)
      |> Enum.map(& Enum.map(&1, fn x -> Float.round(x, 7) end))
      # TODO: The signs don't match
      assert a == result
    end
  end
end

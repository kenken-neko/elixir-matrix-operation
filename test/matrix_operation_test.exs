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
end

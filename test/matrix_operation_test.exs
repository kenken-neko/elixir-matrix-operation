defmodule MatrixOperationTest do
  use ExUnit.Case

  doctest MatrixOperation
  import MatrixOperation

  describe "inverse_matrix" do
    test "property" do
      a = [[1, 1, -1], [-2, -1, 1], [-1, -2, 1]]
      result = a
      |> MatrixOperation.inverse_matrix()
      |> product(a)

      expect = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
      assert result == expect
    end
  end
end

defmodule MatrixOperation.MixProject do
  use Mix.Project

  def project do
    [
      app: :matrix_operation,
      version: "0.5.0",
      elixir: "~> 1.12.3",
      description: "Matrix operation library",
      start_permanent: Mix.env() == :prod,
      package: [
        maintainers: ["tanaka kenta"],
        licenses: ["MIT"],
        links: %{"GitHub" => "https://github.com/kenken-neko/elixir-matrix-operation"}
      ],
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [{:ex_doc, "~> 0.10", only: :dev}]
  end
end

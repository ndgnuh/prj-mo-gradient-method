from src.vector_function import VectorFunction


f = VectorFunction(
    [lambda x: x[0]**2 + 3 * (x[1] - 1)**2,
     lambda x: 2 * (x[0] - 1)**2 + x[1]**2, ]
)

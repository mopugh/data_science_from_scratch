from typing import List, Callable, TypeVar, Iterator
import random

Vector = List[float]
T = TypeVar('T')

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) ->float:
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad


def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates 'batch_size'-sized minibatches from dataset"""
    # start index 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

if __name__ == '__main__':

    # x ranges from -50 to 49, y is always 20 * x + 5
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

    # Start with random values for slope and intercept
    theta = [random.uniform(-1,1), random.uniform(-1,1)]

    learning_rate = 0.001

    # for epoch in range(5000):
    #     # Compute the mean of the gradients
    #     grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    #     # Take a step in that direction
    #     theta = gradient_step(theta, grad, -learning_rate)
    #     print(epoch, theta)

    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta

    assert 19.9 < slope < 20.1
    assert 4.9 < intercept < 5.1

import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from typing import NamedTuple, Union, List, Callable
from jaxtyping import Array, Float
import jax.numpy as jnp
from dynamax.parameters import ParameterProperties
import itertools
from itertools import product
from utils.diffrax_utils import adjust_rhs
from flax import linen as nn

class LearnableNN_TwoLayerGeLU(NamedTuple):
    """Two-layer neural network with Gaussian Error Linear Units
    weights1: weights of the first layer
    bias1: bias of the first layer
    weights2: weights of the second layer
    bias2: bias of the second layer

    f(x) = weights2 @ gelu(weights1 @ x + bias1) + bias2
    """

    weights1: Union[Float[Array, "hidden_dim input_dim"], ParameterProperties]
    bias1: Union[Float[Array, "hidden_dim"], ParameterProperties]
    weights2: Union[Float[Array, "output_dim hidden_dim"], ParameterProperties]
    bias2: Union[Float[Array, "output_dim"], ParameterProperties]
    scale: Union[Float, ParameterProperties]

    def f(self, x, u=None, t=None):
        """This rhs operates in original space."""

        # compute derivative given by NN
        foo = self.weights2 @ nn.gelu(self.weights1 @ x + self.bias1) + self.bias2

        rhs = adjust_rhs(x, foo, lower_bound=-100, upper_bound=100,
               lower_bound_derivative=-1000, upper_bound_derivative=1000)

        return rhs
    
class LearnableKLFeatures(NamedTuple):
    """Model based on truncated Karhunen-Loeve expansion of Gaussian Process with RBF kernel.

    Represents a function f: R^d_in -> R^d_out with learnable parameters.
    At a high level, the model is:

        f(x) = output_stdev * weights @ sqrt(eigenvalues) @ eigenvectors(x)

    The model is based on the following equation:

    Each output dimension for i=1,...,d_out is represented as:
        f_i(x) = output_stdev_i * sum_n weights_n,i * sqrt(lambda_n) * phi_n(x),

            where n = (n1,...nd) a multi-index, lambda_n are eigenvalues, phi_n(x) are basis functions, and output_stdev_i a scalar standard deviation.

        Basis functions can be decomposed as:

              phi_n(x) = prod_j phi_j,n_j(x_j)

            with
                phi_j,n_j(x_j) = sqrt(1 / a_j) * cos(n_j * pi * x_j / a_j) for even n_j
                               = sqrt(1 / a_j) * sin(n_j * pi * x_j / a_j) for odd n_j

                and a_j = 1/2 the domain length in dimension j.

            NOTES:
                - a_j is absorbed into the normalization_factor, which is computed externally
                - we use all cosines, and shift the phase of the odd terms by pi/2
                - The parameters A and B are precomputed for efficiency, and represent the phase and frequency
                of each basis function.

        Eigenvalues lambda_n are precomputed and fixed as:

            lambda_n = prod_j lambda_j,n_j

            with
                lambda_j,n_j = ell_j * sqrt(2 / a_j) * exp(-1 / sqrt(2) * (n_j * ell_j * pi / a_j)^2)

            where ell_j is the length scale in dimension j.

            NOTES:
                - Technically, the eigenvalues contain output_stdev, but we separate them so these stdevs easily be learned.
    """

    weights: Union[Float[Array, "output_dim num_basis"], ParameterProperties]
    output_stdevs: Union[Float[Array, "output_dim"], ParameterProperties]
    eigenvalues: Union[Float[Array, "num_basis"], ParameterProperties]  # Number of eigenfunctions per dimension
    A: Union[Float[Array, "num_basis input_dim"], ParameterProperties]
    B: Union[Float[Array, "num_basis input_dim"], ParameterProperties]
    normalization_factor: Union[Float, ParameterProperties]

    def f(self, x, u=None, t=None):

        # Compute basis functions efficiently using precomputed A and B
        phi = jnp.prod(jnp.cos(self.A * x + self.B), axis=1) * self.normalization_factor

        # Compute f(x) using eigenvalues and standard deviations
        f_x = jnp.dot(phi * jnp.sqrt(self.eigenvalues), self.weights)  # Shape (d_output,)
        f_x *= self.output_stdevs  # Apply standard deviations

        # Adjust the output to prevent numerical instability
        adj_rhs = adjust_rhs(
            x, f_x, lower_bound=-100, upper_bound=100, lower_bound_derivative=-1000, upper_bound_derivative=1000
        )

        return adj_rhs
    
    def variance(self, x, cov_matrix):
        """
        Compute the variance of the function at x given the covariance matrix of the weights.

        Parameters:
        x: Input vector of shape (input_dim,)
        cov_matrix: Covariance matrix of the weights of shape (num_basis, output_dim, num_basis, output_dim)

        Returns:
        Variance of the function at x.

        WARNING: we are NOT using the cross-covariance between output dimensions, which may underestimate the variance???
        """
        # Compute basis functions efficiently using precomputed A and B
        phi = jnp.prod(jnp.cos(self.A * x + self.B), axis=1) * self.normalization_factor

        # multiply the basis functions by the square root of the eigenvalues
        phi *= jnp.sqrt(self.eigenvalues)

        # Compute the variance using the covariance matrix of the weights, then rescale by the output standard deviations
        # WARNING: we are NOT using the cross-covariance between output dimensions, which may underestimate the variance???
        variance = self.output_stdevs**2 * jnp.stack([phi @ cov_matrix[:, i, :, i] @ phi for i in range(self.A.shape[1])])

        return variance

def precompute_eigenvalues_and_basis(truncation, length_scales, domain):
    """
    Precompute eigenvalues and eigenvectors for the KL expansion.

    Parameters:
        truncation (int): Number of terms to include in the KL expansion.
        length_scales (list or np.ndarray): Length scales \ell for each dimension, shape (d_input,).
        domain (list of tuples): Domain for each dimension, e.g., [(-1, 1), (-1, 1)] for 2D.

    Returns:
        tuple: eigenvalues (array), A (matrix), B (vector), normalization_factor (scalar)
    """
    d_input = len(length_scales)  # Number of input dimensions
    indices = list(product(range(truncation), repeat=d_input))  # All tensorized index combinations
    num_basis = len(indices)  # Total number of basis functions

    # Compute domain scaling factor
    a = jnp.array([(dom[1] - dom[0]) / 2 for dom in domain])

    # Precompute normalization factor
    normalization_factor = 1 / jnp.sqrt(jnp.prod(a))

    # Precompute eigenvalues
    lambda_n = jnp.zeros(num_basis)
    for idx, index_set in enumerate(indices):
        lambda_n = lambda_n.at[idx].set(jnp.prod(jnp.array([
            length_scales[dim] * jnp.sqrt(2 * a[dim]) * \
            jnp.exp(-1 / jnp.sqrt(2) * (index_set[dim] * length_scales[dim] * jnp.pi / a[dim])**2)
            for dim in range(d_input)
        ]))) # Shape (num_basis,)

    # Precompute A and B for basis function
    A = jnp.array([
        [index_set[dim] * jnp.pi / a[dim] for dim in range(d_input)]
        for index_set in indices
    ])  # Shape (num_basis, d_input)

    B = jnp.array([
        [0 if index_set[dim] % 2 == 0 else -jnp.pi / 2 for dim in range(d_input)]
        for index_set in indices
    ])  # Shape (num_basis, d_input)


    return lambda_n, A, B, normalization_factor


class LearnablePolynomialModel(NamedTuple):
    """Model with learnable parameters applied to an efficient monomial dictionary.

    weights: weights for the linear combination of dictionary terms
    exponents: matrix of exponents for each monomial term

    f(x) = weights @ monomial_terms(x)
    """

    weights: Union[Float[Array, "output_dim num_terms"], ParameterProperties]
    exponents: Union[Float[Array, "num_terms input_dim"], ParameterProperties]

    def f(self, x, u=None, t=None):
        """
        Compute the linear combination of monomial terms with learnable parameters.

        Parameters:
        x: Input vector of shape (input_dim,)

        Returns:
        Linear combination of monomial terms.
        """

        # Compute all monomial terms in one vectorized operation
        monomial_terms = jnp.prod(x**self.exponents, axis=1)

        # Compute the linear combination using weights
        foo = self.weights @ monomial_terms

        adj_rhs = adjust_rhs(x, foo, lower_bound=-100, upper_bound=100,
               lower_bound_derivative=-1000, upper_bound_derivative=1000)

        return adj_rhs

class LearnableDictionaryModel(NamedTuple):
    """Function with learnable parameters applied to a dictionary of arbitrary transformations.

    weights: weights for the linear combination of dictionary terms
    dictionary_functions: list of transformations applied to input x to generate terms

    f(x) = weights @ dictionary_terms(x)
    """

    weights: Union[Float[Array, "output_dim num_terms"], Array]
    dictionary_functions: List[Callable[[Array], Array]]  # Arbitrary transformations, including constants

    def f(self, x, u=None, t=None):
        """
        Compute the linear combination of dictionary terms with learnable parameters.

        Parameters:
        x: Input vector of shape (input_dim,)

        Returns:
        Linear combination of transformed terms.
        """
        # Apply each transformation to x and stack results into the dictionary terms
        dictionary_terms = jnp.stack([func(x) for func in self.dictionary_functions], axis=0)

        # Compute the linear combination using weights
        foo = self.weights @ dictionary_terms
    
        adj_rhs = adjust_rhs(x, foo, lower_bound=-100, upper_bound=100, 
            lower_bound_derivative=-1000, upper_bound_derivative=1000)

        return adj_rhs


def generate_monomial_exponents(N: int, D: int) -> Array:
    """
    Generates an array of exponents representing all N-input to 1-output monomials
    up to degree D. Each row of the output array represents the exponents for a monomial term.

    Parameters:
    N: Number of input variables (dimension of input vector)
    D: Maximum degree of monomials

    Returns:
    Integer Array of shape (num_terms, N), where each row represents exponents for one monomial term

    NB: It is best to return integers, as this has implications for how gradients are computed.
    """
    # List to store all exponent combinations
    exponents = []

    # Generate all exponent combinations up to degree D
    for degree in range(0, D + 1):  # Including the constant term with degree 0
        for combination in itertools.combinations_with_replacement(range(N), degree):
            # Create an array for the term's exponents, initialized to 0
            term_exponents = jnp.zeros(N, dtype=jnp.float32)
            # Count occurrences in the combination to set exponents
            for idx in combination:
                term_exponents = term_exponents.at[idx].add(1.0)
            exponents.append(term_exponents)

    exponents = jnp.stack(exponents, axis=0) 

    # convert to integers
    exponents = exponents.astype(jnp.int32)

    return exponents


def generate_monomial_transformations(N: int, D: int) -> List[Callable[[Array], Array]]:
    """
    Generates a list of transformations representing all N-input to 1-output monomials
    up to degree D. Each transformation is a function that takes an input vector x and
    returns a monomial term of x.

    Parameters:
    N: Number of input variables (dimension of input vector)
    D: Maximum degree of monomials

    Returns:
    List of functions, each representing a monomial term
    """
    # List to store all monomial transformations
    transformations = []

    # Generate all exponent combinations up to degree D
    for degree in range(1, D + 1):
        for exponents in itertools.combinations_with_replacement(range(N), degree):

            def monomial(x, exponents=exponents):
                # Raise each component of x to the appropriate power and multiply
                term = jnp.prod(x[jnp.array(exponents)])
                return term

            transformations.append(monomial)

    # Include a constant term as the first transformation
    transformations.insert(0, lambda x: jnp.ones_like(x[0]))  # Constant term (e.g., 1)

    return transformations

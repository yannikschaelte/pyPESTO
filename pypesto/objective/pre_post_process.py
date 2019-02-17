import numpy as np


from .constants import GRAD, HESS, RES, SRES


class PrePostProcessor:
    """
    Implements the methods preprocess and postprocess that are called at the
    beginning and at the end of the objective call, in order to handle the
    mapping of optimization parameters to simulation parameters.

    This class acts as a dummy base implementation, not performing any
    changes on the passed objects.
    """

    def __init__(self):
        pass

    def preprocess(self, x):  # pylint: disable=R0201
        """
        Just return x without modifications.
        """
        return x

    def postprocess(self, result):  # pylint: disable=R0201
        """
        Convert all arrays into np.ndarrays if necessary, and return them
        without further modifications.
        """
        result = PrePostProcessor.as_ndarrays(result)
        return result

    @staticmethod
    def as_ndarrays(result):
        """
        Convert all array_like objects to np.ndarrays. This has the advantage
        of a uniform output datatype which offers various methods to assess
        the data.
        """
        keys = [GRAD, HESS, RES, SRES]
        for key in keys:
            if key in result:
                value = result[key]
                if value is not None:
                    result[key] = np.array(value)
        return result


class FixedParametersProcessor(PrePostProcessor):
    """
    Extends the processor to handle the fixing of parameters.
    """

    def __init__(self, dim_full,
                 x_free_indices, x_fixed_indices, x_fixed_vals):
        super().__init__()
        self.dim_full = dim_full
        self.x_free_indices = x_free_indices
        self.x_fixed_indices = x_fixed_indices
        self.x_fixed_vals = x_fixed_vals

    def preprocess(self, x):
        x = super().preprocess(x)

        x_full = np.zeros(self.dim_full)
        x_full[self.x_free_indices] = x
        x_full[self.x_fixed_indices] = self.x_fixed_vals

        return x_full

    def postprocess(self, result):
        result = super().postprocess(result)

        if GRAD in result:
            grad = result[GRAD]
            if grad.size == self.dim_full:
                grad = grad[self.x_free_indices]
                result[GRAD] = grad
        if HESS in result:
            hess = result[HESS]
            if hess.shape[0] == self.dim_full:
                hess = hess[np.ix_(self.x_free_indices, self.x_free_indices)]
                result[HESS] = hess
        if SRES in result:
            sres = result[SRES]
            if sres.shape[-1] == self.dim_full:
                sres = sres[..., self.x_free_indices]
                result[SRES] = sres

        return result

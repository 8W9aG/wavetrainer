"""The prototype normaliser class."""

from ..fit import Fit
from ..params import Params


class Normaliser(Params, Fit):
    """The prototype normaliser class."""

    @classmethod
    def name(cls) -> str:
        """The name of the normaliser."""
        raise NotImplementedError("name not implemented in parent class.")

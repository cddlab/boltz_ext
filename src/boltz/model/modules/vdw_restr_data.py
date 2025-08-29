from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .chiral_data import length


@dataclass
class VdwRestrData:
    """Class for vdw restraints data."""

    aid0: int
    aid1: int
    r0: float
    w: float = 0.05
    # half: bool = True

    def is_valid(self) -> bool:
        """Check if the bond data is valid."""
        return self.aid0 >= 0 and self.aid1 >= 0 and self.w > 0.0

    def reset_indices(self) -> None:
        """Reset the indices."""
        self.aid0 = -1
        self.aid1 = -1

    def setup(self, ind: int, aid: int) -> None:
        """Set up bond data."""
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        else:
            msg = f"Invalid data {ind=} {aid=}"
            raise ValueError(msg)

    def calc(self, crds: np.ndarray) -> float:
        """Calculate the bond data."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        n1l = length(v1)

        r1 = self.r0
        delta = n1l - r1

        if delta < 0:
            # If half bond and delta is negative, no energy contribution
            return 0.0

        ene = self.w * delta * delta
        return ene

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        """Calculate the gradient."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        n1l = length(v1)

        r1 = self.r0 - self.slack
        delta = r1 / n1l

        if 1.0 < delta:
            # If half bond and delta is negative, no gradient contribution
            return

        con = 2.0 * self.w * (1.0 - delta)
        # if not self.half:
        #     grad[self.aid0] += v1 * con
        grad[self.aid1] -= v1 * con

    def print(self, crds: np.ndarray) -> None:
        """Print the bond data."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        n1l = length(v1)

        print(  # noqa: T201
            f"V {self.aid0}-{self.aid1}:"
            f" cur {n1l:.2f} ref {self.r0:.2f} dif {n1l - self.r0:.2f}"
        )

    def calc_sd(self, crds: np.ndarray) -> None:
        """Calculate squared difference."""
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        v1 = a0 - a1
        n1l = length(v1)
        return (n1l - self.r0) ** 2

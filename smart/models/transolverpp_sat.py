from __future__ import annotations

from .transolverpp import TransolverPPBase


class TransolverPPSAT(TransolverPPBase):
    expects_geo_log_density = True

    def __init__(self, **kwargs):
        super().__init__(density_compensated=True, **kwargs)

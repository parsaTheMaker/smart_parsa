from __future__ import annotations

from .abupt import ABUPTBase


class ABUPTSAT(ABUPTBase):
    expects_geo_log_density = True

    def __init__(self, **kwargs):
        super().__init__(density_compensated=True, **kwargs)

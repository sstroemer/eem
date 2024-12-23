from .entsoepywrapper import EntsoePyWrapper
from .analysis.describe import describe


class Task:
    def __init__(self, epw: EntsoePyWrapper, *, countries, kpis, t0, t1):
        self._countries = countries
        self._kpis = kpis
        self._t0 = t0
        self._t1 = t1

        self._epw = epw
        self._description = describe(self._epw, self)

    @property
    def countries(self):
        return self._countries

    @property
    def kpis(self):
        return self._kpis

    @property
    def t0(self):
        return self._t0

    @property
    def t1(self):
        return self._t1

    @property
    def description(self):
        return self._description

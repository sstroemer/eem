class Task:
    def __init__(self, *, countries, kpis, t0, t1):
        self._countries = countries
        self._kpis = kpis
        self._t0 = t0
        self._t1 = t1

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

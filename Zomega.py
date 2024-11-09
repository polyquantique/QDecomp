from Domega import Domega


class Zomega(Domega):
    def __init__(self, a, b, c, d):
        Domega.__init__(self, (a, 0), (b, 0), (c, 0), (d, 0))

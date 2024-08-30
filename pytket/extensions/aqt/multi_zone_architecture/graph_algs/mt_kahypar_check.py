import importlib

MT_KAHYPAR_INSTALLED = importlib.util.find_spec("mtkahypar") is not None


class MissingMtKahyparInstallError(Exception):

    def __init__(self):
        super().__init__("Graph partitioning requires mtkahypar package")

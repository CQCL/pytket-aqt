import importlib

MT_KAHYPAR_INSTALLED = importlib.util.find_spec("mtkahypar") is not None
"""Whether or not the mtkahypar package is available"""


class MissingMtKahyparInstallError(Exception):

    def __init__(self):
        super().__init__("Graph partitioning requires mtkahypar package")

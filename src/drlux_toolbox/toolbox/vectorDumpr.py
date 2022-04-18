from drlux_toolbox.metaclasses.dumper import Dumper
from loguru import logger


# cacha tutto e poi ogni tot scarica su file
# c'è un dizionario di cose da dumpare
class VectorDumper(Dumper):
    def __init__(self):
        super().__init__()
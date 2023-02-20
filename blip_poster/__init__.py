import sys
import os
from pprint import pprint

_PATH_ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(_PATH_ROOT, "BLIP"))

pprint(sys.path)

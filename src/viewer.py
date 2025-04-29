# viewer.py

import sys
import pickle

# Import your VTK show code here
# (I'm assuming your 'show' function you posted earlier.)

from cadquery import *
from cadquery.vis import show

if __name__ == "__main__":
    # Load pickled args
    with open(sys.argv[1], "rb") as f:
        args, kwargs = pickle.load(f)

    # Call the show function
    show(*args, **kwargs)

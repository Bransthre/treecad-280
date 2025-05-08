import gc
from cadquery import *
from naive_autoregressive.no_interaction_vis import no_interact_show
import cadquery as cq
import matplotlib.pyplot as plt

cad_code = """
w0=cq.Workplane('YZ',origin=(-4,0,0));r=w0.sketch().segment((-76,-66),(-33,-66)).arc((-72,-21),(-33,24)).segment((-76,24)).close().assemble().reset().face(w0.sketch().segment((-21,64),(0,52)).segment((6,63)).segment((-15,74)).close().assemble()).finalize().extrude(-96).union(w0.sketch().push([(59,-47)]).rect(34,54).rect(20,22,mode='s').finalize().extrude(104))
"""

rolls = [
    -60,
    -30,
    0,
    30,
    60,
]
elevations = [-60, -30, 0, 30, 60]
r = None
exec(cad_code)
for roll in rolls:
    for elevation in elevations:
        img = no_interact_show(
            r,
            roll=roll,
            elevation=elevation,
            interact=False,
        )
        plt.imsave(
            f"./img_debug/{2}-img_r{roll}_e{elevation}.png",
            img,
        )

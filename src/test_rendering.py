import gc
from cadquery import *
from naive_autoregressive.no_interaction_vis import no_interact_show
import cadquery as cq
import matplotlib.pyplot as plt

cad_code = """
w0=cq.Workplane('ZX',origin=(0,3,0));r=w0.sketch().segment((-55,-41),(-41,-59)).segment((-40,-58)).segment((-24,-72)).segment((-12,-60)).segment((-27,-47)).segment((0,-27)).segment((0,-29)).segment((77,-29)).segment((77,2)).arc((77,69),(31,21)).segment((0,21)).segment((0,0)).close().assemble().push([(-29,-55)]).circle(5,mode='s').finalize().extrude(-62).union(w0.workplane(offset=-30/2).moveTo(-86.5,4).box(27,28,30)).union(w0.sketch().push([(38,44)]).circle(28).push([(38,43.5)]).rect(8,41,mode='s').finalize().extrude(57))
"""

rolls = [
    -180,
    -150,
    -135,
    -120,
    -90,
    -60,
    -45,
    -30,
    0,
    30,
    45,
    60,
    90,
    120,
    135,
    150,
    180,
]
elevations = [-90, -60, -45, -30, 0, 30, 45, 60, 90]
r = None
exec(cad_code)
for example_id in [266, 366, 466]:
    for roll in rolls:
        for elevation in elevations:
            img = no_interact_show(
                r,
                screenshot=f"./img_debug/{example_id}-img_r{roll}_e{elevation}.png",
                roll=roll,
                elevation=elevation,
                interact=False,
            )
            plt.imsave(
                f"./img_debug/{example_id}-img_r{roll}_e{elevation}.png",
                img,
            )

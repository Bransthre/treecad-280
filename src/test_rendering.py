import gc
from cadquery import *
from no_interaction_vis import no_interact_show
import cadquery as cq

w0 = cq.Workplane("ZX", origin=(0, 66, 0))
r = (
    w0.sketch()
    .segment((-52, -38), (85, 36))
    .segment((85, 100))
    .segment((-52, 100))
    .close()
    .assemble()
    .finalize()
    .extrude(-133)
    .union(
        w0.sketch()
        .segment((-85, -100), (14, -100))
        .segment((14, 16))
        .arc((-31, -13), (-85, -14))
        .close()
        .assemble()
        .push([(61, 60)])
        .circle(11)
        .finalize()
        .extrude(-105)
    )
)

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

for example_id in [266, 366, 466]:
    for roll in rolls:
        for elevation in elevations:
            print(
                f"Rendering example {example_id} with roll {roll} and elevation {elevation}"
            )
            no_interact_show(
                r,
                screenshot=f"./img_debug/{example_id}-img_r{roll}_e{elevation}.png",
                roll=roll,
                elevation=elevation,
                interact=False,
            )

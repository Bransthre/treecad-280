import cadquery as cq
from no_interaction_vis import no_interact_show
import matplotlib.pyplot as plt
import os


def render_cad_from_str(file_content, roll, elevation, rank=0):
    r = None  # The default name of dataset results
    exec(file_content, globals(), locals())
    no_interact_show(
        r,
        screenshot=f"./training_tmp/tmp.png",
        roll=roll,
        elevation=elevation,
        interact=False,
    )
    screenshot = plt.imread(f"./training_tmp/tmp_{rank}.png")
    os.remove(f"./training_tmp/tmp_{rank}.png")
    return screenshot

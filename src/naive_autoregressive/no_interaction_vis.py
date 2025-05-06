from cadquery import (
    Shape,
    Workplane,
    Assembly,
    Sketch,
    Vector,
)
from cadquery.occ_impl.assembly import toVTKAssy

from typing import Union, Any, List, Tuple, Iterable, cast, Optional

from OCP.TopoDS import TopoDS_Shape
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkMapper,
    vtkActor,
    vtkProp3D,
    vtkRenderWindow,
    vtkWindowToImageFilter,
    vtkRenderer,
)
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.util import numpy_support


DEFAULT_COLOR = (1, 0.8, 0)
DEFAULT_EDGE_COLOR = (0, 0, 0)
DEFAULT_PT_SIZE = 7.5
DEFAULT_PT_COLOR = "darkviolet"
DEFAULT_CTRL_PT_COLOR = "crimson"
DEFAULT_CTRL_PT_SIZE = 7.5

SPECULAR = 0.3
SPECULAR_POWER = 100
SPECULAR_COLOR = vtkNamedColors().GetColor3d("White")

ShapeLike = Union[Shape, Workplane, Assembly, Sketch, TopoDS_Shape]
Showable = Union[
    ShapeLike, List[ShapeLike], Vector, List[Vector], vtkProp3D, List[vtkProp3D]
]

from cadquery.vis import _split_showables, _to_vtk_pts, _to_vtk_axs, _to_assy


def no_interact_show(
    *objs: Showable,
    scale: float = 0.2,
    alpha: float = 1,
    tolerance: float = 1e-3,
    edges: bool = False,
    specular: bool = True,
    title: str = "CQ viewer",
    screenshot: Optional[str] = None,
    interact: bool = True,
    zoom: float = 1.0,
    roll: float = -35,
    elevation: float = -45,
    position: Optional[Tuple[float, float, float]] = None,
    focus: Optional[Tuple[float, float, float]] = None,
    width: Union[int, float] = 0.5,
    height: Union[int, float] = 0.5,
    trihedron: bool = True,
    bgcolor: tuple[float, float, float] = (1, 1, 1),
    gradient: bool = True,
    xpos: Union[int, float] = 0,
    ypos: Union[int, float] = 0,
):
    """
    Show CQ objects using VTK. This functions optionally allows to make screenshots.
    """

    # split objects
    shapes, vecs, locs, props = _split_showables(objs)

    # construct the assy
    assy = _to_assy(*shapes, alpha=alpha)

    # construct the points and locs
    pts = _to_vtk_pts(vecs)
    axs = _to_vtk_axs(locs, scale=scale)

    # assy+renderer
    renderer = vtkRenderer()
    renderer.AddActor(toVTKAssy(assy, tolerance=tolerance))

    # VTK window boilerplate
    win = vtkRenderWindow()

    # Render off-screen when not interacting
    win.SetOffScreenRendering(1)

    win.SetWindowName(title)
    win.AddRenderer(renderer)

    # get renderer and actor
    for act in cast(Iterable[vtkActor], renderer.GetActors()):

        propt = act.GetProperty()

        if edges:
            propt.EdgeVisibilityOn()

        if specular:
            propt.SetSpecular(SPECULAR)
            propt.SetSpecularPower(SPECULAR_POWER)
            propt.SetSpecularColor(SPECULAR_COLOR)

    # rendering related settings
    vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1, 0)
    vtkMapper.SetResolveCoincidentTopologyLineOffsetParameters(-1, 0)

    # construct an axes indicator
    axes = vtkAxesActor()
    axes.SetDragable(0)

    tp = axes.GetXAxisCaptionActor2D().GetCaptionTextProperty()
    tp.SetColor(0, 0, 0)

    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tp)
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tp)

    # use gradient background
    renderer.SetBackground(*bgcolor)

    if gradient:
        renderer.GradientBackgroundOn()

    # use FXXAA
    renderer.UseFXAAOn()

    # add pts and locs
    renderer.AddActor(pts)
    renderer.AddActor(axs)

    # add other vtk actors
    for p in props:
        renderer.AddActor(p)

    # set camera
    camera = renderer.GetActiveCamera()
    camera.Roll(roll)
    camera.Elevation(elevation)
    camera.Zoom(zoom)

    if position or focus:
        if position:
            camera.SetPosition(*position)
        if focus:
            camera.SetFocalPoint(*focus)
    else:
        renderer.ResetCamera()

    # show and return
    win.Render()

    # make a screenshot
    win2image = vtkWindowToImageFilter()
    win2image.SetInput(win)
    win2image.SetInputBufferTypeToRGB()
    win2image.ReadFrontBufferOff()
    win2image.Update()
    image_data = win2image.GetOutput()
    image = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
    image_arr = image.reshape(image_data.GetDimensions()[:-1] + (3,))

    # clean up
    win2image.SetInput(None)

    renderer.RemoveActor(pts)
    renderer.RemoveActor(axs)
    for p in props:
        renderer.RemoveActor(p)
    renderer.RemoveAllViewProps()

    renderer.SetRenderWindow(None)

    win.RemoveRenderer(renderer)
    win.Finalize()

    del win, axes, camera
    del (renderer, pts, axs, props)
    del win2image
    return image_arr

import vtk
import vtk
from vtk.util import numpy_support
import tempfile

import stl
from dshin import io_utils
import sys
import os
import io
import numpy as np
import hashlib
import vtk
import functools
from PIL import Image
from os import path
import numpy.linalg as la
from dshin import transforms
from matplotlib import cm


def vtk_write_png(filename, renderer, wh=(400, 300)):
    filename = path.expanduser(filename)
    width, height = wh

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    return filename


def vtk_image_from_renderer(renderer, wh=(400, 300)):
    width, height = wh

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = numpy_support.vtk_to_numpy(writer.GetResult())
    return Image.open(io.BytesIO(data.tobytes()))


def loadStl(fname):
    # https://gist.github.com/Hodapp87/8874941
    """Load the given STL file, and return a vtkPolyData object for it."""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(fname)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def edgeActor(polydata):
    edges = vtk.vtkExtractEdges()
    edges.SetInputData(polydata)
    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputData(edges.GetOutput())

    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(1, 0.5, 0)

    return edge_actor


def meshToActor(fv):
    filename = io_utils.temp_filename(prefix='vtk_render_tmp_', suffix='.stl')
    io_utils.save_stl(fv, filename)
    polydata = loadStl(filename)
    return polyDataToActor(polydata)


def color_triangles_by_z(polydata):
    Colors = vtk.vtkUnsignedCharArray()

    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")

    faces = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    verts = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData()).reshape(-1, 3)
    triangles = verts[faces]
    zvalues = np.mean(triangles, axis=1, keepdims=False)[:, 2]
    cmin = np.min(zvalues)
    cmax = np.max(zvalues)
    cvalues = np.round((zvalues - cmin) / (cmax - cmin + 1e-10) * 255.0).astype(np.int)
    assert (cvalues >= 0).all()
    assert (cvalues <= 255).all()
    for i, c in enumerate(cvalues):
        rgb = cm.jet(c)[:3]
        rgb = tuple(int(255 * value) for value in rgb)
        Colors.InsertNextTupleValue(rgb)

    polydata.GetCellData().SetScalars(Colors)
    return polydata


def renderMesh(fv, wh=(500, 500), campos=(1, 1, 1), edge_width=None, is_campos_spherical=False, zoom=1.0, up_axis=(0, 0, 1), object_pos=(0, 0, 0), view_angle=60, background=(1, 1, 1)):
    campos = np.array(campos, dtype=np.float64)

    if is_campos_spherical:
        is_campos_spherical = False
        campos = transforms.sph_to_xyz(campos, is_input_radians=False)

    object_pos = np.array(object_pos, dtype=np.float64)
    up_axis = np.array(up_axis, dtype=np.float64)

    filename = io_utils.temp_filename(prefix='vtk_render_tmp_', suffix='.stl')
    io_utils.save_stl(fv, filename)
    polydata = loadStl(filename)
    polydata = color_triangles_by_z(polydata)

    mesh = polyDataToActor(polydata, edge_width=edge_width)
    edge = edgeActor(polydata)
    vtk.vtkPolyDataMapper().SetResolveCoincidentTopologyToPolygonOffset()

    normalize = lambda x: x / la.norm(x)

    eye_v = object_pos - campos
    right_v = normalize(np.cross(eye_v, up_axis))
    up_v = normalize(np.cross(right_v, eye_v))

    camera = vtk.vtkCamera()
    camera.SetPosition(campos)

    camera.SetViewUp(up_v)
    camera.SetFocalPoint(object_pos)
    camera.SetViewAngle(view_angle)
    # camera.SetClippingRange(0.01, 10)
    camera.Zoom(zoom)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(background)
    renderer.AddActor(mesh)
    renderer.AddActor(edge)

    def add_light(position, intensity):
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetColor(1.0, 1.0, 1.0)
        light.SetIntensity(intensity)
        light.SetPositional(False)
        light.SetPosition(position)
        renderer.AddLight(light)

    # add_light((0.0, 0.0, 2 * la.norm(campos)), 1.0)
    add_light(campos, 1.0)

    renderer.SetActiveCamera(camera)
    # renderer.SetAmbient((1.0, 1.0, 1.0))

    out = vtk_image_from_renderer(renderer, wh=wh)

    os.remove(filename)

    return out


def polyDataToActor(polydata, edge_width=None, lighting_on=True):
    """Wrap the provided vtkPolyData object in a mapper and an actor, returning
    the actor."""
    # https://gist.github.com/Hodapp87/8874941


    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        # mapper.SetInput(reader.GetOutput())
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # actor.GetProperty().SetRepresentationToWireframe()
    # actor.GetProperty().SetRepresentationToPoints()
    actor.GetProperty().SetColor(0.5, 0.5, 1.0)
    actor.GetProperty().ShadingOff()

    if lighting_on:
        actor.GetProperty().LightingOn()
    else:
        actor.GetProperty().LightingOff()

    actor.GetProperty().SetInterpolationToFlat()
    actor.GetProperty().SetSpecular(0.0)
    actor.GetProperty().SetSpecularPower(0.0)
    actor.GetProperty().SetDiffuse(0.9)
    actor.GetProperty().SetAmbient(0.1)

    if edge_width is not None:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetLineWidth(edge_width)
    return actor


def showInteractiveActor(actor):
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    if isinstance(actor, (list, tuple)):
        for a in actor:
            ren.AddActor(a)
    else:
        ren.AddActor(actor)

    ren.SetBackground(0.1, 0.1, 0.1)
    iren.Initialize()
    renWin.Render()
    iren.Start()

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


def renderMesh(fv, wh=(500, 500)):
    filename = io_utils.temp_filename(prefix='vtk_render_tmp_', suffix='.stl')
    io_utils.save_stl(fv, filename)
    polydata = loadStl(filename)

    mesh = polyDataToActor(polydata)
    edge = edgeActor(polydata)
    vtk.vtkPolyDataMapper().SetResolveCoincidentTopologyToPolygonOffset()

    camera = vtk.vtkCamera()
    camera.SetPosition([0, -2, 3])
    camera.SetFocalPoint([0, 0, 0])

    VtkRenderer = vtk.vtkRenderer()
    VtkRenderer.SetBackground(1.0, 1.0, 1.0)
    VtkRenderer.AddActor(mesh)
    VtkRenderer.AddActor(edge)
    VtkRenderer.SetActiveCamera(camera)

    out = vtk_image_from_renderer(VtkRenderer, wh=wh)

    os.remove(filename)

    return out


def polyDataToActor(polydata):
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
    actor.GetProperty().SetColor(0.5, 0.5, 1.0)
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

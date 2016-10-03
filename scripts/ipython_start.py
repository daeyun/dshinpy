"""
%run -i ipython_start.py {page_width}
"""
import sys
import collections

from IPython import display as IPy_display


def set_page_width(width='1700', full=False):
    if full:
        width = '100%'
    else:
        if not width.endswith('%'):
            width = '{}px'.format(width)
    IPy_display.display(IPy_display.HTML(
        "<style>.container {{ width:{width} !important; }}</style>".format(width=width)))


if len(sys.argv) > 1:
    set_page_width(sys.argv[1])
else:
    set_page_width()

from os import path


def __extend_sys_path(prepend_paths=(), append_paths=()):
    pathset = set(sys.path)
    for p in prepend_paths:
        fullpath = path.realpath(path.expanduser(p))
        if fullpath not in pathset:
            sys.path.insert(0, fullpath)
    for p in append_paths:
        fullpath = path.realpath(path.expanduser(p))
        if fullpath not in pathset:
            sys.path.append(fullpath)
    sys.path = [p for p in sys.path if p]


__extend_sys_path(
    prepend_paths=[],
    append_paths=['~/Dropbox/git/dshinpy/', '~/Dropbox/git/multiview_shape/'],
)

import os
import multiprocessing as mp
import re
import threading
import time
import abc
import math
import typing
import functools
import contextlib
import array
import struct
import itertools

import psutil
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt
import tensorflow as tf

from IPython import get_ipython

ipython = get_ipython()

ipython.magic('autoreload 2')
# ipython.magic('aimport dshin')
# ipython.magic('aimport multiview_shape')


# TODO(daeyun): this is temporary.
ipython.run_cell_magic('javascript', '', """
require(['codemirror/keymap/vim'], function() {
    // enable the 'Ctrl-C' mapping
    // change the code mirror configuration
    var cm_config = require("notebook/js/cell").Cell.options_default.cm_config;
    delete cm_config.extraKeys['Ctrl-C'];
    // change settings for existing cells
    Jupyter.notebook.get_cells().map(function(cell) {
        var cm = cell.code_mirror;
        if (cm) {
            delete cm.getOption('extraKeys')['Ctrl-C'];
        }
    });
    // map the keys
    CodeMirror.Vim.map("<C-c>", "<Esc>", "insert");
});
""")

from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic, needs_local_scope)

from dshin import log
from dshin import nn
from dshin import geom2d
from dshin import geom3d
from dshin import transforms


@register_cell_magic
@needs_local_scope
def tf_init(self, cell):
    global sess, g, sess_conf

    tf.reset_default_graph()
    # conf = nn.utils.default_sess_config()
    try:
        if 'sess' in globals():
            sess.close()
    except:
        pass

    sess_conf = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    g = tf.get_default_graph()
    sess = tf.InteractiveSession(graph=g, config=sess_conf)

    ip = get_ipython()
    ip.run_cell(cell)

    filename = nn.graph_utils.save_graph_text_summary(g, random_dirname=True, verbose=False)

    IPy_display.display(IPy_display.HTML("""
    <span style="font-size:80%">{0}</span>
    """.format(filename)))

    # sess.run(tf.initialize_all_variables())
    # sess.run(tf.initialize_local_variables())
    # tf.train.start_queue_runners(sess)


from IPython import Application


@register_line_magic
def restart(line):
    app = Application.instance()
    app.kernel.do_shutdown(True)

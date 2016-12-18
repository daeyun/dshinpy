import libtmux
from os import path
import functools
from dshin import log
import psutil
import typing


def _reload(tmux_object: typing.Union[libtmux.Pane, libtmux.Window]):
    if isinstance(tmux_object, libtmux.Pane):
        tmux_object.server._update_panes()
    elif isinstance(tmux_object, libtmux.Window):
        tmux_object.server._update_windows()
    return tmux_object


def _extract_id(id_or_id_string, prefix='$'):
    """
    If `id_or_id_string` is a string, convert it to integer. If it has a non-numeric prefix, assert that it matches `prefix`.

    prefix can be:
        % pane
        @ window
        $ session
    """
    assert prefix in ('$', '%', '@')
    object_id = None
    if isinstance(id_or_id_string, int):
        return id_or_id_string
    elif isinstance(id_or_id_string, str):
        if id_or_id_string.startswith('%'):
            id_string = id_or_id_string[1:]
            assert id_string.isdigit()
            object_id = int(id_string)
        else:
            if not id_or_id_string[0].isdigit():
                raise ValueError('Incorrect tmux id string {}. Expected prefix {}.'.format(id_or_id_string, prefix))
            if not id_or_id_string.isdigit():
                raise ValueError('tmux id string is not a number: {}'.format(id_or_id_string))
            object_id = int(id_or_id_string)
    else:
        raise ValueError(id_or_id_string)
    assert object_id is not None
    assert isinstance(object_id, int)
    return object_id


def _find_session(server, session_name_or_id):
    session_name = None
    session_id = None
    if isinstance(session_name_or_id, str):
        if session_name_or_id.startswith('$'):
            session_id_string = session_name_or_id[1:]
            assert session_name.isdigit(session_id_string)
            session_id = int(session_id_string)
        else:
            # Could be a name or id.
            session_name = session_name_or_id
            if session_name.isdigit():
                session_id = int(session_name_or_id)
    elif isinstance(session_name_or_id, int):
        session_id = session_name_or_id
    else:
        raise ValueError('Unrecognized type for `session_name_or_id` {}'.format(type(session_name_or_id)))

    session = None
    try:
        if session is None and session_name is not None:
            session = server.find_where({'session_name': session_name})
        if session is None and session_id is not None:
            session = server.find_where({'session_id': session_id})
        assert session is not None
    except Exception as ex:
        log.exception('Could not find session.')
        raise ValueError('Unrecognized `session_name_or_id`: {}'.format(session_name_or_id))

    assert isinstance(session, libtmux.Session)
    return session


class TmuxSession(object):
    def __init__(self, session_name_or_id):
        self.server = libtmux.Server()
        self.session = _find_session(self.server, session_name_or_id)  # type: libtmux.Session
        self._processes = {}

    def find_pane_by_id(self, pane_id, window=None):
        _reload(pane_id)
        pane = None
        if isinstance(pane_id, libtmux.Pane):
            pane = pane_id
        else:
            pane = self._find_pane_by_id(pane_id, window=window)
        # Calling this will assert pane exists, in case the object is outdated.
        pane.get('pane_id')
        return pane

    @functools.lru_cache(maxsize=1024)
    def _find_pane_by_id(self, pane_id, window=None):
        pane_id = _extract_id(pane_id, '%')
        if window is not None:
            w = self.find_window_by_id(window)
            assert w is not None
            pane = w.find_where({'pane_id': '%{}'.format(pane_id)})
            if pane is None:
                raise ValueError('pane not found in window {}: {}'.format(window, pane_id))
            return pane
        pane = None
        self.session.list_windows()
        for w in self.session.list_windows():
            pane = w.find_where({'pane_id': '%{}'.format(pane_id)})
            if pane is not None:
                break
        if pane is None:
            raise ValueError('pane not found: {}'.format(pane_id))
        assert isinstance(pane, libtmux.Pane)
        return pane

    def find_window_by_id(self, window_id):
        _reload(window_id)
        if isinstance(window_id, libtmux.Pane):
            return window_id
        return self._find_window_by_id(window_id)

    @functools.lru_cache(maxsize=1024)
    def _find_window_by_id(self, window_id):
        if isinstance(window_id, libtmux.Pane):
            return window_id
        window_id = _extract_id(window_id, '@')
        window = self.session.find_where({'window_id': '@{}'.format(window_id)})
        assert window is not None, window_id
        assert isinstance(window, libtmux.Window)
        return window

    def get_pane_process(self, pane, window=None):
        pane = self.find_pane_by_id(pane, window=window)
        if pane['pane_id'] not in self._processes:
            self._processes[pane['pane_id']] = psutil.Process(int(pane['pane_pid']))
        return self._processes[pane['pane_id']]

    def is_pane_busy(self, pane, window=None, ignore_child_processes=True):
        pane = self.find_pane_by_id(pane, window=window)
        proc = self.get_pane_process(pane, window=window)

        # e.g. "zsh"
        if proc.name() == pane['pane_current_command']:
            if ignore_child_processes or len(proc.children()) == 0:
                return False
        return True

    def pane_working_directory(self, pane, window=None):
        proc = self.get_pane_process(pane, window=window)
        return proc.cwd()

    def run_command_in_pane(self, command, pane, window=None, assert_not_busy=True, suppress_history=True):
        assert isinstance(command, str)
        is_busy = self.is_pane_busy(pane, window=window, ignore_child_processes=True)
        if assert_not_busy and is_busy:
            raise RuntimeError('Pane {} is busy.'.format(pane))
        p = self.find_pane_by_id(pane, window=window)
        if not is_busy:
            # In case line is not empty.
            p.send_keys('C-c', enter=False, suppress_history=False)
        p.send_keys(command, enter=True, suppress_history=suppress_history)

    def pane_change_directory(self, dirpath, pane, window=None, assert_not_busy=True):
        dirpath = path.expanduser(dirpath)
        assert path.isdir(dirpath), dirpath
        self.run_command_in_pane('cd {}'.format(dirpath), pane=pane, window=window, assert_not_busy=assert_not_busy, suppress_history=True)

    def split_window_if_pane_not_exists(self, pane, pane_window=None, target_window=None, attach=False, vertical=False, max_width=None, min_width=None):
        try:
            pane = self.find_pane_by_id(pane_id=pane, window=pane_window)
        except (ValueError, AssertionError, IndexError):
            if target_window is None:
                target_window = self.session.attached_window
            else:
                target_window = self.find_window_by_id(window_id=target_window)
            pane = target_window.split_window(attach=attach, vertical=vertical)
            if max_width is not None:
                if int(pane['pane_width']) > max_width:
                    pane.set_width(width=max_width)
        assert isinstance(pane, libtmux.Pane)
        if min_width is not None:
            if int(pane['pane_width']) < min_width:
                pane.set_width(width=min_width)
        return pane

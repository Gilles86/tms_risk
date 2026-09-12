"""Drop the XLA flag that autoflatten sets but the cluster's jaxlib (0.11) rejects.

autoflatten's pyflatten backend calls ``configure_threading(n)`` whenever a core
count is split between hemispheres, which appends
``--xla_cpu_multi_thread_eigen_thread_count=n`` to XLA_FLAGS. jaxlib 0.11 no
longer knows that flag and aborts the worker process (``F... Unknown flag in
XLA_FLAGS``), which surfaces only as ``BrokenProcessPool`` after the projection
step has already succeeded.

Put this directory on PYTHONPATH (``autoflatten.sh`` does): Python imports
``sitecustomize`` at start-up in every process, the pool's workers included, so
the patched function is what pyflatten imports.
"""
try:
    import os

    import autoflatten.flatten.threading as _threading

    _configure = _threading.configure_threading

    def configure_threading(n_threads=None):
        _configure(n_threads)
        flags = [f for f in os.environ.get('XLA_FLAGS', '').split()
                 if not f.startswith('--xla_cpu_multi_thread_eigen_thread_count')]
        os.environ['XLA_FLAGS'] = ' '.join(flags)

    _threading.configure_threading = configure_threading
except Exception:                                   # never break the interpreter
    pass

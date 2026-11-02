"""Optional process-wide native thread budget, applied before workers start.

No environment settings means no change to existing water-only deployments.
Import model libraries first because their initialization can change OpenCV's
thread setting. CPU backends may subsequently choose their own torch budget;
the measured CPU-water/GPU-lamp profile uses 8 for both libraries.
"""
import os
from importlib import import_module


def configure_native_threads(environ=None):
    environ=os.environ if environ is None else environ
    settings={}
    for key in ('BARRIGUARD_OPENCV_THREADS','BARRIGUARD_TORCH_THREADS'):
        raw=environ.get(key)
        if raw is None:continue
        if not str(raw).isdigit() or int(raw)<1:raise ValueError(key+' must be a positive integer')
        settings[key]=int(raw)
    if not settings:return None
    import_module('ultralytics')
    cv=import_module('cv2');torch=import_module('torch')
    if 'BARRIGUARD_OPENCV_THREADS' in settings:cv.setNumThreads(settings['BARRIGUARD_OPENCV_THREADS'])
    if 'BARRIGUARD_TORCH_THREADS' in settings:
        target=settings['BARRIGUARD_TORCH_THREADS']
        if torch.get_num_threads()!=target:torch.set_num_threads(target)
    return dict(opencv_threads=cv.getNumThreads(),torch_threads=torch.get_num_threads())

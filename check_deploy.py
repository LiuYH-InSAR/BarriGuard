"""Check a Git checkout without starting RTSP readers, HTTP or callbacks."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'deploy/app'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'output/model_settings'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/model_settings/matplotlib'))


def verify_files(manifest, result):
    for item in manifest['files']:
        path = (ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            result['errors'].append('Missing/outside checkout: '+item['path'])
            continue
        data = path.read_bytes()
        if item['hash_mode'] == 'text_lf':
            data = data.replace(b'\r\n',b'\n')
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            key = 'warnings' if item.get('configurable') else 'errors'
            result[key].append('Changed file: '+item['path'])


def check_runtime(config, load_model, result):
    from server.config import load_runtime
    from night_lamp.lamp_checks import LampChecks
    from water_barrier.yolo_params import PARAMS
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_runtime(str(config))
    weight = PARAMS.inference.weights.resolve()
    if not weight.is_relative_to(ROOT) or not weight.is_file():
        raise ValueError('V6 weight is missing or outside checkout')
    info = json.loads((weight.parent/'model_info.json').read_text(encoding='utf-8'))
    if hashlib.sha256(weight.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError('V6 model checksum mismatch')
    cameras = []
    for camera in cfg.cameras:
        water = camera.algorithms.get('water_gap')
        lamp = camera.algorithms.get('night_lamp')
        if water is not None and water.spec.backend != 'yolo':
            raise ValueError(f'{camera.id}: water backend is not yolo')
        regions = None
        if lamp is not None:
            if not lamp.spec.lamp_checks.enabled:
                raise ValueError(f'{camera.id}: fixed lamp polygons are disabled')
            for path in (lamp.spec.lamp_checks.polygons,lamp.spec.lamp_roi.get('calibration')):
                if path and not Path(path).resolve().is_relative_to(ROOT):
                    raise ValueError('Lamp calibration refers outside checkout')
            regions = len(LampChecks(lamp.spec.lamp_checks).polys)
        cameras.append(dict(id=camera.id,water_backend=water.spec.backend if water else None,
                            lamp_polygons=regions))
    result.update(cameras=cameras,model=weight.relative_to(ROOT).as_posix(),
        model_sha256=info['sha256'],device=PARAMS.inference.device,
        callback_enabled=cfg.callback.enabled,debug_enabled=PARAMS.debug.enabled,
        ffmpeg_available=bool(shutil.which('ffmpeg')),model_loaded=False)
    if not result['ffmpeg_available']:
        result['warnings'].append('FFmpeg not found; real HEVC/RTSP decoding is unverified. Docker provides the decoder.')
    if load_model:
        for key in ('YOLO_CONFIG_DIR','MPLCONFIGDIR'):
            Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
        import numpy as np
        from water_barrier.yolo_detector import Detector
        detector = Detector(PARAMS)
        pixels = np.zeros((PARAMS.inference.imgsz,PARAMS.inference.imgsz,3),np.uint8)
        detector.detect(pixels)
        result['model_loaded'] = True
    result.update(rtsp_started=False,callback_sent=False,http_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'deploy/config.yaml')
    parser.add_argument('--load-model',action='store_true')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    result = dict(errors=[],warnings=[])
    try:
        manifest = json.loads((ROOT/'deploy_manifest.json').read_text(encoding='utf-8'))
        verify_files(manifest,result)
        if not result['errors']:
            check_runtime(args.config,args.load_model,result)
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['passed'] = not result['errors']
    text = json.dumps(result,ensure_ascii=False,indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(text,encoding='utf-8')
    print(text)
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())

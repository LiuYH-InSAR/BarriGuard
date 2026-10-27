"""仅管理本算法私有目录。日志和旧项目目录不参与清理。"""
import json
import re
from pathlib import Path

import cv2 as cv
import numpy as np
from loguru import logger


def serial(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,dict):
        return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [serial(v) for v in value]
    return value


class Store:
    def __init__(self, params, camera):
        self._p = params
        camera = re.sub(r'[^0-9A-Za-z_.-]','_',str(camera)).strip('._') or 'camera'
        self.root = (Path(params.storage.root)/camera).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self._counter = 0
        self._last_cleanup = None

    def _path(self, group, stamp, suffix):
        self._counter += 1
        folder = self.root/group
        folder.mkdir(parents=True,exist_ok=True)
        return folder/f'{stamp:.6f}_{self._counter}{suffix}'

    def record(self, group, stamp, data, image=None):
        path = self._path(group,stamp,'.json')
        if image is not None:
            image_path = path.with_suffix('.jpg')
            if not cv.imwrite(str(image_path),image,[cv.IMWRITE_JPEG_QUALITY,self._p.storage.jpeg_quality]):
                raise OSError(f'Image write failed: {image_path}')
            data = dict(data,image=str(image_path.relative_to(self.root)))
        self._atomic(path,data)
        return str(path.relative_to(self.root))

    def snapshot(self, stamp, data):
        self._atomic(self.root/'current.json',dict(saved_at=stamp,**data))

    def load(self, stamp):
        path = self.root/'current.json'
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not 0 <= stamp-data['saved_at'] < self._p.storage.retention_s:
                return None
            return data
        except (OSError,ValueError,KeyError):
            logger.warning('[water_yolo] unusable checkpoint {}',path)
            return None

    def read_image(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Reference path escapes algorithm storage')
        return cv.imread(str(path)) if path.is_file() else None

    def read_record(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Record path escapes algorithm storage')
        return json.loads(path.read_text(encoding='utf-8'))

    def cleanup(self, stamp):
        if self._last_cleanup is not None and stamp-self._last_cleanup < self._p.storage.cleanup_s:
            return
        self._last_cleanup = stamp
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.is_symlink():
                continue
            for path in folder.iterdir():
                if not path.is_file() or path.is_symlink():
                    continue
                # 文件名写入生成时间，不以最后读取时间延长寿命。
                try:
                    created = float(path.stem.split('_')[0])
                except ValueError:
                    continue
                if stamp-created >= self._p.storage.retention_s:
                    try:
                        path.unlink()
                    except OSError:
                        logger.exception('[water_yolo] cleanup failed {}',path)

    @staticmethod
    def _atomic(path, data):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(serial(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(path)

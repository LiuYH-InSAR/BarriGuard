"""Atomic current state and self-contained seven-day archives.

Owns only current/ and archive/ beneath its configured root. Runtime state is
never subject to archive retention. Reference PNG bytes travel with snapshots.
"""
from pathlib import Path
import json
import os
import re
import shutil
import uuid

RETENTION_S = 7*24*60*60


def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(value)):
        raise ValueError('Unsafe camera identifier')
    return str(value)


class LampArchive:
    def __init__(self,root,camera_id):
        self.root=Path(root).resolve()/safe_id(camera_id)
        self.root.mkdir(parents=True,exist_ok=True)

    def _write(self,base,state,images):
        base.mkdir(parents=True,exist_ok=True)
        for name,data in images.items():
            if not re.fullmatch(r'[A-Za-z0-9_-]+\.png',name):
                raise ValueError('References must be local PNG basenames')
            (base/name).write_bytes(data)
        (base/'state.json').write_text(json.dumps(state,ensure_ascii=False,allow_nan=False),encoding='utf-8')

    def save_current(self,state,images=None):
        current=self.root/'current';current.mkdir(exist_ok=True)
        # Publish a generation then atomically swap the pointer. Old generations
        # remain recoverable if the process dies before pointer replacement.
        generation=uuid.uuid4().hex
        folder=current/generation
        self._write(folder,state,images or {})
        temporary=current/(generation+'.tmp')
        temporary.write_text(generation,encoding='ascii')
        os.replace(temporary,current/'HEAD')
        for old in current.iterdir():
            if old.is_dir() and not old.is_symlink() and old.name!=generation:
                shutil.rmtree(old)

    def load_current(self):
        head=self.root/'current'/'HEAD'
        if not head.exists():return None
        generation=head.read_text(encoding='ascii')
        if not re.fullmatch(r'[0-9a-f]{32}',generation):raise ValueError('Invalid generation')
        return json.loads((head.parent/generation/'state.json').read_text(encoding='utf-8'))

    def archive(self,state,stamp,images=None):
        directory=self.root/'archive';directory.mkdir(exist_ok=True)
        temporary=directory/('pending_'+uuid.uuid4().hex)
        self._write(temporary,dict(state,archived_at=stamp),images or {})
        destination=directory/(f'{stamp:.6f}_'+uuid.uuid4().hex)
        os.replace(temporary,destination)
        return destination

    def sweep(self,stamp):
        archive=self.root/'archive'
        if not archive.exists():return 0
        removed=0
        for folder in archive.iterdir():
            if not folder.is_dir() or folder.is_symlink():continue
            try:created=float(folder.name.split('_')[0])
            except ValueError:continue
            if stamp-created>=RETENTION_S:
                shutil.rmtree(folder);removed+=1
        return removed

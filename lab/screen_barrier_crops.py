"""Hypothesis: per-block appearance outliers help locate unusable crop samples.

Read-only screening; scores nominate images for visual review, never deletion.
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def inspect(path):
    im = cv2.imread(str(path))
    row = dict(path=str(path), camera=path.parent.parent.name, block=path.parent.name)
    if im is None:
        return row | dict(error='unreadable')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(im, (24, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return row | dict(error='', width=im.shape[1], height=im.shape[0],
        mean=float(gray.mean()), std=float(gray.std()),
        dark=float((gray < 15).mean()), bright=float((gray > 245).mean()),
        sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        saturation=float(hsv[:, :, 1].mean()), feature=small.reshape(-1).astype(float))


def sheet(rows, path):
    cell_w, cell_h, cols = 230, 170, 6
    canvas = np.full((((len(rows)+cols-1)//cols)*cell_h, cols*cell_w, 3), 35, np.uint8)
    for i, row in enumerate(rows):
        im = cv2.imread(row['path'])
        x, y = (i % cols)*cell_w, (i//cols)*cell_h
        if im is not None:
            scale = min((cell_w-8)/im.shape[1], (cell_h-42)/im.shape[0])
            thumb = cv2.resize(im, (max(1, round(im.shape[1]*scale)), max(1, round(im.shape[0]*scale))))
            canvas[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
        text = f'{row["id"]} {row["camera"]}/{row["block"]}'
        cv2.putText(canvas, text, (x+3,y+cell_h-24), cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        cv2.putText(canvas, Path(row['path']).stem[-28:], (x+3,y+cell_h-8), cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
    if not cv2.imwrite(str(path),canvas):
        raise OSError(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    files=sorted(args.images.rglob('*.jpg'))
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,row in enumerate(pool.map(inspect,files)):
            row['id']=i+1
            rows.append(row)
            if (i+1)%5000==0:
                print(f'Scanned {i+1}/{len(files)}',flush=True)
    groups={}
    for row in rows:
        groups.setdefault((row['camera'],row['block']),[]).append(row)
    candidates, overview=[],[]
    for group in groups.values():
        valid=[r for r in group if not r['error']]
        if not valid:
            continue
        # Brightness strata keep ordinary night/day differences from dominating.
        centers={}
        for bucket in {int(r['mean']//40) for r in valid}:
            peers=[p for p in valid if int(p['mean']//40)==bucket]
            if len(peers)<5:
                peers=valid
            features=np.stack([p['feature'] for p in peers])
            centers[bucket]=np.median(features,axis=0)
        for r in valid:
            distance=float(np.abs(r['feature']-centers[int(r['mean']//40)]).mean())
            r['distance']=distance
            r['score']=distance + 25*(r['std']<6) + 25*(r['dark']>.95) + 25*(r['bright']>.9)
        ranked=sorted(valid,key=lambda r:r['score'],reverse=True)
        candidates.extend(ranked[:4])
        ordered=sorted(valid,key=lambda r:r['mean'])
        overview.extend(ordered[round((len(ordered)-1)*q)] for q in [0,.25,.5,.75,1])
    for kind,selected in [('candidates',candidates),('overview',overview)]:
        dest=args.out/kind;dest.mkdir()
        for start in range(0,len(selected),60):
            sheet(selected[start:start+60],dest/f'{start//60+1:03d}.jpg')
    fields=['id','path','camera','block','error','width','height','mean','std','dark','bright','sharp','saturation','distance','score']
    with (args.out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (args.out/'review_index.json').write_text(json.dumps({str(r['id']):{k:v for k,v in r.items() if k!='feature'} for r in rows},indent=2),encoding='utf-8')
    summary=dict(images=len(rows),unreadable=sum(bool(r['error']) for r in rows),blocks=len(groups),
                 tiny=sum(not r['error'] and min(r['width'],r['height'])<16 for r in rows),
                 candidates=len(candidates),overview=len(overview))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    main()

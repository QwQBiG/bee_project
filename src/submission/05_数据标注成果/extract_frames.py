"""Reproduce annotated crops from official videos; no frames are shipped here.

Requires Python 3.11+, numpy and opencv-python. Example:
python extract_frames.py --video-root D:/official_videos --output D:/reproduced
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes declared root: {relative}')
    return path


def jpeg(frame, quality):
    import cv2
    ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return data


def check_annotations(manifest_path):
    manifest_path = Path(manifest_path)
    package = manifest_path.resolve().parent
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = document['samples']
    labels = [r['label_file'] for r in records]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError('empty or duplicate annotation entries')
    if set(labels) != {p.relative_to(package).as_posix() for p in (package / 'annotations').rglob('*.txt')}:
        raise ValueError('annotation files do not match manifest')
    instances = 0
    for r in records:
        path = contained(package, r['label_file'])
        if sha256(path) != r['label_sha256']:
            raise ValueError(f'annotation checksum mismatch: {r["sample"]}')
        rows = path.read_text(encoding='utf-8').splitlines()
        if len(rows) != r['accepted_rows']:
            raise ValueError('annotation row count mismatch')
        for row in rows:
            values = list(map(float, row.split()))
            if len(values) != 14 or not all(map(math.isfinite, values)) or values[0] != 0:
                raise ValueError('invalid annotation fields or class')
            if not all(0 <= values[i] <= 1 for i in (1,2,3,4,5,6,8,9,11,12)):
                raise ValueError('annotation coordinate is outside [0, 1]')
            if min(values[3:5]) <= 0 or not all(values[i] in (0,1,2) for i in (7,10,13)):
                raise ValueError('invalid box dimensions or keypoint visibility')
            if any(c-s/2 < -1e-6 or c+s/2 > 1+1e-6 for c,s in ((values[1],values[3]),(values[2],values[4]))):
                raise ValueError('annotation box extends outside crop')
        instances += len(rows)
    groups = [(package / f'splits/{name}.txt').read_text(encoding='utf-8').splitlines()
              for name in ('train', 'val')]
    if any(len(items) != len(set(items)) for items in groups):
        raise ValueError('duplicate split entries')
    train, val = map(set, groups)
    if train & val or train | val != set(labels):
        raise ValueError('split entries must be disjoint and cover all labels')
    if any(r['label_file'] not in (train if r['split'] == 'train' else val) for r in records):
        raise ValueError('split assignment does not match manifest')
    if {r['sequence'] for r in records if r['label_file'] in train} & {r['sequence'] for r in records if r['label_file'] in val}:
        raise ValueError('source video occurs in both splits')
    if len(records) != document['counts']['images'] or instances != document['counts']['instances']:
        raise ValueError('manifest totals do not match labels')
    return document


def extract(manifest_path, video_root, output):
    import cv2
    manifest_path, video_root, output = map(Path, (manifest_path, video_root, output))
    document = check_annotations(manifest_path)
    records = document['samples']
    if not records:
        raise ValueError('manifest contains no annotated samples')
    package = manifest_path.resolve().parent
    if output.resolve().is_relative_to(package):
        raise ValueError('output must be outside the annotation submission directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output must be empty; existing files are never overwritten')
    names = [r['image_file'] for r in records]
    if len(set(names)) != len(names):
        raise ValueError('duplicate output image names')
    videos = {v['sequence']: v for v in document['videos']}
    sources = {}
    for sequence in sorted({r['sequence'] for r in records}):
        v = videos[sequence]
        path = contained(video_root, v['video_file'])
        if not path.is_file() or sha256(path) != v['sha256']:
            raise ValueError(f'missing or mismatched official video: {sequence}')
        sources[sequence] = path
    for r in records:
        contained(output, r['image_file'])
        label = contained(package, r['label_file'])
        if not label.is_file() or sha256(label) != r['label_sha256']:
            raise ValueError(f'annotation checksum mismatch: {r["sample"]}')
        index = r['frame_index_zero_based']
        if type(index) is not int or index < 0 or r['frame_id_one_based'] != index+1:
            raise ValueError('invalid frame index or frame numbering')
    generated, checks = [], []
    # Reproduce the recorded two JPEG stages: full frame, then cropped frame.
    for sequence, path in sources.items():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f'cannot open video: {sequence}')
        try:
            for r in [item for item in records if item['sequence'] == sequence]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, r['frame_index_zero_based'])
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'cannot decode frame: {r["sample"]}')
                v = videos[sequence]
                if frame.shape[:2] != (v['height'], v['width']):
                    raise ValueError('decoded dimensions do not match manifest')
                full_jpeg = jpeg(frame, document['jpeg_quality'])
                frame = cv2.imdecode(full_jpeg, cv2.IMREAD_COLOR)
                x, y, w, h = (r['crop'][k] for k in ('x','y','width','height'))
                if not (0 <= x < x+w <= frame.shape[1] and 0 <= y < y+h <= frame.shape[0]):
                    raise ValueError('crop extends outside source frame')
                encoded = jpeg(frame[y:y+h, x:x+w], document['jpeg_quality'])
                digest = hashlib.sha256(encoded.tobytes()).hexdigest()
                checks.append(dict(sample=r['sample'], image_sha256=digest,
                                   exact_reference_match=digest == r['image_sha256']))
                generated.append((contained(output, r['image_file']), encoded))
        finally:
            capture.release()
    # Decode and validate every sample before writing the output batch.
    for path, data in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data.tobytes())
    report = dict(images=len(generated), opencv=cv2.__version__, checks=checks,
                  note='JPEG bytes may vary between codec builds; positions and crop geometry are fixed.')
    with (output / 'reproduction_report.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    parser.add_argument('--video-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true', help='validate annotation files without videos or optional dependencies')
    args = parser.parse_args()
    if not args.check_only and (args.video_root is None or args.output is None):
        parser.error('--video-root and --output are required for frame extraction')
    try:
        if args.check_only:
            document = check_annotations(args.manifest)
            print(json.dumps(dict(status='ok', **document['counts'])))
            return
        report = extract(args.manifest, args.video_root, args.output)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Extraction failed: {error}\n')
    print(json.dumps(dict(images=report['images'], output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()

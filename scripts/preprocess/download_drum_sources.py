"""採用した3データセットを取得・検証し、data/raw に展開する。"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

SOURCES = {
    'lakh': {
        'version': '0.1', 'license': 'CC-BY-4.0',
        'homepage': 'https://colinraffel.com/projects/lmd/',
        'url': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz',
        'filename': 'lmd_full.tar.gz', 'algorithm': 'sha256',
        'checksum': '6fcfe2ac49ca08f3f214cec86ab138d4fc4dabcd7f27f491a838dae6db45a12b',
        'checksum_source': 'https://muspy.readthedocs.io/en/stable/_modules/muspy/datasets/lmd.html',
        'citation': 'Colin Raffel. Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching. PhD thesis, 2016.',
    },
    'groove': {
        'version': '1.0.0', 'license': 'CC-BY-4.0',
        'homepage': 'https://magenta.withgoogle.com/datasets/groove',
        'url': 'https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip',
        'filename': 'groove-v1.0.0-midionly.zip', 'algorithm': 'sha256',
        'checksum': '651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e',
        'citation': 'Jon Gillick et al. Learning to Groove with Inverse Sequence Transformations. ICML, 2019.',
    },
    'groove2groove': {
        'version': '1.0.0', 'license': 'CC-BY-NC-4.0',
        'homepage': 'https://groove2groove.telecom-paris.fr/',
        'url': 'https://zenodo.org/records/3958000/files/groove2groove-data-v1.0.0.tar.gz?download=1',
        'filename': 'groove2groove-data-v1.0.0.tar.gz', 'algorithm': 'md5',
        'checksum': 'c407de7b3676267660c88dc6ee351c79',
        'citation': 'Ondrej Cifka et al. Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data. TASLP, 2020. doi:10.1109/TASLP.2020.3019642',
    },
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=4096)
def resolved_directory(root, parent):
    root = Path(root).resolve()
    directory = (root / parent).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f'アーカイブ外へのパス: {parent}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_target(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'アーカイブ外へのパス: {name}')
    target = resolved_directory(str(root), str(relative.parent)) / relative.name
    if target.is_symlink():
        raise ValueError(f'出力先がシンボリックリンク: {target}')
    return target


def unpack(archive, root):
    # 通常ファイルのみを扱い、所有者・パーミッション・リンクを持ち込まない。
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                dest = safe_target(root, item.filename)
                with z.open(item) as src, dest.open('wb') as out:
                    shutil.copyfileobj(src, out)
    else:
        # NASでは小ファイルの逐次作成が律速になるため、書き込みだけを並列化。
        # 展開キューを64件に制限し、アーカイブ全体をメモリに載せない。
        def save_member(name, payload):
            safe_target(root, name).write_bytes(payload)

        pending = deque()
        with ThreadPoolExecutor(max_workers=16) as pool, tarfile.open(archive, 'r|gz') as tar:
            for item in tar:
                if not item.isfile():
                    continue
                pending.append(pool.submit(save_member, item.name, tar.extractfile(item).read()))
                if len(pending) >= 64:
                    pending.popleft().result()
            for future in pending:
                future.result()


def acquire(dataset, data_root):
    spec = SOURCES[dataset]
    root = data_root / 'raw' / dataset
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec['filename']
    if not archive.exists():
        part = archive.with_name(archive.name + '.part')
        # 接続切断時は取得済みの範囲から再開。検証後だけ正式名に変更する。
        for attempt in range(50):
            result = subprocess.run(['curl', '-fL', '--silent', '--show-error', '-C', '-',
                                     '--connect-timeout', '20', '--max-time', '1800',
                                     '--speed-time', '15', '--speed-limit', '131072',
                                     spec['url'], '-o', str(part)])
            if result.returncode == 0:
                break
        else:
            raise RuntimeError(f'{dataset}: ダウンロード失敗。{part} から再開できます')
        if digest(part, spec['algorithm']) != spec['checksum']:
            raise ValueError(f'{dataset}: チェックサム不一致: {part}')
        part.rename(archive)
    actual = digest(archive, spec['algorithm'])
    if actual != spec['checksum']:
        raise ValueError(f'{dataset}: チェックサム不一致: {archive}')
    marker = root / 'acquisition.json'
    if not marker.exists():
        print(f'{dataset}: 検証済み。展開中', flush=True)
        unpack(archive, root)
        marker.write_text(json.dumps(dict(spec, archive_bytes=archive.stat().st_size,
                                         archive_sha256=digest(archive)), ensure_ascii=False, indent=2) + '\n')
    if dataset == 'lakh':
        names = root / 'md5_to_paths.json'
        if not names.exists():
            tmp = names.with_suffix('.json.part')
            subprocess.run(['curl', '-fLsS', '--retry', '3',
                            'https://colinraffel.com/projects/lmd/md5_to_paths.json',
                            '-o', str(tmp)], check=True)
            json.loads(tmp.read_text())
            tmp.rename(names)
    print(f'{dataset}: 完了 {root}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--datasets', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    args = p.parse_args()
    for name in args.datasets:
        acquire(name, args.data_root)


if __name__ == '__main__':
    main()

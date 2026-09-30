"""再開用バッチ・学習用シャード・来歴の入出力。"""
import gzip
import hashlib
import json
from pathlib import Path
import pickle


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def write_pickle(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(gzip.compress(pickle.dumps(value,protocol=4),compresslevel=1,mtime=0))
    temp.replace(path)


def read_pickle(path):
    return pickle.loads(gzip.decompress(Path(path).read_bytes()))


def iter_pairs(root,split='train'):
    """信頼できるローカル生成物のシャードを順に読み込む。"""
    root=Path(root)
    index=json.loads((root/'index.json').read_text())
    for row in index['shards']:
        if row['split']==split:
            yield from read_pickle(root/row['path'])

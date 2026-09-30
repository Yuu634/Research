"""図の8段階に従い、小節間の難→易ペアを再開可能なバッチ処理で生成する。"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sqlite3
import time
import zlib

import numpy as np

from data.pairdata.core import Config,VERSION,make_record,mine
from data.pairdata.io import file_hash,read_pickle,write_json,write_pickle
from src.data.drum_difficulty_v2 import ATTRS,decode_bars,discretize

SPLITS=('itest','test','val','train')
WORKER={}


def source_records(meta,limit=None):
    splits={}
    for split in SPLITS:
        files=pickle.loads((meta/f'{split}_split.pkl').read_bytes())
        for filename in files:
            if Path(filename).name!=filename or not filename.endswith('.pkl') or filename[:-4] in splits:
                raise ValueError('元分割のファイル名または重複が不正')
            splits[filename[:-4]]=split
    rows=[]; seen=set(); counts=Counter()
    for line in (meta/'manifest.jsonl').open():
        r=json.loads(line)
        if r['status']!='ok':continue
        if r['id'] in seen or splits.get(r['id'])!=r['split']:
            raise ValueError('元manifestと分割が不一致')
        seen.add(r['id'])
        group=r['dataset'],r['split']
        counts[group]+=1
        if limit is None or counts[group]<=limit:
            r.pop('flags',None)
            rows.append(r)
    if seen!=set(splits):raise ValueError('元分割ID不足')
    return sorted(rows,key=lambda r:(SPLITS.index(r['split']),r['dataset'],r['id']))


def initialize(input_dir,meta,work,config):
    vocab=pickle.loads((Path(meta)/'drum_vocab.pkl').read_bytes())
    WORKER.update(input=Path(input_dir),meta=Path(meta),work=Path(work),config=Config(**config).validate(),
                  vocab=vocab['idx2token'],eos=vocab['token2idx']['<EOS>'],
                  bounds=json.loads((Path(meta)/'bounds.json').read_text())['boundaries'],
                  db=sqlite3.connect(f'file:{(Path(meta)/"scores.sqlite").resolve()}?mode=ro',uri=True))


def mine_batch(task):
    name,records=task
    path=WORKER['work']/f'{name}.pkl.gz'
    receipt=path.with_suffix('.json')
    ids=[r['id'] for r in records]
    if path.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['ids']!=ids or saved['sha256']!=file_hash(path):
            raise ValueError('再開バッチの整合性不一致')
        return saved
    rows=[]; stats=Counter(); per_piece=[]
    for source in records:
        content=(WORKER['input']/(source['id']+'.pkl')).read_bytes()
        if hashlib.sha256(content).hexdigest()!=source['output_sha256']:
            raise ValueError('入力PKLのチェックサム不一致: '+source['id'])
        starts,tokens,labels=pickle.loads(content)
        bars=decode_bars(starts,tokens,WORKER['vocab'])
        saved=WORKER['db'].execute('SELECT info,payload FROM scores WHERE id=?',(source['id'],)).fetchone()
        info=json.loads(saved[0])
        raw,_=pickle.loads(zlib.decompress(saved[1]))
        classes=np.array([[r[a] for a in ATTRS] for r in labels],dtype=np.int16)
        if info['status']!='ok' or len(bars)!=source['bars'] or not np.array_equal(classes,discretize(raw,WORKER['bounds'])):
            raise ValueError('元ラベル・生スコア不一致')
        pairs,summary=mine(bars,raw,classes,WORKER['bounds'],WORKER['config'])
        rows.extend(make_record(source,bars,starts,tokens,raw,classes,p,WORKER['eos'],WORKER['config']) for p in pairs)
        stats.update(summary)
        per_piece.append(dict(id=source['id'],dataset=source['dataset'],split=source['split'],**summary))
    write_pickle(path,dict(rows=rows,pieces=per_piece))
    result=dict(name=name,ids=ids,sha256=file_hash(path),stats=dict(stats),pairs=len(rows))
    write_json(receipt,result)
    return result


def batches(records,size):
    groups=defaultdict(list)
    for r in records:groups[r['split'],r['dataset']].append(r)
    tasks=[]
    for (split,dataset),rows in groups.items():
        for start in range(0,len(rows),size):
            tasks.append((f'{split}_{dataset}_{start//size:05d}',rows[start:start+size]))
    return tasks


def package(args,tasks,records,config):
    """holdout優先。現在小節の内容が分割をまたがないよう、ペア単位で除外する。"""
    root=args.output
    shards=[]; pending=[]; current=None; seen_pairs=set(); pattern_owner={}
    aggregate=Counter(); rejected=Counter(); counts=Counter(); dataset_counts=Counter(); source_ids=set()
    pieces=[]; score_sums=Counter(); delta_hist=defaultdict(Counter)
    def flush():
        nonlocal pending
        if not pending:return
        name=f'shards/{current}-{sum(s["split"]==current for s in shards):05d}.pkl.gz'
        write_pickle(root/name,pending)
        shards.append(dict(path=name,split=current,pairs=len(pending),sha256=file_hash(root/name)))
        pending=[]
    for name,rs in tasks:
        path=args.work/f'{name}.pkl.gz'
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['ids']!=[r['id'] for r in rs] or receipt['sha256']!=file_hash(path):
            raise ValueError('バッチの内容不一致')
        batch=read_pickle(path)
        aggregate.update(receipt['stats']); pieces.extend(batch['pieces'])
        for row in batch['rows']:
            split=row['source']['split']
            patterns=row['before_pattern'],row['after_pattern']
            if row['pair_id'] in seen_pairs:
                rejected[split+'/exact_pair_duplicate']+=1
                continue
            if any(p in pattern_owner and pattern_owner[p]!=split for p in patterns):
                rejected[split+'/cross_split_bar_overlap']+=1
                continue
            if split!=current:flush(); current=split
            seen_pairs.add(row['pair_id'])
            for p in patterns:pattern_owner[p]=split
            pending.append(row); source_ids.add(row['source']['id'])
            counts[split]+=1; dataset_counts[row['source']['dataset']+'/'+split]+=1
            for k in ('q','difficulty_distance'):score_sums[k]+=row[k]
            for a in ATTRS[:5]:
                delta_hist[a][row['difficulty_after'][a]-row['difficulty_before'][a]]+=1
            if len(pending)>=args.shard_size:flush()
    flush()
    expected={s['path'] for s in shards}
    actual={str(p.relative_to(root)) for p in (root/'shards').glob('*.pkl.gz')}
    if actual!=expected:raise ValueError('余分なシャード。出力先を確認してください')
    total=sum(counts.values())
    if total==0:raise ValueError('採用ペア0件。診断を確認し、閾値を無断で緩めないでください')
    for name in ('drum_vocab.pkl','bounds.json'):
        shutil.copyfile(args.source_meta/name,root/name)
    with (root/'sources.jsonl').open('w') as f:
        for source in records:
            if source['id'] in source_ids:
                f.write(json.dumps(source,ensure_ascii=False)+'\n')
    with (root/'piece_audit.jsonl').open('w') as f:
        for p in pieces:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    write_json(root/'index.json',dict(version=VERSION,shards=shards))
    write_json(root/'summary.json',dict(version=VERSION,input_pieces=len(records),
        source_pieces_with_output=len(source_ids),pairs=total,split_counts={s:counts[s] for s in SPLITS},
        dataset_split_counts=dict(dataset_counts),mining=dict(aggregate),packaging_rejections=dict(rejected),
        mean_scores={k:v/total for k,v in score_sums.items()},class_delta_histograms=dict(delta_hist),
        current_bar_patterns=len(pattern_owner),config=asdict(config),
        context_note='区間名は不明。局所32小節・4小節周期位置・Fill代理・前後小節を用いる。',
        confidence_note='qはルールの類似度で、正しい編曲である確率ではない。',
        leakage_policy='元の曲分割を保持し、現在小節の完全一致をテンポ非依存で分割間から除外。直前・直後文脈と近似重複は別途注意。'))
    print('シャード生成完了:',dict(counts),'合計',total,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/processed/drum_collection_difficulty_v2'))
    p.add_argument('--source-meta',type=Path,default=Path('data/metadata/drum_collection_difficulty_v2'))
    p.add_argument('--output',type=Path,default=Path('data/processed/drum_pairdata_v1'))
    p.add_argument('--work',type=Path,default=Path('data/work/drum_pairdata_v1'))
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--shard-size',type=int,default=4096)
    p.add_argument('--limit-per-group',type=int)
    p.add_argument('--config',type=Path,help='Config各項目のJSON。指定しない項目は既定値。')
    p.add_argument('--stage',choices=['mine','package','validate','all'],default='all')
    args=p.parse_args()
    if min(args.workers,args.batch_size,args.shard_size)<1 or (args.limit_per_group is not None and args.limit_per_group<1):
        p.error('件数・並列数は正数')
    paths=[args.input.resolve(),args.source_meta.resolve(),args.output.resolve(),args.work.resolve()]
    if len(set(paths))!=4:p.error('入力・出力・作業領域は分離してください')
    config=Config(**(json.loads(args.config.read_text()) if args.config else {})).validate()
    records=source_records(args.source_meta,args.limit_per_group)
    args.output.mkdir(parents=True,exist_ok=True); args.work.mkdir(parents=True,exist_ok=True)
    source_identity=json.loads((args.source_meta/'identity.json').read_text())
    if source_identity['formula_sha256']!=file_hash('src/data/drum_difficulty_v2.py'):
        raise ValueError('ラベル算出時とv2式が異なる')
    identity=dict(version=VERSION,config=asdict(config),input=str(args.input.resolve()),
        source_meta=str(args.source_meta.resolve()),output=str(args.output.resolve()),work=str(args.work.resolve()),
        limit_per_group=args.limit_per_group,batch_size=args.batch_size,shard_size=args.shard_size,
        source_hashes={n:file_hash(args.source_meta/n) for n in
                       ('manifest.jsonl','identity.json','bounds.json','drum_vocab.pkl','scores.sqlite',
                        'train_split.pkl','val_split.pkl','test_split.pkl','itest_split.pkl')},
        code_hashes={str(f):file_hash(f) for f in sorted(Path('data/pairdata').glob('*.py'))},
        formula_sha256=source_identity['formula_sha256'],attributes=list(ATTRS[:5]),rhythm_reference_only=True)
    for root in (args.output,args.work):
        path=root/'identity.json'
        if path.exists() and json.loads(path.read_text())!=identity:
            raise ValueError('コード・入力・設定が変わったため別の出力先／作業領域が必要')
        write_json(path,identity)
    tasks=batches(records,args.batch_size)
    if args.stage in ('mine','all'):
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                initargs=(args.input,args.source_meta,args.work,asdict(config))) as pool:
            for n,result in enumerate(pool.map(mine_batch,tasks),1):
                if n%10==0 or n==len(tasks):
                    print(f'候補抽出 {n}/{len(tasks)}バッチ ({time.monotonic()-start:.0f}秒)',flush=True)
    if args.stage in ('package','all'):package(args,tasks,records,config)
    if args.stage in ('validate','all'):
        from data.pairdata.validate import run_validation
        run_validation(args.output,args.workers)


if __name__=='__main__':main()

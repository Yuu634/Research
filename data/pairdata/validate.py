"""全ペアを原譜へ照合し、類似度・再接続後の難易度・分割漏洩を検証する。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import pickle
import time

import numpy as np

from data.pairdata.core import Config,assess,bar_tokens,canonical_hash,eligible,features,pattern_hash
from data.pairdata.io import file_hash,read_pickle,write_json
from src.data.drum_difficulty_v2 import ATTRS,decode_bars,discretize,score_bar

STATE={}


def initialize(root):
    root=Path(root)
    identity=json.loads((root/'identity.json').read_text())
    vocab=pickle.loads((root/'drum_vocab.pkl').read_bytes())
    sources={r['id']:r for r in map(json.loads,(root/'sources.jsonl').read_text().splitlines())}
    STATE.update(root=root,input=Path(identity['input']),config=Config(**identity['config']),sources=sources,
                 vocab=vocab['idx2token'],eos=vocab['token2idx']['<EOS>'],
                 bounds=json.loads((root/'bounds.json').read_text())['boundaries'])
    piece.cache_clear()


@lru_cache(maxsize=8)
def piece(key):
    path=STATE['input']/(key+'.pkl')
    if file_hash(path)!=STATE['sources'][key]['output_sha256']:
        raise ValueError('原譜が変更された: '+key)
    starts,tokens,labels=pickle.loads(path.read_bytes())
    bars=decode_bars(starts,tokens,STATE['vocab'])
    return starts,tokens,bars,np.array([[d[a] for a in ATTRS] for d in labels])


def validate_record(row):
    source=row['source']; key=source['id']; i=source['before_bar']; j=source['after_bar']
    saved=STATE['sources'][key]
    for name in ('dataset','split'):
        if source[name]!=saved[name]:raise ValueError('出典・分割変更')
    if source['sha256']!=saved['output_sha256'] or source['license']!=saved['source_metadata']['license']:
        raise ValueError('来歴不一致')
    if source['original_sha256']!=saved['source_metadata']['source_sha256']:
        raise ValueError('原データハッシュ不一致')
    starts,tokens,bars,classes=piece(key)
    if i==j or not 0<=i<len(bars) or not 0<=j<len(bars):raise ValueError('小節参照が不正')
    def excerpt(index):
        return bar_tokens(starts,tokens,index,STATE['eos']) if 0<=index<len(bars) else None
    if row['X_before']!=excerpt(i) or row['X_after']!=excerpt(j) or row['bar_positions']!=[0]:
        raise ValueError('ペアの音符列が原譜と異なる')
    context=row['context']
    for name,index in [('previous',i-1),('following',i+1),('original_after_previous',j-1),('original_after_following',j+1)]:
        if context[name]!=excerpt(index):raise ValueError('前後文脈不一致')
    config=STATE['config']
    if context['section'] is not None or context['section_source']!='unavailable':raise ValueError('区間名を推測していないこと')
    if context['local_block']!=i//config.local_bars or context['phrase_phase_proxy']!=i%config.phrase_bars:
        raise ValueError('文脈代理の位置不一致')
    if context['bpm']!=bars[i].bpm or context['after_difficulty_context']!='before_previous':
        raise ValueError('再接続文脈不一致')
    if context['fill_proxy']!=features(bars[i])[2]:raise ValueError('Fill代理情報不一致')
    raw=np.zeros((len(bars),6))
    for n in (i,j):
        raw[n]=score_bar(bars[n],bars[n-1] if n else None)[0]
        if eligible(bars[n],bars[n-1] if n else None,raw[n],config):raise ValueError('品質除外対象')
    np.testing.assert_allclose(raw[i],row['raw_before'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(raw[j],row['original_after_raw'],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(discretize(raw[[i,j]],STATE['bounds']),classes[[i,j]])
    result,reason=assess(bars,i,j,raw,classes,STATE['bounds'],config)
    if result is None:raise ValueError('採用条件の再検証失敗: '+str(reason))
    np.testing.assert_allclose(row['raw_after'],result['after_raw'],rtol=0,atol=1e-10)
    for k in ('rhythm','skeleton','context'):
        if not np.isclose(row['similarities'][k],result[k],atol=1e-12):raise ValueError('類似度不一致')
    if not np.isclose(row['q'],result['q'],atol=1e-12):raise ValueError('信頼度不一致')
    if not np.isclose(row['difficulty_distance'],result['distance'],atol=1e-12):raise ValueError('難易度距離不一致')
    for field,values in [('difficulty_before',classes[i]),('original_after_classes',classes[j]),('difficulty_after',result['after_classes'])]:
        if row[field]!=dict(zip(ATTRS,map(int,values))):raise ValueError('クラス不一致')
    before=np.array([row['difficulty_before'][a] for a in ATTRS[:5]],dtype=float)/7
    after=np.array([row['difficulty_after'][a] for a in ATTRS[:5]],dtype=float)/7
    np.testing.assert_allclose(row['d_before'],before,rtol=0,atol=1e-12)
    np.testing.assert_allclose(row['d_after'],after,rtol=0,atol=1e-12)
    np.testing.assert_allclose(row['delta_d'],after-before,rtol=0,atol=1e-12)
    if not np.all(after<=before) or not np.linalg.norm(after-before)>config.difficulty_threshold:
        raise ValueError('5属性の単調減少条件違反')
    bh,ah=pattern_hash(bars[i]),pattern_hash(bars[j])
    if (bh,ah)!=(row['before_pattern'],row['after_pattern']) or row['pair_id']!=canonical_hash([bh,ah]):
        raise ValueError('内容署名不一致')
    return row['pair_id'],bh,ah,key


def validate_shard(entry):
    path=STATE['root']/entry['path']
    if file_hash(path)!=entry['sha256']:raise ValueError('シャードSHA-256不一致')
    rows=read_pickle(path)
    if len(rows)!=entry['pairs']:raise ValueError('シャード件数不一致')
    result=[]
    for row in rows:
        if row['source']['split']!=entry['split']:raise ValueError('シャード分割不一致')
        result.append(validate_record(row))
    return entry['split'],result


def run_validation(root,workers=16):
    root=Path(root)
    identity=json.loads((root/'identity.json').read_text())
    for path,expected in identity['code_hashes'].items():
        if file_hash(path)!=expected:raise ValueError('実行時からコード変更')
    if file_hash('src/data/drum_difficulty_v2.py')!=identity['formula_sha256']:raise ValueError('v2式変更')
    for name in ('bounds.json','drum_vocab.pkl'):
        if file_hash(root/name)!=identity['source_hashes'][name]:raise ValueError('語彙・境界変更')
    index=json.loads((root/'index.json').read_text())
    entries=index['shards']
    if {r['path'] for r in entries}!={str(p.relative_to(root)) for p in (root/'shards').glob('*.pkl.gz')}:
        raise ValueError('シャード一覧不一致')
    if len(entries)!=len({r['path'] for r in entries}):raise ValueError('索引の重複シャード')
    counts=Counter(); pair_ids=set(); patterns={}; sources={}; start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(root,)) as pool:
        for n,(split,rows) in enumerate(pool.map(validate_shard,entries),1):
            for pid,bh,ah,key in rows:
                if pid in pair_ids:raise ValueError('完全一致ペア重複')
                pair_ids.add(pid)
                for pattern in (bh,ah):
                    if pattern in patterns and patterns[pattern]!=split:raise ValueError('分割間の小節内容重複')
                    patterns[pattern]=split
                if key in sources and sources[key]!=split:raise ValueError('分割間の曲重複')
                sources[key]=split
            counts[split]+=len(rows)
            print(f'全件検証 {n}/{len(entries)}シャード・{sum(counts.values())}ペア ({time.monotonic()-start:.0f}秒)',flush=True)
    summary=json.loads((root/'summary.json').read_text())
    if sum(counts.values())!=summary['pairs'] or any(counts[s]!=c for s,c in summary['split_counts'].items()):
        raise ValueError('集計件数不一致')
    write_json(root/'validation.json',dict(passed=True,pairs=sum(counts.values()),split_counts=dict(counts),
        every_pair_source_tokens_checked=True,every_pair_similarity_recomputed=True,
        every_pair_original_and_reconnected_difficulty_recomputed=True,
        every_attribute_nonincreasing=True,unique_pairs=True,current_bar_content_split_disjoint=True,
        source_piece_split_disjoint=True,source_pieces=len(sources),current_bar_patterns=len(patterns),
        shard_checksums_checked=True,index_sha256=file_hash(root/'index.json'),
        limitation='人手の音楽的評価・近似重複・前後文脈の分割間共有・他トラックとの整合性は未検証。'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/processed/drum_pairdata_v1'))
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    if args.workers<1:p.error('workersは正数')
    run_validation(args.output,args.workers)


if __name__=='__main__':main()

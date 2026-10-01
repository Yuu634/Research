"""ステップ2の固定人工例で、探索スコアの内訳を確認する。モデル選定は行わない。"""
import argparse
from dataclasses import asdict,replace
import hashlib
import html
import json
from pathlib import Path
import pickle

from scripts.evaluate.review_arrangement_rules import svg
from src.arrangement.examples import make_bar
from src.arrangement.rules import apply_rule
from src.arrangement.safety import default_guards
from src.arrangement.scoring import DIFFICULTY_KEYS,ScoreConfig,ScoringModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta',type=Path,default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--output',type=Path,default=Path('output/beamsearch_step2'))
    args=parser.parse_args()
    bounds=json.loads((args.meta/'bounds.json').read_text())['boundaries']
    vocab=pickle.loads((args.meta/'drum_vocab.pkl').read_bytes())['token2idx']
    model=ScoringModel(bounds,vocab)
    original=make_bar([(p,'hh_closed') for p in range(0,96,6)]+[(p,'kick') for p in range(0,96,24)]+
                      [(24,'snare'),(72,'snare','Accent'),(18,'snare','Ghost'),(66,'snare','Ghost')])
    guards=default_guards(original)
    first=apply_rule(original,'hh_16_to_8',original=original,guards=guards)
    second=apply_rule(first.after,'remove_ghost_notes',original=original,guards=guards)
    ghost=apply_rule(original,'remove_ghost_notes',original=original,guards=guards)
    if first.status!='applied' or second.status!='applied' or ghost.status!='applied':
        raise ValueError('確認例のルールが適用されていない')
    target=model.difficulty(second.after).normalized
    cases=[('無編集',original,0),('HH16分→8分',first.after,1),('Ghost削除',ghost.after,1),
           ('HH間引き＋Ghost削除',second.after,2),('同じ譜面へ冗長な3ルールで到達した場合',second.after,3),
           ('保護Kickを削除した不適格候補',original.with_notes(n for n in original.notes if n.instrument!='kick'),1),
           ('BPMを変えた不適格候補',replace(original,bpm=180),1)]
    rows=[]; sections=[]
    for label,candidate,rule_count in cases:
        result=model.score(original,candidate,target,rule_count=rule_count,guards=guards)
        row=dict(label=label,score=asdict(result),notes=[asdict(n) for n in candidate.notes])
        rows.append(row)
        content=html.escape(json.dumps(asdict(result),ensure_ascii=False,indent=2,allow_nan=False))
        sections.append(f'<article><h2>{html.escape(label)}</h2>{svg(candidate)}<pre>{content}</pre></article>')
    report=dict(config=asdict(model.config),attribute_order=list(DIFFICULTY_KEYS),target=target,
        target_note='固定人工例で2ルールを適用した譜面の難易度。学習・重み調整・モデル選定には使用しない。',
        bounds_sha256=hashlib.sha256((args.meta/'bounds.json').read_bytes()).hexdigest(),
        formula_sha256=hashlib.sha256(Path('src/data/drum_difficulty_v2.py').read_bytes()).hexdigest(),
        scoring_sha256=hashlib.sha256(Path('src/arrangement/scoring.py').read_bytes()).hexdigest(),cases=rows)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    header='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>探索スコア：ステップ2</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto;padding:10px;background:#f5f7fa}article{background:white;padding:20px;margin:25px 0;border:1px solid #ddd}svg{width:100%;max-height:260px}pre{white-space:pre-wrap;font-size:13px}</style>
<h1>探索スコア：ステップ2</h1><p>J = −1.0 L_difficulty + 0.5 S_skeleton − 0.02 N_rules − 0.1 L_edit</p>
<p>高い値を優先。valid=falseの候補は総合点にかかわらず不採用。係数は確認用の初期値で、最適化済みではありません。</p>'''
    (args.output/'index.html').write_text(header+'<h2>入力譜面</h2>'+svg(original)+
        '<p>目標（属性名付き）: '+html.escape(str(target))+'</p>'+''.join(sections)+'</html>')
    print(json.dumps([dict(label=r['label'],valid=r['score']['valid'],total=r['score']['total']) for r in rows],ensure_ascii=False,indent=2))


if __name__=='__main__':main()

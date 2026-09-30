"""Zenodoの40刺激で、承認済みLZ反復性ラベルを外部診断する。式は調整しない。"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.data.drum_difficulty_v2 import Hit, rhythm


def correlation(x, y, samples):
    rho = float(spearmanr(x,y).statistic)
    tau = float(kendalltau(x,y).statistic)
    bootstrap = [float(spearmanr(x[ix],y[ix]).statistic) for ix in samples]
    finite = np.array(bootstrap)[np.isfinite(bootstrap)]
    return dict(spearman=rho, kendall_tau_b=tau,
                spearman_ci95=np.quantile(finite,[.025,.975]).tolist(), valid_bootstraps=len(finite))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('output/revise_attribute_review/Data_240229.zip'))
    parser.add_argument('--output',type=Path,default=Path('output/revise_attribute_v2/zenodo'))
    parser.add_argument('--deps',type=Path,default=Path('output/revise_attribute_v2/deps'))
    parser.add_argument('--bootstrap',type=int,default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('bootstrapは正数')
    sys.path.insert(0,str(args.deps.resolve()))
    import rdata
    archive = args.archive.read_bytes()
    if hashlib.md5(archive).hexdigest() != 'b4be12a160658ba49486278237da2ce1':
        raise ValueError('指定されたZenodo v4と異なるアーカイブ')
    with zipfile.ZipFile(args.archive) as z:
        def read(name):
            path = next(p for p in z.namelist() if p.endswith('/'+name+'.Rda') and not p.startswith('__MACOSX'))
            return rdata.conversion.convert(rdata.parser.parse_data(z.read(path)))[name]
        events, stimuli = read('drumpatterns'), read('stimuli')
    rows = []
    families = {'BassDrum':'kick','SnareDrum':'snare','Cymbal':'hh_closed'}
    errors = []
    for _, row in stimuli.iterrows():
        sid = int(row['Stimulus'])
        subset = events[(events['Stimulus']==sid) & (events['Beats']>=0) & (events['Beats']<16)]
        variants = [[],[]]
        for rounded in (False,True):
            bars = [[] for _ in range(4)]
            for _, event in subset.iterrows():
                beat = float(event['Beats'])
                grid = int(round(beat*24)) if rounded else int(np.floor(beat*24+1e-9))
                if not rounded:
                    errors.append(abs(beat-grid/24))
                grid = min(grid,383)
                b,p = divmod(grid,96)
                bars[b].append(Hit(p,families[event['InstrumentFamily']]))
            variants[int(rounded)] = [rhythm(bar) for bar in bars]
        rows.append(dict(stimulus=sid, title=str(row['SongTitle']),
                         perceived=float(row['PerceivedComplexity']), rhythm=float(np.mean(variants[0])),
                         rhythm_round_sensitivity=float(np.mean(variants[1])), per_bar=variants[0],
                         evaluated_notes=len(subset), published_notes=int(row['NumberOfOnsets']),
                         syncopation=float(row['SyncopationIndex']),
                         revised_syncopation=float(row['RevisedSyncopationIndex']),
                         bdm=float(row['KolmogorovComplexity'])))
    rows.sort(key=lambda r:r['stimulus'])
    if [r['stimulus'] for r in rows] != list(range(1,41)):
        raise ValueError('40刺激のID不一致')
    y = np.array([r['perceived'] for r in rows])
    rng = np.random.default_rng(20260930)
    samples = rng.integers(0,len(rows),size=(args.bootstrap,len(rows)))
    results = {}
    for name in ['rhythm','rhythm_round_sensitivity','evaluated_notes','published_notes','syncopation','revised_syncopation','bdm']:
        x = np.array([r[name] for r in rows])
        results[name] = correlation(x,y,samples)
    lx = np.array([r['rhythm'] for r in rows])
    counts = np.array([r['evaluated_notes'] for r in rows])
    summary = dict(stimuli=40, rating='PerceivedComplexity (Bradley-Terry)',
                   interpretation='知覚的複雑さの診断。身体的な演奏難易度の正解ではない。',
                   primary_window='0 <= Beats < 16; 4小節平均',
                   primary_quantization='floor(Beats*24+1e-9); 24分割',
                   correlations=results, rhythm_vs_note_count_spearman=float(spearmanr(lx,counts).statistic),
                   max_quantization_error_beats=max(errors),
                   off_grid_events=sum(e>1e-7 for e in errors), evaluated_events=len(errors),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   formula_sha256=hashlib.sha256(Path('src/data/drum_difficulty_v2.py').read_bytes()).hexdigest(),
                   archive_sha256=hashlib.sha256(archive).hexdigest(), rdata_version=importlib.metadata.version('rdata'),
                   parameters_fitted_on_ratings=False, note='既存対照値は配布資料の区間定義。窓差に注意する。')
    args.output.mkdir(parents=True,exist_ok=True)
    for name, value in [('summary',summary),('stimulus_scores',rows)]:
        (args.output/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

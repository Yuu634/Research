"""ステップ1の人工例・変更差分・安全制約・試聴用MIDIを出力する。探索は実行しない。"""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import re

import mido

from src.arrangement.examples import examples, PARAMETERS
from src.arrangement.rules import REGISTRY, apply_rule
from src.arrangement.safety import Guards, edits

PITCH={'kick':36,'snare':38,'snare_rim':37,'hh_closed':42,'hh_open':46,'hh_pedal':44,
       'tom_high':50,'tom_mid':47,'tom_low':45,'floor':43,'crash':49,'ride_cymbal':51,
       'ride_bell':53,'splash':55,'china':52,'cowbell':56,'tambourine':54}


def midi(bar,path):
    score=mido.MidiFile(ticks_per_beat=480); track=mido.MidiTrack(); score.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(bar.bpm),time=0))
    track.append(mido.MetaMessage('time_signature',numerator=4,denominator=4,time=0))
    events=[]
    for n in bar.notes:
        pitch=PITCH[n.instrument]; velocity={'Ghost':10,'Normal':80,'Accent':110}[n.level]
        events.append((n.position*20,1,mido.Message('note_on',channel=9,note=pitch,velocity=velocity)))
        events.append((min(1920,n.position*20+10),0,mido.Message('note_off',channel=9,note=pitch,velocity=0)))
    last=0
    for time,_,message in sorted(events,key=lambda e:(e[0],e[1],e[2].note)):
        message.time=time-last; track.append(message); last=time
    track.append(mido.MetaMessage('end_of_track',time=1920-last)); score.save(path)


def svg(bar):
    instruments=sorted({n.instrument for n in bar.notes})
    height=40+24*max(1,len(instruments)); rows={inst:32+24*k for k,inst in enumerate(instruments)}
    out=[f'<svg viewBox="0 0 730 {height}" role="img" aria-label="打点位置図">']
    for p in range(0,97,6):
        x=140+p*6
        out.append(f'<path d="M{x} 15 V{height}" stroke="{"#778" if p%24==0 else "#ddd"}"/>')
    for inst,y in rows.items():out.append(f'<text x="2" y="{y+4}" font-size="12">{inst}</text>')
    for n in bar.notes:
        fill={'Ghost':'#bbb','Normal':'#176887','Accent':'#d14636'}[n.level]
        label=html.escape(f'{n.id}: {n.position} {n.token()}',quote=True)
        out.append(f'<circle cx="{140+n.position*6}" cy="{rows[n.instrument]}" r="5" fill="{fill}"><title>{label}</title></circle>')
    return ''.join(out)+ '</svg>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/beamsearch_step1'))
    parser.add_argument('--spec',type=Path,default=Path('drum_arrangement_rule_functions.txt'))
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'midi').mkdir(exist_ok=True)
    spec=args.spec.read_text(); names=re.findall(r'^([a-z][a-z0-9_]*)\(',spec,re.M)
    descriptions={m.group(1):m.group(2) for m in re.finditer(r'^([a-z][a-z0-9_]*)\([^\n]*\)\n([^\n]+)',spec,re.M)}
    relaxed=Guards(max_added=None,max_deleted=None,max_edit_ratio=None)
    fixtures=examples(); rows=[]; sections=[]
    for name in names:
        if name not in REGISTRY:continue
        trials=[(label,apply_rule(bar,name,guards=relaxed,**PARAMETERS.get(name,{}))) for label,bar in fixtures.items()]
        applicable=[t for t in trials if t[1].status=='applied' and sum(t[1].step_edits[k] for k in ('added','deleted','modified'))>0]
        if not applicable:raise ValueError('変更を確認できる人工例がない: '+name)
        label,result=max(applicable,key=lambda t:(t[1].step_edits['ratio'],t[0]))
        safe=apply_rule(result.before,name,**PARAMETERS.get(name,{}))
        row=dict(rule=name,description=descriptions[name],fixture=label,parameters=PARAMETERS.get(name,{}),
            isolated=asdict(result),default_guards=asdict(safe))
        rows.append(row)
        for suffix,bar in [('before',result.before),('isolated_after',result.after),('safe_after',safe.after)]:
            midi(bar,args.output/'midi'/f'{name}_{suffix}.mid')
        detail=html.escape(json.dumps(row,ensure_ascii=False,indent=2))
        sections.append(f'<article><h2>{name}</h2><p>{html.escape(descriptions[name])}</p>'
            f'<p>例: {label} ／ パラメータ: {html.escape(str(PARAMETERS.get(name,{})))}</p>'
            f'<h3>編集前</h3>{svg(result.before)}<h3>ルール単体（保護・編集量制限なし）</h3>{svg(result.after)}'
            f'<p>編集量: {html.escape(str(result.step_edits))}</p>'
            f'<h3>既定の保護・編集量制限を適用</h3><p>{safe.status}: {html.escape(str(safe.reason))}</p>{svg(safe.after)}'
            f'<p><a href="midi/{name}_before.mid">編集前MIDI</a> ／ '
            f'<a href="midi/{name}_isolated_after.mid">単体編集後MIDI</a> ／ '
            f'<a href="midi/{name}_safe_after.mid">制約付き編集後MIDI</a></p><details><summary>音符・ID・編集詳細</summary><pre>{detail}</pre></details></article>')
    constraints=[dict(name=n,description=descriptions[n]) for n in names if n not in REGISTRY]
    summary=dict(edit_rules=len(rows),constraints=len(constraints),all_edit_rules_have_changed_example=True,
        default_guard_status={s:sum(r['default_guards']['status']==s for r in rows) for s in ('applied','unchanged','rejected')},
        note='人工例での動作確認。難易度低下・音楽的品質・BeamSearch性能の検証ではない。')
    (args.output/'review.json').write_text(json.dumps(dict(summary=summary,rules=rows,constraints=constraints),ensure_ascii=False,indent=2)+'\n')
    head='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>編曲ルール：ステップ1確認</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto;padding:10px;background:#f5f7fa}article{background:white;padding:20px;margin:25px 0;border:1px solid #ddd}svg{width:100%;max-height:260px}pre{white-space:pre-wrap;font-size:12px}h2{overflow-wrap:anywhere}</style>
<h1>編曲ルール：ステップ1確認</h1><p>青は通常音、灰色はGhost、赤はAccent。横軸は1小節96位置、太線は拍頭。各点にカーソルを置くと音符情報を表示します。</p>
<p>「ルール単体」は編集内容を確認するため保護・編集量の制限を外した例です。実運用の既定制約付き結果と区別してください。</p>
<p>MIDIは打点と強弱を試聴する補助です。Rimshot・Choke等の奏法や、単一トークンに圧縮された装飾音の音響は忠実に再現しません。</p>'''
    tail='<h2>骨格・安全制約11項目</h2><ul>'+''.join(f'<li><code>{r["name"]}</code>: {html.escape(r["description"])}</li>' for r in constraints)+'</ul></html>'
    (args.output/'index.html').write_text(head+''.join(sections)+tail)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

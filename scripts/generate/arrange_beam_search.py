"""選定済みBeamSearch設定とステップ3のJSON入力で1小節を編曲する。"""
import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path

from scripts.evaluate.review_arrangement_inputs import read_request
from scripts.evaluate.review_arrangement_rules import midi, svg
from src.beamsearch.beam_search import beam_search
from src.common.evaluation import evaluate_result
from src.common.inputs import prepare_input
from src.common.representation import to_tokens
from src.beamsearch.selected_model import load_selected_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('config/beamsearch_selected.json'))
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--output', type=Path, default=Path('output/beamsearch_arrangement'))
    args = parser.parse_args()
    try:
        model, search, actions, evaluation, selected = load_selected_model(args.model, args.meta)
        payload = read_request(args.input)
        prepared = prepare_input(payload, model)
        result = beam_search(prepared, model, search, actions)
        row = evaluate_result(prepared, result.candidate, model, case_id='input', source_id='input',
                              rule_count=len(result.history), elapsed_seconds=result.elapsed_seconds, config=evaluation)
        report = dict(model=selected['candidate'], request=payload,
                      model_file_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      history=[action.as_dict() for action in result.history], score=asdict(result.score),
                      evaluation=row, statistics=result.statistics, candidate=asdict(result.candidate),
                      tokens=to_tokens(result.candidate, model.vocabulary),
                      midi_note='試聴用MIDIは奏法を専用音源のアーティキュレーションへ変換しない。強弱・位置・楽器を再生する。')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        midi(prepared.original, args.output/'before.mid')
        midi(result.candidate, args.output/'after.mid')
        (args.output/'index.html').write_text(
            '<!doctype html><html lang="ja"><meta charset="utf-8"><title>BeamSearch編曲</title>'
            '<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto}svg{width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>BeamSearchによる単一小節の編曲</h1><h2>入力</h2>'+svg(prepared.original)+
            '<h2>出力</h2>'+svg(result.candidate)+'<p><a href="before.mid">入力MIDI</a> / '
            '<a href="after.mid">出力MIDI</a></p><p>'+html.escape(report['midi_note'])+'</p>'
            '<pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))+'</pre></html>', encoding='utf-8')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f'{selected["candidate"]["name"]}: {len(result.history)}ルール適用、結果: {args.output}/index.html')


if __name__ == '__main__':
    main()

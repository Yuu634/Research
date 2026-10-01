"""学習済みの単体／ハイブリッドを読み、入力JSONから1小節を編曲する。"""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
from time import perf_counter
import torch

from data.pairdata.io import file_hash, write_json
from scripts.evaluate.review_arrangement_inputs import read_request
from scripts.evaluate.review_arrangement_rules import midi, svg
from src.arrangement.inputs import prepare_input
from src.beamsearch import beam_search
from src.beam_muse.data import Example, collate, load_metadata
from src.beam_muse.evaluation import evaluate_output
from src.beam_muse.experiment import load_checkpoint
from src.beam_muse.features import extract_skeleton
from src.beam_muse.generation import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--beam-model', type=Path, default=Path('config/beamsearch_selected.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    try:
        if args.threads < 1:
            raise ValueError('threadsは正整数')
        torch.set_num_threads(args.threads)
        scorer, search, actions, evaluation, _, vocabulary = load_metadata(args.meta, args.beam_model)
        model, checkpoint = load_checkpoint(args.checkpoint, vocabulary, device=args.device)
        for key, path in (('vocabulary', args.meta/'drum_vocab.pkl'), ('bounds', args.meta/'bounds.json'), ('beam_model', args.beam_model)):
            if checkpoint['identity'][key] != file_hash(path):
                raise ValueError('学習時のメタデータと不一致: '+key)
        prepared = prepare_input(read_request(args.input), scorer)
        variant = checkpoint['trial']['variant']
        started = perf_counter()
        intermediate, history, beam_seconds = prepared.original, [], 0.
        if variant == 'beam_muse':
            result = beam_search(prepared, scorer, search, actions)
            intermediate, history, beam_seconds = result.candidate, [a.as_dict() for a in result.history], result.elapsed_seconds
        example = Example('input', 'input', 'inference', prepared, intermediate, None, history, beam_seconds)
        batch = collate([example], vocabulary, scorer, variant, args.device, training=False)
        with torch.no_grad():
            generated = generate(model, batch, prepared, vocabulary)
        row = evaluate_output(prepared, generated.candidate, scorer, case_id='input', source_id='input',
                              elapsed_seconds=perf_counter()-started, rule_count=len(history),
                              failure_reason=generated.failure_reason, config=evaluation)
        report = dict(variant=variant, checkpoint_sha256=file_hash(args.checkpoint), smoke=checkpoint['identity']['smoke'],
                      model=checkpoint['trial'], beam_history=history, skeleton=extract_skeleton(prepared.original,
                      section=prepared.context.get('section')).report(), absolute_target=prepared.target,
                      intermediate_difficulty=scorer.difficulty(intermediate, prepared.previous).normalized,
                      tokens=generated.tokens, candidate=asdict(generated.candidate) if generated.candidate else None,
                      evaluation=row)
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'result.json', report)
        figures = []
        for name, bar in (('before', prepared.original), ('rule', intermediate), ('final', generated.candidate)):
            if bar is not None:
                midi(bar, args.output/f'{name}.mid')
                figures.append(f'<h2>{name}</h2>'+svg(bar)+f'<p><a href="{name}.mid">MIDI</a></p>')
        (args.output/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8">'
             '<title>ルール×MuseMorphose</title><h1>編曲結果</h1><p>status: '+html.escape(row['status'])+'</p>'
             +''.join(figures)+'<pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre></html>', encoding='utf-8')
        print(f'{variant}: {row["status"]} / {args.output}/index.html')
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

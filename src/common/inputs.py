"""ステップ3：明示的な小節・文脈・目標指定を採点器へ渡す入力契約。"""
from dataclasses import asdict, dataclass, replace
import math

from src.common.representation import Bar, from_tokens, to_tokens
from src.common.safety import Guards, default_guards, violations
from src.common.difficulty import DIFFICULTY_KEYS, Difficulty


def _object(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError('JSONオブジェクトが必要')
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f'必須キー不足={sorted(missing)}, 未対応キー={sorted(extra)}')


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('有限の数値が必要（文字列・真偽値は不可）')
    return float(value)


def bar_to_payload(bar: Bar):
    """標準トークンだけでは失われる注釈を併記する。IDはトークン順に再採番される。"""
    tokens = to_tokens(bar)
    ordered = sorted(bar.notes, key=lambda n: (n.position, n.instrument, n.id))
    annotations = {
        f'n{i}': {'ornament_group': n.ornament_group, 'ornament_role': n.ornament_role}
        for i, n in enumerate(ordered)
        if n.ornament_group is not None or n.ornament_role != 'main'
    }
    return dict(tokens=tokens, meter=list(bar.meter), length=bar.length,
                fill_regions=[list(region) for region in bar.fill_regions],
                note_annotations=annotations)


def parse_bar(value, vocabulary):
    _object(value, ('tokens', 'meter', 'length'), ('fill_regions', 'note_annotations'))
    if value['meter'] != [4, 4] or any(type(v) is not int for v in value['meter']):
        raise ValueError('meterは[4, 4]のみ対応')
    if type(value['length']) is not int or value['length'] != 96:
        raise ValueError('lengthは整数96のみ対応')
    tokens = value['tokens']
    if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
        raise ValueError('tokensはstandard文字列の配列。ID列は語彙で復号して渡す')
    missing = set(tokens) - vocabulary
    if missing:
        raise ValueError('語彙外トークン: ' + ', '.join(sorted(missing)))
    bar = from_tokens(tokens)
    # 位置指定の逆行・重複・空の位置ブロックも拒否し、入力順による曖昧さをなくす。
    beat = -1
    position = -1
    awaiting_note = False
    for token in tokens[2:-1]:
        if token.startswith('<BEAT_'):
            current = int(token[6:-1])
            if awaiting_note or current <= beat:
                raise ValueError('拍指定の逆行・重複または空の位置指定')
            beat, position, awaiting_note = current, -1, True
        elif token.startswith('<POS_'):
            current = int(token[5:-1])
            if (awaiting_note and position >= 0) or current <= position:
                raise ValueError('拍内位置の逆行・重複または空の位置指定')
            position, awaiting_note = current, True
        else:
            awaiting_note = False
    if awaiting_note:
        raise ValueError('打撃のない拍・位置指定')
    regions = value.get('fill_regions', [])
    if not isinstance(regions, list) or any(not isinstance(r, list) or len(r) != 2 for r in regions):
        raise ValueError('fill_regionsは[start, end]の配列')
    annotations = value.get('note_annotations', {})
    if not isinstance(annotations, dict) or not annotations.keys() <= {n.id for n in bar.notes}:
        raise ValueError('note_annotationsは存在する音符IDをキーとする辞書')
    notes = []
    for note in bar.notes:
        annotation = annotations.get(note.id, {})
        _object(annotation, (), ('ornament_group', 'ornament_role'))
        group = annotation.get('ornament_group')
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError('ornament_groupは空でない文字列またはnull')
        if annotation.get('ornament_role', 'main') not in ('main', 'grace', 'roll'):
            raise ValueError('ornament_roleが不正')
        notes.append(replace(note, **annotation))
    bar = replace(bar, notes=tuple(notes), fill_regions=tuple(tuple(r) for r in regions)).validate()
    sorted_regions = sorted(bar.fill_regions)
    if any(a[1] > b[0] for a, b in zip(sorted_regions, sorted_regions[1:])):
        raise ValueError('Fill区間の重複は不可')
    to_tokens(bar, vocabulary)
    errors = violations(bar, bar, default_guards(bar))
    if errors:
        raise ValueError('入力譜面の制約違反: ' + ', '.join(errors))
    return bar


@dataclass(frozen=True)
class PreparedInput:
    original: Bar
    previous: Bar | None
    following: Bar | None
    target_mode: str
    requested_values: dict[str, float]
    target: dict[str, float]
    original_difficulty: Difficulty
    guards: Guards
    context: dict

    def report(self, model):
        baseline = model.score(self.original, self.original, self.target,
                               previous=self.previous, guards=self.guards)
        return dict(
            schema_version=1, target_mode=self.target_mode, requested_values=self.requested_values,
            absolute_target=self.target, original_difficulty=asdict(self.original_difficulty),
            note_ids=[asdict(n) for n in self.original.notes],
            fill_regions=self.original.fill_regions,
            guards={**asdict(self.guards), 'locked': sorted(self.guards.locked),
                    'accents': sorted(self.guards.accents)},
            previous_present=self.previous is not None, following_present=self.following is not None,
            context=self.context, baseline=asdict(baseline),
            notes=[
                '目標の達成可能性は未判定。探索スコアの重み・制約により無編集が選ばれる場合もある。',
                'following・section・phraseは保存のみ。現スコアには加算しない。',
                'previous=nullは前小節なしとして計算する明示指定。曲中の欠損を補完しない。',
                'Fill区間・装飾音グループは明示注釈のみ。代理特徴から推測しない。',
            ],
        )


def prepare_input(payload, model):
    _object(payload, ('schema_version', 'original', 'previous', 'target'),
            ('following', 'context', 'extra_locked_ids'))
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('schema_versionは整数1')
    original = parse_bar(payload['original'], model.vocabulary)
    previous = None if payload['previous'] is None else parse_bar(payload['previous'], model.vocabulary)
    following_value = payload.get('following')
    following = None if following_value is None else parse_bar(following_value, model.vocabulary)
    # 文脈小節についても難易度算出可能な入力であることを検査する。
    for bar in (previous, following):
        if bar is not None:
            model.difficulty(bar)
    difficulty = model.difficulty(original, previous)
    target_spec = payload['target']
    _object(target_spec, ('mode', 'values'))
    mode = target_spec['mode']
    if mode not in ('absolute', 'delta'):
        raise ValueError('target.modeはabsoluteまたはdeltaを明示')
    _object(target_spec['values'], DIFFICULTY_KEYS)
    values = {key: _number(target_spec['values'][key]) for key in DIFFICULTY_KEYS}
    target = {key: value + (difficulty.normalized[key] if mode == 'delta' else 0.)
              for key, value in values.items()}
    if any(not 0 <= value <= 1 for value in target.values()):
        raise ValueError('変換後の絶対目標が0〜1を外れる。自動クリップはしない')
    context = payload.get('context', {})
    _object(context, (), ('section', 'phrase'))
    if any(value is not None and (not isinstance(value, str) or not value) for value in context.values()):
        raise ValueError('section・phraseは空でない文字列またはnull')
    extra = payload.get('extra_locked_ids', [])
    if not isinstance(extra, list) or any(not isinstance(v, str) for v in extra):
        raise ValueError('extra_locked_idsは音符ID文字列の配列')
    if len(set(extra)) != len(extra) or not set(extra) <= {n.id for n in original.notes}:
        raise ValueError('extra_locked_idsの重複または存在しないID')
    guards = default_guards(original)
    guards = replace(guards, locked=guards.locked | frozenset(extra))
    return PreparedInput(original, previous, following, mode, values, target,
                         difficulty, guards, dict(context))


def request_from_pair(pair, idx2token):
    """v1擬似ペアの目標難易度を渡す。教師譜面や候補側文脈は探索入力に含めない。"""
    def decode(ids):
        if ids is None:
            return None
        # v1は4/4・96位置のstandardへ前処理済み。
        return dict(tokens=[idx2token[int(i)] for i in ids], meter=[4, 4], length=96)
    if pair['context']['after_difficulty_context'] != 'before_previous':
        raise ValueError('ペアの候補難易度が原譜前小節の文脈で算出されていない')
    if len(pair['d_after']) != len(DIFFICULTY_KEYS):
        raise ValueError('ペア難易度はv2の5属性が必要')
    return dict(schema_version=1, original=decode(pair['X_before']),
                previous=decode(pair['context']['previous']),
                following=decode(pair['context']['following']),
                target=dict(mode='absolute', values=dict(zip(DIFFICULTY_KEYS, pair['d_after']))),
                context=dict(section=pair['context']['section'], phrase=None))

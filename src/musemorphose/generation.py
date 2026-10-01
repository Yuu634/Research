"""standard文法に従う自己回帰生成。温度1のsampleは学習の方策勾配にも使う。"""
from dataclasses import dataclass
import torch

from src.common.representation import INSTRUMENTS, from_tokens
from src.data.drum_difficulty_calculator import HAND_INSTRUMENTS
from src.common.comparison import align_note_ids


class Grammar:
    def __init__(self, vocabulary):
        self.ids = vocabulary['token2idx']
        self.tokens = vocabulary['idx2token']
        self.notes = {}
        for token, index in self.ids.items():
            for instrument in sorted(INSTRUMENTS, key=len, reverse=True):
                if token.startswith(instrument.upper()+'_'):
                    # 実際の復号器で解釈できる奏法・強弱だけを許す。
                    from_tokens(['<TEMPO_120>', '<BAR>', '<BEAT_1>', '<POS_0>', token, '<EOS>'])
                    self.notes[index] = instrument
                    break

    def allowed(self, prefix):
        last = self.tokens[prefix[-1]]
        beat, position, used = 0, -1, set()
        for index in prefix[2:]:
            token = self.tokens[index]
            if token.startswith('<BEAT_'):
                beat, position, used = int(token[6:-1]), -1, set()
            elif token.startswith('<POS_'):
                position, used = int(token[5:-1]), set()
            elif index in self.notes:
                used.add(self.notes[index])
        allowed = []
        if last == '<BAR>' or prefix[-1] in self.notes:
            allowed.append(self.ids['<EOS>'])
            allowed.extend(self.ids[f'<BEAT_{b}>'] for b in range(beat+1, 5) if f'<BEAT_{b}>' in self.ids)
        if last.startswith('<BEAT_') or prefix[-1] in self.notes:
            allowed.extend(self.ids[f'<POS_{p}>'] for p in range(position+1, 24) if f'<POS_{p}>' in self.ids)
        if last.startswith('<POS_') or prefix[-1] in self.notes:
            hands = len(used & HAND_INSTRUMENTS)
            allowed.extend(i for i, inst in self.notes.items() if inst not in used and (inst not in HAND_INSTRUMENTS or hands < 2))
        return sorted(set(allowed))


@dataclass
class Generation:
    candidate: object
    tokens: list
    log_probability: torch.Tensor
    failure_reason: str | None


def generate(model, batch, prepared, vocabulary, *, sample=False, grammar=None):
    """1小節生成。呼出側が推論時no_grad、学習時勾配有効を指定する。"""
    if batch['encoder_tokens'].shape[0] != 1:
        raise ValueError('生成は1件ずつ実行する')
    grammar = Grammar(vocabulary) if grammar is None else grammar
    device = next(model.parameters()).device
    ids = vocabulary['token2idx']
    prefix = [ids[f'<TEMPO_{int(prepared.original.bpm)}>'], ids['<BAR>']]
    encoded = model.encode(batch, sample=False)
    log_probs = []
    failure = 'max_tokens_without_eos'
    for _ in range(model.config.max_tokens-len(prefix)):
        allowed = grammar.allowed(prefix)
        if not allowed:
            failure = 'no_grammar_continuation'
            break
        tokens = torch.tensor(prefix, device=device).unsqueeze(1)
        logits = model.decode(tokens, encoded)[-1, 0, allowed]
        if not torch.isfinite(logits).all():
            raise ValueError('生成logitsが非有限')
        distribution = torch.distributions.Categorical(logits=logits)
        choice = distribution.sample() if sample else logits.argmax()
        log_probs.append(distribution.log_prob(choice))
        prefix.append(allowed[int(choice)])
        if prefix[-1] == ids['<EOS>']:
            failure = None
            break
    log_probability = torch.stack(log_probs).sum() if log_probs else encoded['segment'].sum()*0
    strings = [vocabulary['idx2token'][i] for i in prefix]
    candidate = None if failure else align_note_ids(prepared.original, from_tokens(strings))
    return Generation(candidate, strings, log_probability, failure)


def score_generation(model, batch, generated, vocabulary, grammar=None):
    """sampleはno_gradで生成し、確定した系列のlog pを1回のforwardで再計算する。

    逐次生成の全計算グラフを保持しない。学習時もdropout無効の同一方策を使用する。
    """
    grammar = Grammar(vocabulary) if grammar is None else grammar
    prefix = [vocabulary['token2idx'][t] for t in generated.tokens]
    if len(prefix) < 3:
        raise ValueError('確率を再計算する生成系列がない')
    tokens = torch.tensor(prefix[:-1], device=next(model.parameters()).device).unsqueeze(1)
    logits = model.decode(tokens, model.encode(batch, sample=False))[:, 0]
    probabilities = []
    for i in range(2, len(prefix)):
        allowed = grammar.allowed(prefix[:i])
        probabilities.append(logits[i-1, allowed].log_softmax(-1)[allowed.index(prefix[i])])
    return torch.stack(probabilities).sum()

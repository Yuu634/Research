"""標準96位置の小節表現。音符ID・奏法・明示された装飾音グループを保持する。"""
from dataclasses import dataclass, replace
import math

from src.data.drum_difficulty_calculator import HAND_INSTRUMENTS, FOOT_INSTRUMENTS

INSTRUMENTS = HAND_INSTRUMENTS | FOOT_INSTRUMENTS
LEVELS = {'Ghost': 0, 'Normal': 1, 'Accent': 2}


@dataclass(frozen=True)
class Note:
    id: str
    position: int
    instrument: str
    level: str = 'Normal'
    technique: str = 'HIT'
    ornament_group: str | None = None
    ornament_role: str = 'main'
    explicit_level: bool = True

    def token(self):
        base = f'{self.instrument.upper()}_{self.technique}'
        return base + ('_'+self.level if self.explicit_level else '')


@dataclass(frozen=True)
class Bar:
    notes: tuple[Note, ...]
    bpm: float = 120.
    meter: tuple[int,int] = (4,4)
    length: int = 96
    # Fillは自動認識せず、外から明示された半開区間だけを使う。
    fill_regions: tuple[tuple[int,int], ...] = ()

    def validate(self):
        if not math.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError('BPMは有限の正数')
        if self.meter != (4,4) or self.length != 96:
            raise ValueError('この版は4/4・96位置の小節に限定')
        if len({n.id for n in self.notes}) != len(self.notes):
            raise ValueError('音符IDの重複')
        for n in self.notes:
            if type(n.position) is not int or not 0 <= n.position < self.length:
                raise ValueError('位置は0〜95の整数')
            if n.instrument not in INSTRUMENTS or n.level not in LEVELS:
                raise ValueError('未対応の楽器・強弱')
            if n.ornament_role not in ('main','grace','roll'):
                raise ValueError('装飾音の役割が不正')
            if not n.technique or '_' in n.technique:
                raise ValueError('奏法名が不正')
        for start,end in self.fill_regions:
            if not (type(start) is int and type(end) is int and 0 <= start < end <= 96):
                raise ValueError('Fill区間が不正')
        return self

    def with_notes(self, notes):
        return replace(self, notes=tuple(sorted(notes,key=lambda n:(n.position,n.instrument,n.id)))).validate()


def from_tokens(tokens):
    """単一小節のstandard文字列トークンを読む。近接音から奏法は推定しない。"""
    if len(tokens)<3 or not tokens[0].startswith('<TEMPO_') or tokens[1]!='<BAR>' or tokens[-1]!='<EOS>':
        raise ValueError('TEMPO・BARから始まりEOSで終わる単一小節が必要')
    bpm=float(tokens[0][7:-1]); beat=position=None; notes=[]
    for token in tokens[2:-1]:
        if token.startswith('<BEAT_'):
            beat=int(token[6:-1])-1; position=None
            if not 0<=beat<4:raise ValueError('拍番号が不正')
        elif token.startswith('<POS_'):
            position=int(token[5:-1])
            if beat is None or not 0<=position<24:raise ValueError('拍内位置が不正')
        else:
            if beat is None or position is None:raise ValueError('打撃の位置指定が不足')
            inst=next((i for i in sorted(INSTRUMENTS,key=len,reverse=True) if token.startswith(i.upper()+'_')),None)
            if inst is None:raise ValueError('未知の打撃トークン: '+token)
            suffix=token[len(inst)+1:]; parts=suffix.rsplit('_',1)
            explicit=len(parts)==2 and parts[1] in LEVELS
            tech,level=(parts[0],parts[1]) if explicit else (suffix,'Normal')
            notes.append(Note(f'n{len(notes)}',beat*24+position,inst,level,tech,explicit_level=explicit))
    return Bar(tuple(notes),bpm).validate()


def to_tokens(bar, vocabulary=None):
    """standardへ変換。装飾グループ／Fill注釈は別途保持する必要がある。"""
    bar.validate()
    if bar.bpm != int(bar.bpm):raise ValueError('standardのBPMは整数')
    tokens=[f'<TEMPO_{int(bar.bpm)}>','<BAR>']; last_beat=last_pos=None
    for note in sorted(bar.notes,key=lambda n:(n.position,n.instrument,n.id)):
        beat,pos=divmod(note.position,24)
        if beat!=last_beat:
            tokens.append(f'<BEAT_{beat+1}>'); last_beat=beat; last_pos=None
        if pos!=last_pos:tokens.append(f'<POS_{pos}>'); last_pos=pos
        tokens.append(note.token())
    tokens.append('<EOS>')
    if vocabulary is not None:
        missing=sorted(set(tokens)-set(vocabulary))
        if missing:raise ValueError('語彙で表現できないトークン: '+', '.join(missing))
    return tokens

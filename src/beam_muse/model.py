"""BeamSearchの中間譜面と原譜の骨格を条件にするVAE。"""
from src.musemorphose.conditional import ConditionalMuseMorphose, ModelConfig


class BeamMuseMorphose(ConditionalMuseMorphose):
    def __init__(self, n_token, pad_id, bar_id, config=ModelConfig()):
        if not config.use_skeleton:
            raise ValueError('ハイブリッドでは原譜の骨格条件を有効にする')
        super().__init__(n_token, pad_id, bar_id, config)

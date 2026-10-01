"""信頼できるローカルpickleの旧クラス参照を現在の配置へ移す。"""
import io
import pickle


MODULE_MOVES = {
    'src.arrangement.representation': 'src.common.representation',
    'src.arrangement.safety': 'src.common.safety',
    'src.arrangement.inputs': 'src.common.inputs',
    'src.arrangement.evaluation': 'src.common.evaluation',
    'src.arrangement.examples': 'src.common.examples',
    'src.arrangement.beam_search': 'src.beamsearch.beam_search',
    'src.arrangement.rules': 'src.beamsearch.rules',
    'src.arrangement.scoring': 'src.beamsearch.scoring',
    'src.arrangement.selected_model': 'src.beamsearch.selected_model',
    'src.models.musemorphose': 'src.musemorphose.model',
    'src.models.transformer_encoder': 'src.musemorphose.transformer_encoder',
    'src.models.transformer_helpers': 'src.musemorphose.transformer_helpers',
    'src.models.limb_bias': 'src.musemorphose.limb_bias',
    'src.models.position_bias': 'src.musemorphose.position_bias',
    'src.models.bias_schedule': 'src.musemorphose.bias_schedule',
    'src.beam_muse.features': 'src.common.skeleton',
    'src.beam_muse.conditioning': 'src.common.conditioning',
    'src.beam_muse.evaluation': 'src.common.comparison',
    'src.beam_muse.losses': 'src.common.losses',
    'src.beam_muse.generation': 'src.musemorphose.generation',
}


class LayoutUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(MODULE_MOVES.get(module, module), name)


def loads_local_pickle(payload):
    """通常のpickle同様、未信頼の入力に対する安全化機能ではない。"""
    return LayoutUnpickler(io.BytesIO(payload)).load()

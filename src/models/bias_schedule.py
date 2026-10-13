"""
Layer-wise bias schedule for limb_bias and position_bias.

build_layer_bias_schedule() converts a schedule_mode string into a per-layer
list consumed by VAETransformerEncoder and VAETransformerDecoder.

Each element of the list is one of:
  'both'     - apply both limb_bias and position_bias (existing behavior)
  'limb'     - apply limb_bias only
  'position' - apply position_bias only
  'none'     - apply neither
"""

from typing import List, Optional

VALID_ENTRIES = frozenset({'limb', 'position', 'both', 'none'})
VALID_MODES = frozenset({
    'both',
    'alternating_lp',
    'alternating_pl',
    'limb_only',
    'position_only',
    'custom',
})


def build_layer_bias_schedule(
    n_layer: int,
    schedule_mode: str = 'both',
    custom: Optional[List[str]] = None,
) -> List[str]:
    """Return a per-layer bias schedule list of length n_layer.

    Args:
        n_layer: Number of transformer layers.
        schedule_mode: One of VALID_MODES.
            'both'           - all layers get both biases (default / backward-compat)
            'alternating_lp' - even layers: limb, odd layers: position
            'alternating_pl' - even layers: position, odd layers: limb
            'limb_only'      - all layers get limb_bias only
            'position_only'  - all layers get position_bias only
            'custom'         - use `custom` list verbatim (must equal n_layer in length)
        custom: Required when schedule_mode='custom'. List of VALID_ENTRIES per layer.

    Returns:
        List[str] of length n_layer, each element in VALID_ENTRIES.
    """
    if schedule_mode not in VALID_MODES:
        raise ValueError(
            f"bias_schedule must be one of {sorted(VALID_MODES)}, got '{schedule_mode}'"
        )

    if schedule_mode == 'both':
        return ['both'] * n_layer
    if schedule_mode == 'limb_only':
        return ['limb'] * n_layer
    if schedule_mode == 'position_only':
        return ['position'] * n_layer
    if schedule_mode == 'alternating_lp':
        return ['limb' if i % 2 == 0 else 'position' for i in range(n_layer)]
    if schedule_mode == 'alternating_pl':
        return ['position' if i % 2 == 0 else 'limb' for i in range(n_layer)]

    # schedule_mode == 'custom'
    if custom is None:
        raise ValueError("bias_schedule='custom' requires a non-null custom list")
    if len(custom) != n_layer:
        raise ValueError(
            f"bias_schedule custom list length ({len(custom)}) "
            f"must equal n_layer ({n_layer})"
        )
    invalid = [e for e in custom if e not in VALID_ENTRIES]
    if invalid:
        raise ValueError(
            f"Invalid entries in bias_schedule custom list: {invalid}. "
            f"Valid values: {sorted(VALID_ENTRIES)}"
        )
    return list(custom)

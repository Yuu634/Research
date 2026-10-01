from src.arrangement.examples import make_bar
from src.arrangement.safety import edits
from src.beam_muse.evaluation import align_note_ids


def test_ids_are_matched_by_music_not_token_order():
    original = make_bar([(0, 'snare'), (0, 'kick'), (24, 'hh_closed')])
    generated = make_bar([(0, 'kick'), (0, 'snare'), (24, 'hh_closed', 'Ghost'), (48, 'kick')])
    aligned = align_note_ids(original, generated)
    assert [n.id for n in aligned.notes[:3]] == ['n1', 'n0', 'n2']
    assert aligned.notes[3].id not in {n.id for n in original.notes}


def test_moved_note_counts_as_one_edit_for_token_generated_and_rule_outputs():
    original = make_bar([(0, 'kick'), (24, 'snare'), (12, 'hh_closed')])
    moved = make_bar([(0, 'kick'), (24, 'snare'), (18, 'hh_closed')])
    result = align_note_ids(original, moved)
    assert edits(original, result) == dict(added=0, deleted=0, modified=1, ratio=1/3)

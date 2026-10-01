"""原譜だけから骨格を取得する。未注釈のFill・曲区間は推測しない。"""
from dataclasses import dataclass
import numpy as np

CHANNELS = ('kick', 'snare', 'main_kick', 'backbeat_snare', 'accent', 'fill_region', 'fill_start', 'fill_end')
SECTIONS = ('intro', 'verse', 'chorus', 'bridge', 'outro', 'other')
FEATURE_SIZE = 96 * len(CHANNELS) + 2 + len(SECTIONS)


@dataclass(frozen=True)
class SkeletonFeatures:
    grid: np.ndarray
    fill_known: bool
    section_known: bool
    section: np.ndarray

    def vector(self):
        return np.concatenate((self.grid.reshape(-1), [self.fill_known, self.section_known], self.section)).astype(np.float32)

    def report(self):
        return dict(channels=list(CHANNELS), positions={name: np.flatnonzero(self.grid[:, i]).tolist()
                    for i, name in enumerate(CHANNELS)}, fill_known=self.fill_known,
                    section_known=self.section_known, section=self.section.tolist(), size=FEATURE_SIZE)


def extract_skeleton(original, *, section=None, fill_known=None):
    original.validate()
    if section is not None and (not isinstance(section, str) or not section):
        raise ValueError('sectionは空でない文字列またはNone')
    if fill_known is not None and type(fill_known) is not bool:
        raise ValueError('fill_knownはboolまたはNone')
    if original.fill_regions and fill_known is False:
        raise ValueError('Fill区間があるのに不明とは指定できない')
    grid = np.zeros((96, len(CHANNELS)), dtype=np.float32)
    for note in original.notes:
        p = note.position
        if note.instrument == 'kick':
            grid[p, 0] = 1
            grid[p, 2] = float(p % 24 == 0)
        if note.instrument in ('snare', 'snare_rim'):
            grid[p, 1] = 1
            grid[p, 3] = float(p in (24, 72))
        if note.level == 'Accent':
            grid[p, 4] = 1
    for start, end in original.fill_regions:
        grid[start:end, 5] = 1
        positions = [n.position for n in original.notes if start <= n.position < end]
        if positions:
            grid[min(positions), 6] = 1
            grid[max(positions), 7] = 1
    section_vector = np.zeros(len(SECTIONS), dtype=np.float32)
    if section is not None:
        section_vector[SECTIONS.index(section) if section in SECTIONS else -1] = 1
    return SkeletonFeatures(grid, bool(original.fill_regions) if fill_known is None else fill_known,
                            section is not None, section_vector)

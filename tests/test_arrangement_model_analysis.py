"""新モデルの解析を保存済み設定・重み・方式に結び付ける。"""
import json
import math

import pytest
import torch

from data.pairdata.io import file_hash, write_json
from scripts.analysis.analyze_arrangement_models import analyze, weight_statistics
from src.common.provenance import implementation_hashes


def test_weight_statistics_aggregates_components_and_rejects_nonfinite():
    stats = weight_statistics({'encoder.a':torch.tensor([3.,4.]), 'encoder.b':torch.tensor([0.]),
                               'buffer':torch.tensor([1],dtype=torch.long)})
    assert stats == dict(encoder=dict(elements=3, l2_norm=5., rms=math.sqrt(25/3)))
    with pytest.raises(ValueError, match='非有限'):
        weight_statistics({'encoder.a':torch.tensor([float('nan')])})


@pytest.mark.parametrize('variant', ['musemorphose','beam_muse'])
def test_analysis_checks_checkpoint_and_saves_report_without_changing_model(tmp_path, variant):
    run = tmp_path/'run'
    directory = run/'selected_model'
    directory.mkdir(parents=True)
    identity = dict(sources=implementation_hashes('beam_muse'))
    trial = dict(name=directory.name, variant=variant, model=dict(d_model=4))
    path = directory/'best.pt'
    torch.save(dict(trial=trial, identity=identity, epoch=2, state_dict={'encoder.weight':torch.tensor([3.,4.])}), path)
    checksum = file_hash(path)
    write_json(run/'run.json',dict(identity=identity))
    write_json(run/'selection.json',dict(identity=identity, selected_on='val',
        winners={variant:dict(name=directory.name, checkpoint_sha256=checksum)}))
    write_json(directory/'completed.json',dict(identity=identity,trial=trial,epoch=2,validation=dict(n=3),checkpoint_sha256=checksum))
    before = {p: (p.read_bytes(),p.stat().st_mtime_ns) for p in run.rglob('*') if p.is_file()}
    output = tmp_path/'analysis'
    assert analyze(run, variant, output) == output/'index.html'
    report = json.loads((output/'model_analysis.json').read_text())
    assert report['variant'] == variant and report['winner'] == directory.name
    assert report['models'][0]['weights']['encoder']['l2_norm'] == 5.
    assert (output/'selected_model_weight_rms.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert all((p.read_bytes(),p.stat().st_mtime_ns) == value for p,value in before.items())
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='重みが不一致'):
        analyze(run, variant, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()

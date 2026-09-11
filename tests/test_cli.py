import sys

from flypareto import cli


class _Result:
    def to_dict(self):
        return {}


def test_chemotaxis_cli_routes_neural_threshold(monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(cli.Connectome, "load", lambda path: object())

    def fake_trial(graph, annotations, **kwargs):
        del graph, annotations
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(cli, "run_chemotaxis_trial", fake_trial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flypareto",
            "chemotaxis",
            str(tmp_path / "graph"),
            "--annotations",
            str(tmp_path / "annotations.feather"),
            "--neural-threshold",
            "0.8",
        ],
    )
    cli.main()
    assert captured["neural_threshold"] == 0.8
    assert captured["decoder_mode"] == "steering"
    assert captured["odor_channels"] == ("ORN_DM1", "ORN_VA2")
    assert capsys.readouterr().out.strip() == "{}"


def test_embodied_smoke_cli_routes_only_supported_arguments(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.Connectome, "load", lambda path: object())

    def fake_smoke(
        graph,
        annotations,
        neural_steps,
        modality,
        left_stimulus,
        right_stimulus,
        stimulus_steps,
        synaptic_scale,
        proprioceptive_speed_scale,
        decoder_half_saturation,
        decoder_smoothing,
        seed,
    ):
        del (
            graph,
            annotations,
            neural_steps,
            modality,
            left_stimulus,
            right_stimulus,
            stimulus_steps,
            synaptic_scale,
            proprioceptive_speed_scale,
            decoder_half_saturation,
            decoder_smoothing,
            seed,
        )
        return _Result()

    monkeypatch.setattr(cli, "run_closed_loop_smoke", fake_smoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flypareto",
            "embodied-smoke",
            str(tmp_path / "graph"),
            "--annotations",
            str(tmp_path / "annotations.feather"),
            "--neural-steps",
            "3",
        ],
    )
    cli.main()
    assert capsys.readouterr().out.strip() == "{}"

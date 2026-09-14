def test_public_skeleton_imports() -> None:
    import diploma_pinn.boundaries
    import diploma_pinn.config
    import diploma_pinn.data
    import diploma_pinn.evaluation
    import diploma_pinn.formulations
    import diploma_pinn.instrumentation
    import diploma_pinn.integrations
    import diploma_pinn.losses
    import diploma_pinn.models
    import diploma_pinn.observations
    import diploma_pinn.operators
    import diploma_pinn.parity
    import diploma_pinn.runtime
    import diploma_pinn.sampling
    import diploma_pinn.training


def test_cli_declares_all_exp001_lifecycle_commands() -> None:
    from diploma_pinn.cli import build_parser

    parser = build_parser()
    for command in ("validate-data", "parity", "train", "benchmark", "evaluate"):
        namespace = parser.parse_args([command, "--config", "experiment.yaml"])
        assert namespace.command == command

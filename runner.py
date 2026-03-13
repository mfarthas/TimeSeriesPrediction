import argparse
import subprocess
import sys


DEFAULTS = {
    "T": 100_000,
    "F": 8,
    "W": 100,
    "H": 50,
    "normal_noise": 0.24,
    "incident_noise": 0.26,
    "normal_shared": 0.15,
}

DESCRIPTIONS = {
    "T": "total number of timesteps in the generated signal",
    "F": "number of sensor channels",
    "W": "input window size (timesteps fed to the model)",
    "H": "prediction horizon (how far ahead to predict incidents)",
    "normal_noise": "per-channel noise std during normal operation",
    "incident_noise": "per-channel noise std during an incident",
    "normal_shared": "shared component std (controls cross-channel correlation)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full incident prediction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--T", type=int, default=None,
                        help=DESCRIPTIONS["T"])
    parser.add_argument("--F", type=int, default=None,
                        help=DESCRIPTIONS["F"])
    parser.add_argument("--W", type=int, default=None,
                        help=DESCRIPTIONS["W"])
    parser.add_argument("--H", type=int, default=None,
                        help=DESCRIPTIONS["H"])
    parser.add_argument("--normal-noise", type=float, default=None,
                        help=DESCRIPTIONS["normal_noise"])
    parser.add_argument("--incident-noise", type=float, default=None,
                        help=DESCRIPTIONS["incident_noise"])
    parser.add_argument("--normal-shared", type=float, default=None,
                        help=DESCRIPTIONS["normal_shared"])
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip interactive prompts, use defaults or "
                             "provided values")
    return parser.parse_args()


def prompt_params(args):
    cli_overrides = {
        "T": args.T,
        "F": args.F,
        "W": args.W,
        "H": args.H,
        "normal_noise": args.normal_noise,
        "incident_noise": args.incident_noise,
        "normal_shared": args.normal_shared,
    }

    params = {}
    print("\nDataset and model parameters (press Enter to keep default):\n")

    for key, default in DEFAULTS.items():
        cli_val = cli_overrides[key]
        if cli_val is not None:
            params[key] = cli_val
            print(f"  {key}: {cli_val}  (set via CLI)")
            continue

        desc = DESCRIPTIONS[key]
        try:
            raw = input(f"  {key} [{default}]  {desc}\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw == "":
            params[key] = default
        else:
            try:
                params[key] = type(default)(raw)
            except ValueError:
                print(f"  Invalid value, using default: {default}")
                params[key] = default

    return params


def build_env_args(params):
    return [
        f"--T={params['T']}",
        f"--F={params['F']}",
        f"--W={params['W']}",
        f"--H={params['H']}",
        f"--normal-noise={params['normal_noise']}",
        f"--incident-noise={params['incident_noise']}",
        f"--normal-shared={params['normal_shared']}",
    ]


def run_step(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStep failed: {label}")
        sys.exit(result.returncode)


def main():
    args = parse_args()

    if args.yes:
        params = {
            "T": args.T or DEFAULTS["T"],
            "F": args.F or DEFAULTS["F"],
            "W": args.W or DEFAULTS["W"],
            "H": args.H or DEFAULTS["H"],
            "normal_noise": args.normal_noise or DEFAULTS["normal_noise"],
            "incident_noise": (
                args.incident_noise or DEFAULTS["incident_noise"]
            ),
            "normal_shared": args.normal_shared or DEFAULTS["normal_shared"],
        }
    else:
        params = prompt_params(args)

    extra = build_env_args(params)

    print("\nRunning pipeline with:")
    for k, v in params.items():
        print(f"  {k} = {v}")

    run_step("Step 1/4: Generating dataset",
             [sys.executable, "data_generator.py"] + extra)

    run_step("Step 2/4: Training baseline (logistic regression)",
             [sys.executable, "baseline_logistic_regression.py"])

    run_step("Step 3/4: Training LSTM",
             [sys.executable, "LSTM.py"])

    run_step("Step 4/4: Evaluation plots",
             [sys.executable, "evaluation.py"])

    print("\nDone.")


if __name__ == "__main__":
    main()

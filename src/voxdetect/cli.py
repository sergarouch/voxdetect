"""Command line for import, training, evaluation, and inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxdetect import __version__
from voxdetect.evaluate import evaluate
from voxdetect.importer import DatasetImportError, import_collected, import_dataset
from voxdetect.infer import infer_paths
from voxdetect.synthetic import write_synthetic
from voxdetect.train import fit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxdetect",
        description="Train a voice-activity detector and score new audio.",
    )
    parser.add_argument("--version", action="version", version=f"voxdetect {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    import_cmd = commands.add_parser("import", help="Read audio and write train/val/test manifests")
    source = import_cmd.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Folder with speech/ and non_speech/ subfolders")
    source.add_argument("--csv", help="CSV with path,label columns")
    source.add_argument("--collected", help="Catalog directory written by voxdetect collect")
    import_cmd.add_argument("--out", default="data/manifests", help="Where to write the manifests")
    import_cmd.add_argument("--seed", type=int, default=42)

    train_cmd = commands.add_parser("train", help="Train the CRNN from a manifest directory")
    train_cmd.add_argument("--manifests", default="data/manifests", help="Directory with train.csv and val.csv")
    train_cmd.add_argument("--epochs", type=int, default=20)
    train_cmd.add_argument("--batch-size", type=int, default=16)
    train_cmd.add_argument("--lr", type=float, default=1e-3)
    train_cmd.add_argument("--out", default="runs/best.pt")
    train_cmd.add_argument("--seed", type=int, default=42)

    eval_cmd = commands.add_parser("evaluate", help="Score a manifest with a checkpoint")
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--manifest", default="data/manifests/test.csv")
    eval_cmd.add_argument("--batch-size", type=int, default=16)
    eval_cmd.add_argument("--threshold", type=float, default=None)

    infer_cmd = commands.add_parser("infer", help="Score a file or a folder of audio")
    infer_cmd.add_argument("--checkpoint", required=True)
    infer_cmd.add_argument("--audio", required=True, help="Audio file or directory")
    infer_cmd.add_argument("--threshold", type=float, default=None)

    synth_cmd = commands.add_parser(
        "synthetic",
        help="Write a tiny synthetic dataset to check the pipeline",
    )
    synth_cmd.add_argument("--out", default="data/raw/synthetic")
    synth_cmd.add_argument("--clips", type=int, default=8)
    synth_cmd.add_argument("--seed", type=int, default=0)

    collect_cmd = commands.add_parser("collect", help="Run the anonymous collection website")
    collect_cmd.add_argument("--host", default="0.0.0.0")
    collect_cmd.add_argument("--port", type=int, default=8741)
    collect_cmd.add_argument("--data", default="data/collected", help="Where anonymous clips are stored")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "import":
            if args.collected:
                report = import_collected(args.collected, out_dir=args.out, seed=args.seed)
            else:
                report = import_dataset(source=args.input, csv_path=args.csv, out_dir=args.out, seed=args.seed)
            print(report.summary())
            return 0
        if args.command == "train":
            root = Path(args.manifests)
            val_path = root / "val.csv"
            fit(
                train_manifest=root / "train.csv",
                val_manifest=val_path if val_path.is_file() else None,
                out_path=args.out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
            )
            print(f"Checkpoint: {Path(args.out).resolve()}")
            return 0
        if args.command == "evaluate":
            evaluate(args.checkpoint, args.manifest, batch_size=args.batch_size, threshold=args.threshold)
            return 0
        if args.command == "infer":
            infer_paths(args.checkpoint, args.audio, threshold=args.threshold)
            return 0
        if args.command == "collect":
            import uvicorn

            from voxdetect.collect.app import create_app

            root = Path(args.data).expanduser().resolve()
            print(f"Collection site at http://127.0.0.1:{args.port}")
            print(f"Clips save to {root}")
            print("Share that address with your community. The site stores no names and does not play clips back.")
            uvicorn.run(
                create_app(root),
                host=args.host,
                port=args.port,
                access_log=False,
                log_level="info",
            )
            return 0
        if args.command == "synthetic":
            root = write_synthetic(args.out, clips_per_class=args.clips, seed=args.seed)
            print(f"Wrote synthetic clips to {root.resolve()}")
            print("Import them with:")
            print(f"  voxdetect import --input {root} --out data/manifests")
            return 0
    except (DatasetImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"error: unknown command {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

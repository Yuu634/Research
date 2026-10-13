#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from pprint import pprint


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def summarize(obj, max_items: int = 5, depth: int = 0, max_depth: int = 3):
    indent = "  " * depth

    if depth >= max_depth:
        print(f"{indent}... (max depth reached)")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}dict(len={len(obj)}), keys(sample)={keys[:max_items]}")
        for k in keys[:max_items]:
            print(f"{indent}- key={k!r} -> {type(obj[k]).__name__}")
            summarize(obj[k], max_items=max_items, depth=depth + 1, max_depth=max_depth)

    elif isinstance(obj, (list, tuple)):
        name = type(obj).__name__
        print(f"{indent}{name}(len={len(obj)})")
        for i, item in enumerate(obj[:max_items]):
            print(f"{indent}- [{i}] type={type(item).__name__}")
            summarize(item, max_items=max_items, depth=depth + 1, max_depth=max_depth)

    else:
        val = repr(obj)
        if len(val) > 200:
            val = val[:200] + "..."
        print(f"{indent}{type(obj).__name__}: {val}")


def inspect_remi_piece(obj):
    if not isinstance(obj, (list, tuple)) or len(obj) != 2:
        return

    bar_pos, events = obj
    if not isinstance(bar_pos, list):
        return
    if not isinstance(events, list):
        return

    print("\n[detected format] remi piece (bar_pos, events)")
    print(f"- n_bars: {len(bar_pos)}")
    print(f"- n_events: {len(events)}")
    print(f"- bar_pos head: {bar_pos[:10]}")
    if events:
        print(f"- first event: {events[0]}")
        print(f"- last event: {events[-1]}")
        print("- first 10 events:")
        for ev in events[:10]:
            print(f"  {ev}")


def main():
    parser = argparse.ArgumentParser(description="Inspect pickle file contents.")
    parser.add_argument("pkl_path", help="Path to .pkl file")
    parser.add_argument("--full", action="store_true", help="Print full object via pprint")
    parser.add_argument("--max-items", type=int, default=5, help="Number of items to show per level in summary")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth for summary")
    args = parser.parse_args()

    path = Path(args.pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    obj = load_pickle(path)

    print(f"[file] {path}")
    print(f"[root type] {type(obj).__name__}")

    if args.full:
        print("\n[full output]")
        pprint(obj)
    else:
        print("\n[summary output]")
        summarize(obj, max_items=args.max_items, max_depth=args.max_depth)

    inspect_remi_piece(obj)


if __name__ == "__main__":
    main()

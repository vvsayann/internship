import subprocess
import os
import re
import sys

PRESERVE_EXACT = {
    "ISSUE_TAGS.md",
}

def get_tracked_files():
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [line for line in out.splitlines() if line.strip()]

def to_snake_component(name: str) -> str:
    s = name.lower()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"_(\.[^./]+)$", r"\1", s)
    s = s.strip("_")
    return s


def transform_path(path: str) -> str:
    parts = path.split("/")
    new_parts = [to_snake_component(p) for p in parts]
    return "/".join(new_parts)


def main():
    apply = "--apply" in sys.argv

    files = get_tracked_files()
    renames = []

    for f in files:
        base = os.path.basename(f)

        # Skip dotfiles at repo root (.gitignore, .readme.txt, etc.) and any
        # explicitly preserved filenames.
        if f.startswith(".") or base in PRESERVE_EXACT:
            continue

        new_f = transform_path(f)
        if new_f != f:
            renames.append((f, new_f))

    if not renames:
        print("Nothing to rename -- all tracked files already match the naming rule.")
        return

    print(f"{len(renames)} file(s) will be renamed:\n")
    for old, new in renames:
        print(f"  {old}\n    -> {new}\n")

    if not apply:
        print("Dry run only -- nothing was changed.")
        print("Re-run with --apply to actually perform these renames.")
        return

    print("Applying renames...\n")
    for old, new in renames:
        new_dir = os.path.dirname(new)
        if new_dir:
            os.makedirs(new_dir, exist_ok=True)
        result = subprocess.run(["git", "mv", old, new], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED: {old} -> {new}")
            print(f"    {result.stderr.strip()}")
        else:
            print(f"  OK: {old} -> {new}")

    print("\nDone. Run 'git status' to review, then commit and push.")


if __name__ == "__main__":
    main()

from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_repo_root() -> Path:
    """
    Return the repository root for the source tree layout used by this project.

    The function is intentionally tiny and filesystem-based because configs are
    normally stored in the repository, while jobs may be launched from a
    different working directory.
    """
    return Path(__file__).resolve().parents[2]


def _candidate_paths(
    path: str | Path,
    base_path: str | Path | None = None,
) -> list[Path]:
    raw = Path(path).expanduser()

    if raw.is_absolute():
        return [raw]

    candidates: list[Path] = []

    if base_path is not None:
        base = Path(base_path).expanduser()
        base_dir = base.parent if base.suffix else base
        candidates.append(base_dir / raw)

    candidates.append(Path.cwd() / raw)
    candidates.append(get_repo_root() / raw)

    unique: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        unique.append(candidate)
        seen.add(key)

    return unique


def resolve_path(
    path: str | Path,
    base_path: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a config-declared path robustly.

    Search order for relative paths:
      1. relative to the declaring config file/directory, if provided
      2. relative to the current working directory
      3. relative to the repository root

    This keeps existing repo-root configs working and also supports portable
    run/source configs placed outside the repository.
    """
    candidates = _candidate_paths(path, base_path=base_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if must_exist:
        formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"Could not resolve path: {path}\n"
            f"Tried:\n{formatted}"
        )

    return candidates[0]

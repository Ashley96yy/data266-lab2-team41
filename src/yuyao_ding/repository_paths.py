"""Resolve recorded repository paths without rewriting original experiment metadata."""
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def member_migration(root=ROOT):
    return json.loads((Path(root) / 'audit/yuyao_ding/workspace_relocation.json').read_text())


def restore_member_source(source, root=ROOT):
    """Reverse the recorded reference-only substitutions for provenance checks."""
    for old, new in reversed(member_migration(root)['source_substitutions']):
        source = source.replace(new, old)
    return source


def recorded_file(root, relative):
    """Follow explicit directory migrations for an original repo-relative path.

    This only resolves file locations. Callers must still verify the recorded
    content hash; changed source code does not acquire its historical checksum.
    """
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError('Expected a safe repository-relative POSIX path.')
    root = Path(root).resolve()
    current = str(path)
    for manifest_path in ('eval/yuyao_ding/relocation_manifest.json',
                          'audit/yuyao_ding/workspace_relocation.json'):
        manifest = json.loads((root / manifest_path).read_text())
        mapping = {row['old_path']: row['new_path'] for row in manifest['files']}
        current = mapping.get(current, current)
    result = (root / current).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Recorded path escapes the repository.')
    if not result.is_file():
        raise FileNotFoundError(result)
    return result

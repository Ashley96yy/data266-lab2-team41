"""Resolve recorded repository paths without rewriting original experiment metadata."""
import json
import hashlib
from copy import deepcopy
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def recorded_notebook_snapshot_digest(path):
    """Verify an append-only CPU supplement and recover its prior snapshot hash.

    Historical migration manifests bind the original notebook, rather than a
    later CPU result append. Reconstruct that exact prefix, including metadata,
    and verify its bytes before returning the historical digest.
    """
    raw = Path(path).read_bytes()
    notebook = json.loads(raw)
    proof = notebook.get('metadata', {}).get('data266_cpu_result_append')
    if proof is None:
        return hashlib.sha256(raw).hexdigest()
    count = proof['original_cell_count']
    if type(count) is not int or not 0 < count < len(notebook['cells']):
        raise ValueError('Invalid original notebook prefix.')
    parent = deepcopy(notebook)
    parent['cells'] = parent['cells'][:count]
    del parent['metadata']['data266_cpu_result_append']
    serialization = proof['serialization']
    restored = json.dumps(parent, indent=serialization['indent'],
                          ensure_ascii=serialization['ensure_ascii'])
    if serialization['trailing_newline']:
        restored += '\n'
    actual = hashlib.sha256(restored.encode()).hexdigest()
    if actual != proof['parent_notebook_sha256']:
        raise ValueError('Original notebook cells, outputs or metadata changed.')
    return actual


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

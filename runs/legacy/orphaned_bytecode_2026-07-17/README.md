# Orphaned Python Bytecode Archive

This directory preserves three Python 3.12 bytecode files found under the
ignored `scripts/__pycache__/` directory during the 2026-07-19 cleanup:

| Archive member | SHA-256 |
| --- | --- |
| `p0_contract.cpython-312.pyc` | `a564095fe74b9fba8520b44816d64cadebbeb9ff1c279b3fa79ff09f1287b39a` |
| `p0_exploratory_analysis.cpython-312.pyc` | `1cf603cad6d2924a12eb872db811c89bb45613c078e2a484731588195f5fbd5e` |
| `validate_p0.cpython-312.pyc` | `81db94f19e2e9f63bcea8f9013ff07a51d1e63fbaeaea2efe3fc040503895340` |

The files are stored in `orphaned_python312_bytecode.tar.gz`, whose SHA-256 is:

`aeec30d402090af98d29baf0cc8d820749bb65d9ff4183d485d884d7806308db`

No corresponding `.py` source file exists in the worktree or reachable Git
history. The archive is retained only for forensics. It is not current
AdaptiveServe evidence and must not be executed or treated as reproducible
source code.

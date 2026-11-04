"""Server launcher visibility only; original E2B pipeline/identity stays intact.

Run in the SAME Python process so CLI arguments, exit codes and KeyboardInterrupt
handling in the copied SCW transport retain their ordinary behavior. Only actual
pipeline stages report progress; there is no periodic elapsed-time output.
"""
from pathlib import Path
import runpy
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    entry=root/'main_exp2_paraphrase.py'
    print('Starting E2B: loading the selected-response pipeline.',flush=True)
    old_argv=sys.argv
    inserted=False
    try:
        if str(root) not in sys.path:
            sys.path.insert(0,str(root));inserted=True
        sys.argv=[str(entry),*old_argv[1:]]
        runpy.run_path(str(entry),run_name='__main__')
    finally:
        sys.argv=old_argv
        if inserted:sys.path.remove(str(root))


if __name__=='__main__':main()

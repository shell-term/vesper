"""Separate bounded run records from expert payloads on another filesystem.

These are cooperative free-space admissions, not filesystem quotas. The64GiB
free-space floor stays in reserve_disk; weight imports admit their own SSD space.
"""

from pathlib import Path

from .backbone import LAYERS
from .indexed_store import MAX_INDEX_BYTES
from .resources import GiB, MiB
from .sequence import MAX_CALLS, MAX_EMBEDDINGS, MAX_ROWS
from .source_roots import roots

REPORT_BYTES = 64 * MiB
CASE_BYTES = 4 * MiB


def preparation_file_bounds():
    return {'embedding_cache':MAX_EMBEDDINGS*(10240+4096),
        'original_row_cache':2*MAX_ROWS*(264+4096),
        'frozen_rows':2*(MAX_ROWS*512+4095),
        'frozen_row_manifests':2*12*MiB,
        'runtime_and_flat_indexes':2*len(LAYERS)*MAX_INDEX_BYTES,
        'plan_and_preparation_report':2*REPORT_BYTES,
        'context_scratch_and_reuse_records':16*MiB,
        'live_progress_allowance':128*MiB}


def _device(path):return Path(path).stat().st_dev


def preparation_admission(root):
    if sum(preparation_file_bounds().values())>GiB:raise ValueError('Run-file bounds exceed their declared admission')
    selected=roots(root)
    if selected.mounts and _device(selected.root)!=_device(selected.path(selected.active_bank)):
        return GiB
    return 8*GiB


def capture_admission(plan):
    count=plan['completion']['calls']
    if (type(count) is not int or not 1<=count<=MAX_CALLS or type(plan['calls']) is not list
            or len(plan['calls'])!=count):raise ValueError('Unbounded or incomplete sealed call census')
    return count*CASE_BYTES+REPORT_BYTES+4*MiB

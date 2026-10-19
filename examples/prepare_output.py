"""Fetch only the original output matrix/norm and eight embedding rows."""

import argparse
import json
from pathlib import Path
from vesper.output_assets import prepare

p=argparse.ArgumentParser(description=__doc__); p.add_argument('--metadata',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); prepare(a.metadata,a.output,lambda r:print(json.dumps(r),flush=True))

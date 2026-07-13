 
import copy
from pathlib import Path
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────
 

def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
 
def load_config(test_name, config_root="config"):
    with open(f"{config_root}/constants.yaml") as f:
        constants = yaml.safe_load(f)
    with open(f"{config_root}/tests/test_{test_name}.yaml") as f:
        test = yaml.safe_load(f) or {}
    cfg = _deep_merge(constants, test)
    cfg.setdefault("test_name", test_name)
    return cfg
 
def list_tests(config_root="config"):
    from pathlib import Path
    for p in sorted(Path(config_root + "/tests").glob("test_*.yaml")):
        name = p.stem.replace("test_", "")
        meta = yaml.safe_load(open(p)) or {}
        print(f"  {name:12s}  {meta.get('description', '')}")
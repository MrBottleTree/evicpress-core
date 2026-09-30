"""A store must not report success if canonical Tier 3 rejects the block."""
from pathlib import Path
import tempfile
import unittest

import yaml

from evicpress.config import load_config
from evicpress.manager import EvicPressManager


class StoreRejectionTest(unittest.TestCase):
    def test_oversized_block_is_not_advertised_as_stored(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(__file__).parents[1] / "config" / "config.yaml"
            raw = yaml.safe_load(base.read_text(encoding="utf-8"))
            raw["tier3"]["capacity_bytes"] = 1
            raw["tier3"]["data_dir"] = str(Path(temporary) / "cache")
            raw["quantization"]["enabled"] = False
            cfg_path = Path(temporary) / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            manager = EvicPressManager(load_config(str(cfg_path)))
            success, tier, _quant = manager.store("oversized", b"1234", 0.5)
            self.assertFalse(success)
            self.assertEqual(tier, 0)
            self.assertEqual(manager.lookup("oversized"), (False, 0))


if __name__ == "__main__":
    unittest.main()

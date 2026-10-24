"""Validate committed gallery data and plotting failure behavior."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot_trace", ROOT / "examples/plot_trace.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class ExamplePlots(unittest.TestCase):
    def test_reference_plots(self):
        for model, count in (("trivial", 16), ("dimer", 16), ("nnn", 24)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                source = ROOT / "examples" / model / "expected"
                rows, config = plot.load_trace(source, model)
                self.assertEqual(len(rows), count)
                self.assertEqual(config["seed"], 1234)
                output = Path(tmp) / "trace.png"
                plot.plot_trace(source, model, output)
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                before = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    plot.plot_trace(source, model, output)
                self.assertEqual(output.read_bytes(), before)

    def test_wrong_model(self):
        with self.assertRaises(ValueError):
            plot.load_trace(ROOT / "examples/trivial/expected", "dimer")

    def test_bad_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "metadata.toml").write_text('[parameters]\nL=10\nseed=1234\np_nnn=0.46\n')
            for content in ("sweep,abs_mean_nn,abs_mean_nnn\n1,NaN,0.1\n",
                            "sweep,abs_mean_nn,abs_mean_nnn\n2,0.1,0.2\n1,0.2,0.3\n",
                            "sweep,wrong\n1,0.1\n"):
                (path / "summary.csv").write_text(content)
                with self.assertRaises(ValueError):
                    plot.load_trace(path, "nnn")


if __name__ == "__main__":
    unittest.main()

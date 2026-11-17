import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_start',ROOT/'start.py')
starter=importlib.util.module_from_spec(spec);spec.loader.exec_module(starter)


class PublicLauncherTests(unittest.TestCase):
    def test_fresh_download_starts_without_an_environment_file(self):
        with tempfile.TemporaryDirectory() as temp:
            app=(Path(temp)/'download with spaces').resolve();app.mkdir()
            (app/'run.py').write_text('print("fixture")\n')
            self.assertFalse((app/'.env').exists())
            with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0)) as run,patch('builtins.print'):
                self.assertEqual(starter.launch(app,[]),0)
                self.assertEqual(run.call_args.args[0],[sys.executable,str(app/'run.py'),'--api'])

    def test_check_mode_never_starts_a_server_or_model(self):
        with patch('subprocess.run') as run,patch('builtins.print'):
            self.assertEqual(starter.launch(arguments=['--check']),0)
            run.assert_not_called()

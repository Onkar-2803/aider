from collections import defaultdict
import os
import unittest
from pathlib import Path
import networkx as nx

from aider.dump import dump  # noqa: F401
from aider.io import InputOutput
from aider.repomap import RepoMap
from aider import models
from aider.utils import IgnorantTemporaryDirectory


if __name__ == "__main__":
  print("Run start")
  test_file_ts = "test_file.ts"
  temp_dir = "/Users/onkar/Documents/Code/aider/aider/test_files/"
  other_files = [os.path.join(temp_dir, test_file_ts)]
  io = InputOutput()
  repo_map = RepoMap(root=temp_dir, io=io)
  result = repo_map.get_repo_map([], other_files)
  print(result)

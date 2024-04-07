from collections import defaultdict
import os
import unittest
from pathlib import Path
import networkx as nx

from repomap import RepoMap


if __name__ == "__main__":
  print("Run start")
  test_file_ts = "test_file.ts"
  temp_dir = "/Users/onkar/Documents/Code/aider/Test_Code/test_files"
  other_files = [os.path.join(temp_dir, test_file_ts)]
  repo_map = RepoMap(root=temp_dir)
  result = repo_map.get_repo_map(other_files)
  print(result)

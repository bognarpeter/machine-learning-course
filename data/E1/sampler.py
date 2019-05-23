#!/usr/bin/env python3

import sys

N = 10
idx = 0
for line in sys.stdin:
  if idx%N == 0:
    print(line.strip())
  idx+=1

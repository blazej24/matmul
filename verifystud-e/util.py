import numpy as np
import logging


log = logging.getLogger("util")


def isclose(a, b, rel_tol=1e-04, abs_tol=1e-04):
  return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def read_dense(filename):
  f = open(filename, 'r')
  shape_txt = f.readline()
  shape = shape_txt.split()
  num_rows = int(shape[0])
  num_cols = int(shape[1])
  rows = []
  for row_txt in f:
    row = [float(value) for value in row_txt.split()]
    if len(row) != num_cols:
      raise ValueError("expecting %d cols got %d; line\n%s" % (num_cols, len(row), row_txt))
    rows.append(row)
    if len(rows) == num_rows:
      break
  if len(rows) != num_rows:
    raise ValueError()
  return np.matrix(rows)


def compare_denses(mat_a, mat_b):
  if (mat_a.shape != mat_b.shape):
    return (False, "different shapes: expected %s seen %s" % (str(mat_a.shape), str(mat_b.shape)))
  for row in range(0, mat_a.shape[0]):
    for col in range(0, mat_a.shape[1]):
      a = mat_a[row,col]
      b = mat_b[row,col]
      if not isclose(a,b):
        return (False, "different values at row %d col %d expecting %f got %f" % (row, col, a, b))
  return (True, "")



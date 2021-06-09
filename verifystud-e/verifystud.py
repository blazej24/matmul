''' run instructions on okeanos
move your submission .zip to a directory ../solutions/ , eg., ../solutions/ab123456.zip

bash
module load common/anaconda/3.8
module swap PrgEnv-cray PrgEnv-intel
module load cray-python
module load intel
alias compile="rm -rf build; mkdir build; cd build; cmake ..; make"
salloc --nodes 1 --tasks-per-node 24 --account GC80-33 --time 00:15:00

python3 ./verifystud.py ../solutions/ 'ab123456'
'''

import logging
import os
import subprocess
import signal
import shutil
import zipfile
import itertools
import sys

log = logging.getLogger('verifyprograms')

from util import read_dense, compare_denses

class Alarm(Exception):
  pass


def alarm_handler(signum, frame):
  raise Alarm


def get_run_number(filebase='results'):
  run_number = 1
  while os.path.isfile(filebase+'-run'+str(run_number)+'.csv'):
    run_number += 1
  return run_number


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  zipdir = sys.argv[1]
  students = sys.argv[2].split(' ')

  run_number = get_run_number('../logs/results')
  log.info('run number %d', run_number)
  try:
    os.mkdir('../logs/')
  except FileExistsError:
    pass

  fh = logging.FileHandler('../logs/verifyprograms-run'+str(run_number)+'.log')
  formatter = logging.Formatter('%(asctime)s : %(message)s')
  fh.setFormatter(formatter)
  log.addHandler(fh)

  sink = open('/dev/null', 'w')
  timelimit = 10  # in seconds, should be way less

  basedir = os.getcwd()
  ref_results_dir = basedir + "/../verimat/"

  testdir = '../tests-'+str(run_number)+'/'
  os.makedirs(testdir)

  with open('../logs/results-run'+str(run_number)+'.csv', 'a') as results_stat:
    for solution in students:
      results_stat.flush()
      results_stat.write('\n'+solution+',')
      error_count = 0
      os.chdir(basedir)
      log.info('solution: %s' % solution)

      zip_location = zipdir + solution + '.zip'
      log.info('zip file location: %s', zip_location)
      if not os.path.exists(zip_location):
        log.error('zip file missing')
        results_stat.write('-1, missing')
        continue

      shutil.copy(zip_location, testdir)
      os.mkdir(testdir + solution)
      zip = zipfile.ZipFile(testdir + solution + '.zip')
      try:
        zip.extractall(testdir + solution)
      except zipfile.BadZipFile:
        log.error('zip file misformatted')
        results_stat.write('-1, misformatted')
        continue
      build_dir = testdir + solution + '/build'
      try:
        os.rmdir(build_dir)
      except FileNotFoundError:
        pass
      os.mkdir(build_dir)
      os.chdir(build_dir)
      os.mkdir('outputs')

      retcode = subprocess.call(('cmake', '..'), stdout=sink, stderr=sink)
      if retcode != 0:
        log.error('cmake non-0 retcode')
        results_stat.write('-1, cmake')
        continue
      retcode = subprocess.call('make', stdout=sink, stderr=sink)
      if retcode != 0:
        log.error('make non-0 retcode')
        results_stat.write('-1, make')
        continue

      seeds = [42, 442, ]
      sparse_indexes = { 42 : 0, 442 : 1, }
      exponents = [1, 3]
      sizes = [10, 64, 512]
      comm_avoids = [1, 2, 4]
      mpi_sizes = [4, 8, 16, ]
      algs = ['', '-i']
      with open('errors.txt', 'w') as error_file:
        for (alg, comm_avoid, num_procs, size, seed, exponent
        ) in itertools.product(algs, comm_avoids, mpi_sizes, sizes, seeds, exponents):
          if (num_procs > 4 and size == 10):
            continue
          if comm_avoid >= num_procs:
            continue
          if (num_procs < 8 and size > 64):
            continue
          sparse_index = sparse_indexes[seed]
          sparse_file = ref_results_dir + 'sparse_%05d_%03d' % (size, sparse_index)
          results_filename = ('outputs/mpiresult_ex%d_ca%02d_%05d_%03d' %
                                (exponent, comm_avoid, size, sparse_index))
          with open(results_filename, 'w') as results_file:
            params = ['srun', '--ntasks', str(num_procs), './matrixmul',
                      '-s', str(seed), '-f', sparse_file, '-v',
                      "-c", str(comm_avoid),
                      '-e', str(exponent)]
            if alg != '':
              params.append(alg)
            compact_params = " ".join(params)
            log.info("dir: %s params: %s", solution, compact_params)
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timelimit)
            child = None
            try:
              child = subprocess.Popen(params, stdout = results_file)
              outcode = child.wait()
              signal.alarm(0)
              log.info("subprocess finished")
              if outcode != 0:
                log.info("non-zero exit code %d" % (outcode))
                results_stat.write('-1, outcode, ')
                error_file.write(compact_params+"\n")
                continue
            except FileNotFoundError:
              signal.alarm(0)
              log.info("exec not found")
              results_stat.write('-1, execfile, ')
              error_file.write(compact_params+"\n")
              continue
            except Alarm:
              log.info("timeout!")
              if child is not None:
                log.info("killing the child")
                child.kill()
                log.info("child killed")
                results_stat.write('-1, timeout, ')
                error_file.write(compact_params+"\n")
              continue

          reference_filename = ( ref_results_dir + 'result_%d_%05d_%03d_%05d' %
                                (exponent, size, sparse_index, seed) )
          result = read_dense(results_filename)
          reference = read_dense(reference_filename)
          (is_equal, message) = compare_denses(reference, result)
          if not is_equal:
            results_stat.write('-1, incorrect, ')
            log.info("result is incorrect: %s", message)
            error_file.write(compact_params+"\n")
            continue
          else:
            results_stat.write('1, OK, ')
            log.info("result passed")




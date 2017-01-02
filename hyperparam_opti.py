import sys
import time
import random
import subprocess
import math

sys.path.insert(0, '/home/.../.../LibSVM/')

from svmutil import *

if len(sys.argv) not in (6, 7):
    print 'Usage: python hyperparam_opti.py [grid, random, smac] [cl, re] [training-data-file] [test-data-file] [# rounds] debug'
    exit()
else:
    train_data_file = sys.argv[3]
    test_data_file = sys.argv[4]

debug = True if len(sys.argv) == 7 else False  # debug mode on / off

classific = True if sys.argv[2] == 'cl' else False # classification or regression

train_data = '/home/.../.../data/' + train_data_file
test_data = '/home/.../...data/' + test_data_file

# variables
k = 5  # cross-validation fold, default=5
c_begin = -5  # 2^c_begin
c_end = 15  # 2^c_end
gamma_begin = -15  # 2^gamma_begin
gamma_end = 3  # 2^gamma_end

train_max_size = 10000
test_max_size = 10000

rounds = int(sys.argv[5])
algorithm = sys.argv[1]
print 'Data analysis for ', train_data_file, 'with algorithm', algorithm

# for SMAC algorithm
cmd_smac = '../smac-v2.10.03-master-778/smac --use-instances false --numberOfRunsLimit {0} --pcs-file {1} --algo "python smac_wrapper.py "{2}" {3} {4}" --run-objective QUALITY'.format(
    str(rounds), 'params.pcs', train_data, k, classific)

# scaling data
print 'PROGRESS: Scaling data...'
for t in train_data, test_data:
    p = subprocess.Popen('../svm-scale -l -1 -u 1 "{0}" > "{0}.scale"'.format(t), shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate()
    if debug:
        print out, err
    p.wait()

# read problem
y_train, x_train = svm_read_problem(train_data + '.scale')
if len(y_train) >= train_max_size:
    indices = random.sample(range(len(y_train)), train_max_size)
    x_train = [x_train[i] for i in sorted(indices)]
    y_train = [y_train[i] for i in sorted(indices)]

if algorithm == 'smac':
    # SMAC as optimization algorithm
    pcs_file = open('params.pcs', 'w')
    pcs_file.writelines(['g real [{0:.10f},{1:.10f}] [0.04761904]\n'.format(pow(2, gamma_begin), pow(2, gamma_end)),
                         'c real [{0:.10f},{1:.10f}] [1]'.format(pow(2, c_begin), pow(2, c_end))])
    pcs_file.close()

    print'PROCESS: Running SMAC algorithm...'

    start = time.time()
    p = subprocess.Popen(cmd_smac, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if debug:
        print out, err

    # find substrings in output
    s = out.find('on test set: ')
    accuracy = 100 - float(out[s + 13:s + out[s:].find(', ')])
    find_c = s + out[s:].find('-c') + 4
    c_best = out[find_c:find_c + out[find_c:].find('\'')]
    find_g = s + out[s:].find('-g') + 4
    g_best = out[find_g:find_g + out[find_g:].find('\'')]

    end = time.time()

    print 'RESULT: training time:', end - start, ' seconds'
    print 'RESULT: optimal hyperparameters: c = ', c_best, ', gamma = ', g_best, ' with accuracy: ', accuracy, ' %'

    hyperparams_opt = (c_best, g_best)

elif algorithm in ('grid', 'random'):
    if algorithm == 'grid':
        Cs = []
        Gammas = []
        for r in xrange(0, int(math.sqrt(rounds))):
            c_step = -(c_begin - c_end) / math.sqrt(rounds)
            Cs.append(pow(float(2), (c_begin+c_step*(r+1))))
            gamma_step = -(gamma_begin - gamma_end) / math.sqrt(rounds)
            Gammas.append(pow(float(2), (gamma_begin+gamma_step*(r+1))))
        print 'PROCESS: Running grid search algorithm...'
    else:  # random search
        Cs = []
        Gammas = []
        for r in xrange(0, int(math.sqrt(rounds))):
            Cs.append(pow(float(2), random.uniform(gamma_begin, gamma_end)))
            Gammas.append(pow(float(2), random.uniform(gamma_begin, gamma_end)))
        print 'PROCESS: Runnning random search algorithm...'

    accuracy = []
    hyperparams = []

    start = time.time()
    for c in Cs:
        for gamma in Gammas:
            parameters = '-c {0} -g {1} -v {2} -q'.format(str(c), str(gamma), k)
            curr_acc = svm_train(y_train, x_train, parameters)
            if debug:
                print 'c={0} gamma={1} accuracy={2}'.format(str(c), str(gamma), str(curr_acc))
            accuracy.append(curr_acc)
            hyperparams.append((c, gamma))
    end = time.time()

    if classific:
        max_accuracy = max(accuracy)
        hyperparams_opt = hyperparams[accuracy.index(max_accuracy)]
    else:
        max_accuracy = min(accuracy)
        hyperparams_opt = hyperparams[accuracy.index(max_accuracy)]

    print 'RESULT: training time:', end - start, ' seconds'
    print 'RESULT: number of models trained:', k * len(hyperparams)
    print 'RESULT: optimal hyperparameters: c = ', hyperparams_opt[0], ', gamma= ', hyperparams_opt[1], \
        ' with accuracy: ', max_accuracy, ' %'

start = time.time()
parameters = '-s 0 -t 2 -c {0} -g {1} -q'.format(hyperparams_opt[0], hyperparams_opt[1])

if classific:
    p = subprocess.Popen(
        '../svm-train -c {0} -g {1} "{2}.scale"'.format(hyperparams_opt[0], hyperparams_opt[1], train_data),
        shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    p.wait()
else:
    model = svm_train(y_train, x_train, parameters)

y_test, x_test = svm_read_problem(test_data + '.scale')
if len(y_train) >= train_max_size:
    indices = random.sample(range(len(y_test)), test_max_size)
    x_test = [x_test[i] for i in sorted(indices)]
    y_test = [y_test[i] for i in sorted(indices)]

if classific:
    p = subprocess.Popen(
        '../svm-predict "{1}.scale" "{2}.scale.model" "{1}.predict"'.format(train_data, test_data, train_data_file),
        shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate()
else:
    p_label, p_acc, p_val = svm_predict(y_train, x_train, model)

end = time.time()
if debug:
    print 'output svm-predict: ', out, err
print 'RESULT: test time:', end - start, ' seconds'
if classific:
    print 'RESULT: Best hyperparameters c={0}, best gamma={1} reached an accuracy of {2} % on test data'.format(
        hyperparams_opt[0], hyperparams_opt[1], out[11:18])
else:
    print 'RESULT: Best hyperparameters c={0}, best gamma={1} reached a mean squared error of {2} and a squared correlation coefficient of {3} % on test data'.format(
        hyperparams_opt[0], hyperparams_opt[1], p_acc[1], p_acc[2])

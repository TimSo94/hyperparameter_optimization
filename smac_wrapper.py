#!/usr/bin/python
import sys, subprocess

# For black box function optimization, we can ignore the first 5 arguments.
# The remaining arguments specify parameters using this format: -name value

c = 0
g = 0
v = 0
classific = True if sys.argv[3] == 'True' else False

for i in range(len(sys.argv)-1):
    if (sys.argv[i] == '-c'):
        c = float(sys.argv[i+1].translate(None, '\''))
    elif(sys.argv[i] == '-g'):
        g = float(sys.argv[i+1].translate(None, '\''))

# Compute the branin function:
cmdline = '../svm-train -c %s -g %s -v %s %s' %(str(c), str(g), sys.argv[2], sys.argv[1])
print cmdline
p = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
result, err = p.communicate()
print result
for line in result.split('\n'):
            if str(line).find('Cross') != -1:
                if classific:
                    accuracy = float(100) - float(line.split()[-1][0:-1])
                else:
                    accuracy = float(line.split()[-1][0:-1])


# SMAC has a few different output fields; here, we only need the 4th output:
print "Result of algorithm run: SUCCESS, 0, 0, %f, 0" % (accuracy)


#!/usr/bin/python
# Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
# Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1
#
# This code is licensed under the MIT License.




import sys
import re
import math
import numpy


kernel_consistency = [
    ["mm2meters", "mm2metersKernel"],  
    ["bilateral_filter","bilateralFilterKernel"],
    ["halfSampleRobust","halfSampleRobustImageKernel"],
    ["depth2vertex","depth2vertexKernel"],
    ["vertex2normal","vertex2normalKernel"],
    ["track","trackKernel"],
    ["reduce","reduceKernel"],
    ["integrate","integrateKernel"],
    ["raycast","raycastKernel"],
    ["renderDepth","renderDepthKernel"],
    ["renderLight","renderLightKernel"],
    ["renderTrack","renderTrackKernel"],
    ["renderVolume", "renderVolumeKernel"],
    ["ResetVolume","initVolumeKernel"],
    ["updatePose","updatePoseKernel"]
]

def translateName(n) :
    for variations in kernel_consistency :
        if n in variations :
            return variations[0]
    return n


kernel_log_regex = "([^ ]+)\s([0-9.]+)"


# open files

if len(sys.argv) != 2 :
    print "I need only one parameter, the kernel log file."
    exit (1)

# open log file first
print
print "Kernel-level statistics. Times are in nanoseconds." 
fileref = open(sys.argv[1],'r')
data    = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line

data = {}

for line in lines :
    matching = re.match(kernel_log_regex,line)
    if matching :
        name = translateName(matching.group(1))
        if not  name in data :
            data[name] = []
        data[name].append(float(matching.group(2)))
#    else :
#        print  "Skip SlamBench line : " + line


for variable in sorted(data.keys()) :
    print "%20.20s"% str(variable),
    print "\tCount : %d" % len(data[variable]),
    print "\tMin   : %d" % min(data[variable]),
    print "\tMax   : %d"  % max(data[variable]),
    print "\tMean  : %f" % numpy.mean(data[variable]),
    print "\tTotal : %d" % sum(data[variable])

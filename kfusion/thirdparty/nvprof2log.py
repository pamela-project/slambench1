#!/usr/bin/python
# Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
# Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1
#
# This code is licensed under the MIT License.

import re
import sys

for line in sys.stdin:
    matcher = re.match("([^\s]+)\s+([^\s]+)\s.*\s([^<]+)<?[0-9a-zA-Z_<>=]*\(.*",line)
    if matcher == None :
        continue
    name     = matcher.group(3)
    duration = 0
    if matcher.group(2)[-2:] == "ms" :
        duration = int(float(matcher.group(2)[:-3]) * 1000000)
    else :
        if matcher.group(2)[-2:] == "us":
            duration = int(float(matcher.group(2)[:-3]) * 1000)
        else :
            duration = -1
    print "%s %d" % (name, duration)
    sys.stdout.flush()
    

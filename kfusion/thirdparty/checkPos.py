#!/usr/bin/python
# Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
# Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1
#
# This code is licensed under the MIT License.

import sys
import re
import math
import numpy


kfusion_log_regex  =      "([0-9]+[\s]*)\\t" 
kfusion_log_regex += 8 *  "([0-9.]+)\\t" 
kfusion_log_regex += 3 *  "([-0-9.]+)\\t" 
kfusion_log_regex +=      "([01])\s+([01])" 

nuim_log_regex =      "([0-9]+)" 
nuim_log_regex += 7 * "\\s+([-0-9e.]+)\\s*" 


# open files

if len(sys.argv) != 3 :
    print "I need two parameters, the benchmark log file and the original scene camera position file."
    exit (1)

# open benchmark log file first
print "Get KFusion output data." 
framesDropped = 0
validFrames = 0
lastFrame = -1
untracked = -4;
kfusion_traj = []
fileref = open(sys.argv[1],'r')
data    = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line
headers = lines[0].split("\t")
fulldata = {}
if len(headers) == 15 :
    if headers[14] == "" :
        del headers[14]
if len(headers) != 14 :
    print "Wrong KFusion log  file. Expected 14 columns but found " + str(len(headers))
    exit(1)
for variable in  headers :
    fulldata[variable] = []

for line in lines[1:] :
    matching = re.match(kfusion_log_regex,line)
    if matching :
        dropped =  int( matching.group(1)) - lastFrame - 1
        if dropped>0:     		
    		framesDropped = framesDropped + dropped    		
    		for pad in range(0,dropped) :
    	         kfusion_traj.append( lastValid )     	            
    	         
        kfusion_traj.append( (matching.group(10),matching.group(11),matching.group(12),matching.group(13),1 ) )
        lastValid = (matching.group(10),matching.group(11),matching.group(12),matching.group(13), 0)
        if int(matching.group(13)) == 0 :
            untracked = untracked+1
        validFrames = validFrames +1
        for elem_idx in  range(len(headers)) :
            fulldata[headers[elem_idx]].append(float(matching.group(elem_idx+1)))
        
        lastFrame = int(matching.group(1))
    else :
        #print "Skip KFusion line : " + line
        break

# open benchmark log file first
nuim_traj = []
fileref = open(sys.argv[2],'r')
data    = fileref.read()
fileref.close()
lines = data.split("\n") # remove head + first line
for line in lines :
    matching = re.match(nuim_log_regex,line)
    if matching :
        nuim_traj.append( (matching.group(2),matching.group(3),matching.group(4)) )
    else :
        #print "Skip nuim line : " + line
        break

working_position = min ( len(kfusion_traj) , len(nuim_traj) )
print "KFusion valid frames " + str(validFrames) + ",  dropped frames: " + str(framesDropped)
print "KFusion result        : " + str(len(kfusion_traj)) + " positions."
print "NUIM  result        : " + str(len(nuim_traj)) + " positions."
print "Working position is : " + str(working_position) 
print "Untracked frames: " +str(untracked)
nuim_traj=nuim_traj[0:working_position]
kfusion_traj=kfusion_traj[0:working_position]

print "Shift KFusion trajectory..."

first = nuim_traj[0]
fulldata["ATE"] = []
#ATE_wrt_kfusion does not consider the ATE for frames which were dropped if we are running in non process-every-frame mode
fulldata["ATE_wrt_kfusion"] = []
distance_since_valid=0;
#print "Frame  speed(m/s)   dlv(m) ATE(m)   valid   tracked"
for p in range(working_position) :
    kfusion_traj[p] = (float(kfusion_traj[p][0]) + float(first[0]) , - (float(kfusion_traj[p][1]) + float(first[1]) ) , float(kfusion_traj[p][2]) + float(first[2]), int(kfusion_traj[p][3]),int(kfusion_traj[p][4]) )
    diff = (abs( kfusion_traj[p][0] - float(nuim_traj[p][0])) ,  abs( kfusion_traj[p][1] - float(nuim_traj[p][1] )) ,  abs( kfusion_traj[p][2] - float(nuim_traj[p][2] )) )
    ate=math.sqrt(sum(( diff[0] * diff[0],  diff[1] * diff[1],  diff[2] * diff[2])))
      
    if( p==1 ): 
        lastValid = nuim_traj[p]
    if 1 ==1 :  
        dx = float(nuim_traj[p][0]) - float(lastValid[0])
        dy = float(nuim_traj[p][1]) - float(lastValid[1]) 
        dz = float(nuim_traj[p][2]) - float(lastValid[2])
        distA = math.sqrt((dx*dx) + (dz*dz))
        dist = math.sqrt( (dy*dy) + (distA *distA))
        speed = dist/0.0333
        if (kfusion_traj[p][3]==0):
            tracked = "untracked"
        else:
            tracked = ""
        if (kfusion_traj[p][4]==0):
            valid = "dropped"
        else:
            valid = "-"
        distance_since_valid = distance_since_valid + dist
#        print "%4d %6.6f %6.6f %6.6f %10s %10s"% (p, speed, distance_since_valid, ate, valid, tracked )
        lastValid =  nuim_traj[p]
        if kfusion_traj[p][4]==1:
            distance_since_valid= 0

    if (kfusion_traj[p][4] == 1 ):
        fulldata["ATE_wrt_kfusion"].append(ate)        
    fulldata["ATE"].append(ate)                
        
#print "The following are designed to enable easy macchine readability of key data" 
#print "MRkey:,logfile,ATE,computaion,dropped,untracked"
#print ("MRdata:,%s,%6.6f,%6.6f,%d,%d") % ( sys.argv[1], numpy.mean(fulldata["ATE"]), numpy.mean(fulldata["computation"]), framesDropped, untracked)

print "\nA detailed statistical analysis is provided."
print "Runtimes are in seconds and the absolute trajectory error (ATE) is in meters." 
print "The ATE measure accuracy, check this number to see how precise your computation is."
print "Acceptable values are in the range of few centimeters."

for variable in sorted(fulldata.keys()) :
    if "X" in variable or "Z" in variable or "Y" in variable or "frame" in variable  or "tracked" in variable      or "integrated" in variable  :  
        continue

    if (framesDropped == 0)  and (str(variable) == "ATE_wrt_kfusion"):
        continue
		
    print "%20.20s" % str(variable),
    print "\tMin : %6.6f" % min(fulldata[variable]),
    print "\tMax : %0.6f"  % max(fulldata[variable]),
    print "\tMean : %0.6f" % numpy.mean(fulldata[variable]),
    print "\tTotal : %0.8f" % sum(fulldata[variable])

#first2 = []S
#derive = []

#for row_idx in range(len(rows1)) :
#    col1 = rows1[row_idx].split("\t")
#    col2 = rows2[row_idx].split(" ")
#    v1 = col1[8:11]
#    v2 = col2[1:4]
#    if first2 == [] :
#        first2 = v2
#    v1 = [float(v1[0]) + float(first2[0]) , - (float(v1[1]) + float(first2[1]) ) , float(v1[2]) + float(first2[2]) ]
#    derive.append([abs(float(v1[0]) - float(v2[0])) , abs (float(v1[1]) - float(v2[1]) ) , abs(float(v1[2]) - float(v2[2])) ])

#maxderive = reduce(lambda a,d:  [max(a[0] , d[0]),max(a[1] , d[1]),max(a[2] , d[2])], derive, [0.0,0.0,0.0])
#minderive = reduce(lambda a,d:  [min(a[0] , d[0]),min(a[1] , d[1]),min(a[2] , d[2])], derive, [1000000,10000000.0,10000000.0])
#total = map(lambda x: x/len(rows1), reduce(lambda a,d: [a[0] + d[0],a[1] + d[1],a[2] + d[2]], derive, [0.0,0.0,0.0])) 

#print "Min derivation : " + str(min(minderive))
#print "Max derivation : " + str(max(maxderive))
#print "Average derivation : " + str(total[0])

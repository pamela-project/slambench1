
#### ICL-NUIM TRAJECTORIES PARAMETERS  ####
0 = -s 5.0 -p 0.34,0.5,0.24 -z 4 -c 2 -r 2 -k 481.2,480,320,240
1 = -s 5.0 -p 0.485,0.5,0.55 -z 4 -c 2 -r 2 -k 481.2,480,320,240
2 = -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 2 -k 481.2,480,320,240
3 = -s 5.0 -p 0.2685,0.5,0.4 -z 4 -c 2 -r 2 -k 481.2,480,320,240

all : 
	mkdir -p build/
	cd build/ && cmake ..
	$(MAKE) -C build  $(MFLAGS) 

#### DATA SET GENERATION ####

living_room_traj%_loop.raw : living_room_traj%_loop
	if test -x ./build/kfusion/thirdparty/scene2raw ; then echo "..." ; else echo "do make before"; false ; fi
	./build/kfusion/thirdparty/scene2raw living_room_traj$(*F)_loop living_room_traj$(*F)_loop.raw

living_room_traj%_loop : 
	mkdir $@
	cd $@ ; wget http://www.doc.ic.ac.uk/~ahanda/$@.tgz; tar xzf $@.tgz 

livingRoom%.gt.freiburg : 
	echo  "Download ground truth trajectory..."
	if test -x $@ ; then echo "Done" ; else wget http://www.doc.ic.ac.uk/~ahanda/VaFRIC/$@ ; fi


#### LOG GENERATION ####

%.opencl.log  : living_room_traj%_loop.raw livingRoom%.gt.freiburg
	LD_PRELOAD=./build/kfusion/thirdparty/liboclwrapper.so ./build/kfusion/kfusion-benchmark-opencl $($(*F)) -i  living_room_traj$(*F)_loop.raw -o benchmark.$@ 2> oclwrapper.$@
	cat  oclwrapper.$@ |grep -E ".+ [0-9]+ [0-9]+ [0-9]+" |cut -d" " -f1,4 >   kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@   >> resume.$@

%.cpp.log  :  living_room_traj%_loop.raw livingRoom%.gt.freiburg
	KERNEL_TIMINGS=1 ./build/kfusion/kfusion-benchmark-cpp $($(*F)) -i  living_room_traj$(*F)_loop.raw -o  benchmark.$@ 2> kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@   >> resume.$@

%.openmp.log  :  living_room_traj%_loop.raw livingRoom%.gt.freiburg
	KERNEL_TIMINGS=1 ./build/kfusion/kfusion-benchmark-openmp $($(*F)) -i  living_room_traj$(*F)_loop.raw -o  benchmark.$@ 2> kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@   >> resume.$@

%.cuda.log  : living_room_traj%_loop.raw livingRoom%.gt.freiburg
	nvprof --print-gpu-trace ./build/kfusion/kfusion-benchmark-cuda $($(*F)) -i  living_room_traj$(*F)_loop.raw -o  benchmark.$@ 2> nvprof.$@ || true
	cat  nvprof.$@ | kfusion/thirdparty/nvprof2log.py >   kernels.$@
	./kfusion/thirdparty/checkPos.py benchmark.$@  livingRoom$(*F).gt.freiburg > resume.$@
	./kfusion/thirdparty/checkKernels.py kernels.$@   >> resume.$@


#### GENERAL GENERATION ####

clean :
	rm -rf build
cleanall : 
	rm -rf build
	rm -rf living_room_traj*_loop livingRoom*.gt.freiburg living_room_traj*_loop.raw
	rm -f *.log 


.PHONY : clean bench test all validate

.PRECIOUS: living_room_traj%_loop livingRoom%.gt.freiburg living_room_traj%_loop.raw


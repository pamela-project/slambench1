/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <stdio.h>
#include <iostream>
#include <perfstats.h>
#include "PowerMonitor.h"

struct PapiPower {
	double packagePower;
	double pp1Power;
	double dramPower;
	double pp0Power;
} papiReading;
#ifndef PAPI_MONITORING
/* Dummy funtion definitions for compilation
*/
	int papi_init() { return 0;}
	int papi_start() { return 0;}
	int papi_stop() { return 0;}
	int papi_read() { return 0;}
#endif
#ifdef PAPI_MONITORING
#include <cstring>
#include <papi.h>

int papi_init();
int papi_start();
int papi_stop();
int papi_read();
/* PAPI is based on the rapl_basic in <papi>/src/components/rapl/tests
*/
// Sorry just trying to get it to compile ...
#define MAX_RAPL_EVENTS 64
    long long before_time,after_time;
    int retval;
    int EventSet = PAPI_NULL;
    char event_names[MAX_RAPL_EVENTS][PAPI_MAX_STR_LEN];
    long long *values;
    char units[MAX_RAPL_EVENTS][PAPI_MIN_STR_LEN];
    int data_type[MAX_RAPL_EVENTS];
    int num_events=0;

    double elapsed_time;

#endif
PowerMonitor::PowerMonitor() {
	powerA7 = NULL;
	powerA15 = NULL;
	powerGPU = NULL;
	powerDRAM = NULL;
	sensingMethod = ODROID;
	if (enableSensor(SENSOR_A7) == 0) {
		enableSensor(SENSOR_A15);
		enableSensor(SENSOR_GPU);
		enableSensor(SENSOR_DRAM);
		sensingMethod = ODROID;
		std::cerr << "Power sensors: ODROID" << std::endl;
#ifdef PAPI_MONITORING
	} else {
		/*************************************************
			Insert code to attempt to initialise for PAPI
			here please
		*****************************************************/
		if(papi_init() == 0) {
			/*
			Dependent on what is required for initialisation of CUDA? and when it becomes
			available OpenCL for GPU PAPI then we can either insert the initialisation
			code into papi_init OR to create different initialisation functions ...
			*/
			sensingMethod = PAPI_CPU;
			std::cerr << "Power sensors: PAPI CPU" << std::endl;
		}

	}
#else
	} else {
	  sensingMethod = NONE;
	}
#endif
	#ifdef FORCE_ON
		sensingMethod = DUMMY;
	#endif
}
bool PowerMonitor::isActive() {
	if (sensingMethod == NONE)
		return (false);
	else
		return (true);
}
double PowerMonitor::start() {
	startTime = powerStats.get_time();

	/*
		PAPI might be sensible to reset the counters
		how do we avoid overflow etc
	*/
	if(sensingMethod == PAPI_CPU || sensingMethod == PAPI_ALL) {
		papi_start();
	} 
	if(sensingMethod == PAPI_CUDA || sensingMethod == PAPI_ALL) {
		papi_start();
		// NOT sure if we will need separate initialisation.
		// so we might need to modify papi_init etc
	}
	return (startTime);
}
double PowerMonitor::sample() {
	double time = 0;
	if (sensingMethod == ODROID) {
		double a15 = getPower(SENSOR_A15);
		time = powerStats.sample("Sample_time",
				powerStats.get_time() - startTime, PerfStats::TIME);
		if (powerA7)
			powerStats.sample("Power_A7", getPower(SENSOR_A7),
					PerfStats::POWER);
			powerStats.sample("Power_A15", a15, PerfStats::POWER);
		if (powerGPU)
			powerStats.sample("Power_GPU", getPower(SENSOR_GPU),
					PerfStats::POWER);
		if (powerDRAM)
			powerStats.sample("Power_DRAM", getPower(SENSOR_DRAM),
					PerfStats::POWER);
	}
	if (sensingMethod == PAPI_CPU) {
		papi_read();
		powerStats.sample("package",papiReading.packagePower, PerfStats::POWER);
		powerStats.sample("pp1",papiReading.pp1Power,PerfStats::POWER);
		powerStats.sample("dram",papiReading.dramPower,PerfStats::POWER);
		powerStats.sample("pp0",papiReading.pp0Power,PerfStats::POWER);
	}
	if (sensingMethod == DUMMY) {		
		time = powerStats.sample("Sample_time",powerStats.get_time() - startTime, PerfStats::TIME);
		powerStats.sample("CPU", 1.5, PerfStats::POWER);
		powerStats.sample("GPU", 3.5, PerfStats::POWER);
					
	}
	return (time);
}
PowerMonitor::~PowerMonitor() {
	if (powerA7)
		fclose(powerA7);
	if (powerA15)
		fclose(powerA15);
	if (powerGPU)
		fclose(powerGPU);
	if (powerDRAM)
		fclose(powerDRAM);

	if(sensingMethod == PAPI_CPU || sensingMethod == PAPI_CUDA || sensingMethod == PAPI_ALL)	{
		papi_stop();
	}
}

float PowerMonitor::getPower(Sensor sensor) {
	FILE *tmp = NULL;
	float power;
	if (sensingMethod == ODROID) {
		switch (sensor) {
		case SENSOR_A7:
			tmp = powerA7;
			break;
		case SENSOR_A15:
			tmp = powerA15;
			break;
		case SENSOR_GPU:
			tmp = powerGPU;
			break;
		case SENSOR_DRAM:
			tmp = powerDRAM;
			break;
		}
		if (tmp) {
			rewind(tmp);
			fscanf(tmp, "%f\n", &power);
			return (power);
		}
	}
	return 0;
}

int PowerMonitor::enableSensor(Sensor sensor) {
	char enableFile[256];
	FILE *tmp;
	bool done = false;
	if (sensingMethod == ODROID) {
		for (int dn = 1; dn < 5; dn++) {
			sprintf(enableFile, "/sys/bus/i2c/drivers/INA231/%d-00%d/enable", dn,
				sensor);

			if ((tmp = fopen(enableFile, "a"))!= 0) {
				fprintf(tmp, "1\n");
				fclose(tmp);

				sprintf(enableFile, "/sys/bus/i2c/drivers/INA231/%d-00%d/sensor_W",
					dn, sensor);
				if ((tmp = fopen(enableFile, "r")) != 0) {
					switch (sensor) {
					case SENSOR_A7:
						powerA7 = tmp;
						break;
					case SENSOR_A15:
						powerA15 = tmp;
						break;
					case SENSOR_GPU:
						powerGPU = tmp;
						break;
					case SENSOR_DRAM:
						powerDRAM = tmp;
						break;
					}
					return (0);
				}
			}
	  	}
	} 
	return (1);
}
#ifdef PAPI_MONITORING
#define MAX_RAPL_EVENTS 64

int papi_init ()
{

    int cid,rapl_cid=-1,numcmp;
    int code;
    int r,i, do_wrap = 0;
    const PAPI_component_info_t *cmpinfo = NULL;
    PAPI_event_info_t evinfo;

     retval = PAPI_library_init( PAPI_VER_CURRENT );
     if ( retval != PAPI_VER_CURRENT ) {
	std::cerr << "PAPI_library_init failed " << retval << std::endl;
	
     }


     numcmp = PAPI_num_components();

     for(cid=0; cid<numcmp; cid++) {

	if ( (cmpinfo = PAPI_get_component_info(cid)) == NULL) {
	   std::cerr << "PAPI_get_component_info failed " << 0;
	}

	if (strstr(cmpinfo->name,"rapl")) {

	   rapl_cid=cid;


           if (cmpinfo->disabled) {
		 std::cerr <<"RAPL component disabled: " <<  cmpinfo->disabled_reason;
		 return -2;
           }
	   break;
	}
     }

     /* Component not found */
     if (cid==numcmp) {
	std::cerr << "PAPI Rapl components not found\n";
	return -1;
     }

     /* Create EventSet */
     retval = PAPI_create_eventset( &EventSet );
     if (retval != PAPI_OK) {
	std::cerr << "PAPI_create_eventset FAIL ";
	return -3;
     }

     /* Add all events */

     code = PAPI_NATIVE_MASK;

     r = PAPI_enum_cmp_event( &code, PAPI_ENUM_FIRST, rapl_cid );

     while ( r == PAPI_OK ) {

        retval = PAPI_event_code_to_name( code, event_names[num_events] );
	if ( retval != PAPI_OK ) {
	   std::cerr << "Error translating " << code <<  " PAPI_event_code_to_name" << retval;
	   return -4;
	}

	retval = PAPI_get_event_info(code,&evinfo);
	if (retval != PAPI_OK) {
	     std::cerr << "Error getting event info " << retval;
	     return -5;
	}

	strncpy(units[num_events],evinfo.units,sizeof(units[0])-1);
	// buffer must be null terminated to safely use strstr operation on it below
	units[num_events][sizeof(units[0])-1] = '\0';

	data_type[num_events] = evinfo.data_type;

        retval = PAPI_add_event( EventSet, code );
        if (retval != PAPI_OK) {
	  break; /* We've hit an event limit */
	}
	num_events++;
  	      
        r = PAPI_enum_cmp_event( &code, PAPI_ENUM_EVENTS, rapl_cid );
     }

     values= (long long int *) calloc(num_events,sizeof(long long));
     if (values==NULL) {
	std::cerr << "Failed to allocate  memory  in PAPI initialisation";
	return -6;
     }
     return 0; // That means OK
}
int papi_start() {

     /* Start Counting */
     before_time=PAPI_get_real_nsec();
     retval = PAPI_start( EventSet);
     if (retval != PAPI_OK) {
	std::cerr << "ERROR: PAPI_start() " << retval;
     }
     return retval;
}
int papi_read() {
     /* Stop Counting */
     after_time=PAPI_get_real_nsec();
     retval = PAPI_stop( EventSet, values);
     if (retval != PAPI_OK) {
	std:: cerr <<  "ERROR: PAPI_stop() "<< retval;
		return -1;
     }

     elapsed_time=((double)(after_time-before_time))/1.0e9;

     if (1) {
        //printf("\nStopping measurements, took %.3fs, gathering results...\n\n", elapsed_time);

		//printf("Scaled energy measurements:\n");

		for(int i=0;i<num_events;i++) {
		   if (strstr(units[i],"nJ")) {

			  /*printf("EVENT %d %-40s%12.6f J\t(Average Power %.1fW)\n",i,
				event_names[i],
				(double)values[i]/1.0e9,
				((double)values[i]/1.0e9)/elapsed_time);
			*/
		/*
		for a multisocket machine this will need upgrading, for now we will print this
		on the assumption that we are a single socket machine
		*/
		 	if(strstr(event_names[i],"PACKAGE_ENERGY:PACKAGE0")) papiReading.packagePower = ((double)values[i]/1.0e9)/elapsed_time;
			if(strstr(event_names[i],"PP1_ENERGY:PACKAGE0")) papiReading.pp1Power = ((double)values[i]/1.0e9)/elapsed_time;
			if(strstr(event_names[i],"DRAM_ENERGY:PACKAGE0")) papiReading.dramPower = ((double)values[i]/1.0e9)/elapsed_time;
			if(strstr(event_names[i],"PP0_ENERGY:PACKAGE0")) papiReading.pp0Power = ((double)values[i]/1.0e9)/elapsed_time;
		   }
		}
		//std::cerr << papiReading.packagePower << " " << papiReading.pp1Power << " " << papiReading.dramPower << " " << papiReading.pp0Power << std::endl;

		/*printf("\n");
		printf("Energy measurement counts:\n");

		for(int i=0;i<num_events;i++) {
		   if (strstr(event_names[i],"ENERGY_CNT")) {
			  printf("%-40s%12lld\t%#08llx\n", event_names[i], values[i], values[i]);
		   }
		}

		printf("\n");
		printf("Scaled Fixed values:\n");

		for(int i=0;i<num_events;i++) {
		   if (!strstr(event_names[i],"ENERGY")) {
			 if (data_type[i] == PAPI_DATATYPE_FP64) {

				 union {
				   long long ll;
				   double fp;
				 } result;

				result.ll=values[i];
				printf("%-40s%12.3f %s\n", event_names[i], result.fp, units[i]);
			  }
		   }
		}
	
		printf("\n");
		printf("Fixed value counts:\n");

		for(int i=0;i<num_events;i++) {
		   if (!strstr(event_names[i],"ENERGY")) {
			  if (data_type[i] == PAPI_DATATYPE_UINT64) {
				printf("%-40s%12lld\t%#08llx\n", event_names[i], values[i], values[i]);
			  }
		   }
		}
		*/

     }

#ifdef WRAP_TEST
	double max_time;
	unsigned long long max_value = 0;
	int repeat;
	
	for(i=0;i<num_events;i++) {
		if (strstr(event_names[i],"ENERGY_CNT")) {
			if (max_value < (unsigned) values[i]) {
				max_value = values[i];
		  	}
		}
	}
	max_time = elapsed_time * (0xffffffff / max_value);
	printf("\n");
	printf ("Approximate time to energy measurement wraparound: %.3f sec or %.3f min.\n", 
		max_time, max_time/60);
	
	if (do_wrap) {
		 printf ("Beginning wraparound execution.");
	     /* Start Counting */
		 before_time=PAPI_get_real_nsec();
		 retval = PAPI_start( EventSet);
		 if (retval != PAPI_OK) {
			test_fail(__FILE__, __LINE__, "PAPI_start()",retval);
		 }

		 /* Run test */
		repeat = (int)(max_time/elapsed_time);
		for (i=0;i< repeat;i++) {
			run_test(1);
			printf("."); fflush(stdout);
		}
		printf("\n");
	 
		 /* Stop Counting */
		 after_time=PAPI_get_real_nsec();
		 retval = PAPI_stop( EventSet, values);
		 if (retval != PAPI_OK) {
			test_fail(__FILE__, __LINE__, "PAPI_stop()",retval);
		 }

		elapsed_time=((double)(after_time-before_time))/1.0e9;
		printf("\nStopping measurements, took %.3fs\n\n", elapsed_time);

		printf("Scaled energy measurements:\n");

		for(i=0;i<num_events;i++) {
		   if (strstr(units[i],"nJ")) {

			  printf("%-40s%12.6f J\t(Average Power %.1fW)\n",
				event_names[i],
				(double)values[i]/1.0e9,
				((double)values[i]/1.0e9)/elapsed_time);
		   }
		}
		printf("\n");
		printf("Energy measurement counts:\n");

		for(i=0;i<num_events;i++) {
		   if (strstr(event_names[i],"ENERGY_CNT")) {
			  printf("%-40s%12lld\t%#08llx\n", event_names[i], values[i], values[i]);
		   }
		}
	}

#endif
	return 0;
}
int papi_stop() {
     /* Done, clean up */
     retval = PAPI_cleanup_eventset( EventSet );
     if (retval != PAPI_OK) {
	 std::cerr << "ERROR: PAPI_cleanup_eventset()" << retval;
     }

     retval = PAPI_destroy_eventset( &EventSet );
     if (retval != PAPI_OK) {
         std::cerr << "ERROR: PAPI_destroy_eventset() " << retval;
     }
        
		
     return 0;
}
#endif

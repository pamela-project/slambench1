
find_path(SYCL_DRIVER syclone
  ENV SYCLONE
  PATHS tools/driver_script
)

if(SYCL_DRIVER)
	set(SYCL_FOUND TRUE)
endif()

# Generate appropriate messages
if(SYCL_FOUND)
  if(NOT SYCL_FIND_QUIETLY)
    message("-- Found SYCL driver: ${SYCL_DRIVER}")
  endif(NOT SYCL_FIND_QUIETLY)
else(SYCL_FOUND)
  if(SYCL_FIND_REQUIRED)
	  message(FATAL_ERROR "-- Could NOT find SYCL (missing: SYCL_DRIVER)")
  endif(SYCL_FIND_REQUIRED)
endif(SYCL_FOUND)

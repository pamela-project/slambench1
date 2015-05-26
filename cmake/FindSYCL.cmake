
find_path(SYCL_DRIVER_SCRIPT syclone)  # Ensure syclone exe script is in PATH

if    (SYCL_DRIVER_SCRIPT)
  set (SYCL_FOUND TRUE)
endif (SYCL_DRIVER_SCRIPT)

if    (SYCL_FOUND)
  message ("-- Found SYCL driver: ${SYCL_DRIVER_SCRIPT}/syclone")
else  (SYCL_FOUND)
  message ("-- Could NOT find SYCL driver (missing: SYCL_DRIVER_SCRIPT)")
endif (SYCL_FOUND)

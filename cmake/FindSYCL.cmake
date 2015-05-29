
find_path(SYCL_CC syclone)  # Ensure syclone exe script is in PATH

if    (SYCL_CC)
  set (SYCL_FOUND TRUE)
endif (SYCL_CC)

if    (SYCL_FOUND)
  message ("-- Found SYCL driver: ${SYCL_CC}/syclone")
else  (SYCL_FOUND)
  message ("-- Could NOT find SYCL exe driver script (missing: SYCL_CC)")
endif (SYCL_FOUND)

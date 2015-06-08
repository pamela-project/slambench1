
find_path(SYCL_CC_PATH syclcc)  # Ensure syclcc exe script is in PATH

if    (SYCL_CC_PATH AND "$ENV{CXX}" MATCHES "syclcc")
  set (SYCL_FOUND TRUE)
endif ()

if    (SYCL_FOUND)
  message ("-- Found SYCL driver: ${SYCL_CC_PATH}/syclcc")
else  (SYCL_FOUND)
  message ("-- Could NOT find SYCL driver script (missing: SYCL_CC_PATH)")
endif (SYCL_FOUND)

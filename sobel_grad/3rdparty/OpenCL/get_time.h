#ifndef __GET_TIME__
#define __GET_TIME__

#include <sys/time.h>
#include "debug_print.h"

#if DEBUG 

#define TINIT \
    struct timeval start, stop; \
    double elapsed_time;

#define TIC \
    gettimeofday(&start, NULL); 

#define TOC(x) \
    gettimeofday(&stop, NULL); \
    elapsed_time = (stop.tv_sec - start.tv_sec) * 1000. + \
    (stop.tv_usec - start.tv_usec) / 1000.; \
    Debug_info("%s taskes %f ms\r\n", x, elapsed_time);

#else

#define TINIT 
#define TIC 
#define TOC(x) 

#endif

#endif

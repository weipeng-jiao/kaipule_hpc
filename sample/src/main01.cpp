#include<iostream>

#include <time.h>
#if _WIN32
#include <intrin.h>
#include <Windows.h>
#endif
#if __linux__
#include <immintrin.h>
#include <sys/time.h>
#endif


#include <random>
#include <time.h>
 
using std::default_random_engine;
using std::uniform_real_distribution;

using namespace std;

#if(1)
#define Print_TAG		"HPC_AVX2"
#define Debug_info(fmt,...)      {printf("%s info    -[fun:%-25.25s line:%05d:	]:",Print_TAG,__FUNCTION__,__LINE__);printf(fmt,##__VA_ARGS__);}

#if _WIN32

#define TINIT \
	LARGE_INTEGER cpu_freqence; \
	LARGE_INTEGER start; \
	LARGE_INTEGER end; \
	double run_time = 0.0; \
	QueryPerformanceFrequency(&cpu_freqence);

#define TIC \
  QueryPerformanceCounter(&start);

#define TOC(x) \
	QueryPerformanceCounter(&end); \
	run_time = (((end.QuadPart - start.QuadPart) * 1000.0f) / cpu_freqence.QuadPart); \
    Debug_info("%s taskes %f ms\r\n", x, run_time);

#elif __linux__
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


/* 计算pi */

//正常的逐个累加运算
double compute_pi_naive(size_t dt) {
 
	double pi = 0.0;
 
	double delta = 1.0 / dt;
 
	for (size_t i = 0; i < dt; i++) {
 
		double x = (double)i / dt;
 
		pi += delta / (1 + x * x);
	}
	return pi * 4.0;
}


double compute_pi_sim256(size_t dt) {
 
	alignas(32) double pi = 0.0;
 
	alignas(32) double delta = 1.0 / dt;
 
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4;
 
	ymm0 = _mm256_set1_pd(1.0);
 
	ymm1 = _mm256_set1_pd(delta);
 
	ymm2 = _mm256_set_pd(0.0, delta, delta * 2, delta * 3);
 
	ymm4 = _mm256_setzero_pd();
 
	for (int i = 0; i < dt - 4; i += 4) {
 
		ymm3 = _mm256_set1_pd(i*delta);
 
		ymm3 = _mm256_add_pd(ymm3, ymm2);// 构造 x
 
		ymm3 = _mm256_mul_pd(ymm3, ymm3);// 构造 x^2
 
		ymm3 = _mm256_add_pd(ymm0, ymm3);// 构造 1 + x^2
 
		ymm3 = _mm256_div_pd(ymm1, ymm3);// 构造 delta / ( 1 + x^2 )
 
		ymm4 = _mm256_add_pd(ymm4, ymm3);// 叠加结果
	}
	
	alignas(32) double tmp[4];
	_mm256_store_pd(tmp, ymm4);
 
	pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
 
	return pi * 4.0;
}


int main() {//test_cal_pi
 
    TINIT;
 
	size_t dt = 134217728;
 
	double result1, result2;
    
    //普通函数计时
    TIC;
	result1 = compute_pi_naive(dt);
    TOC("naive")
	cout << "naive: " << result1 << endl;
    
    //simd256计时
    TIC;
	result2 = compute_pi_sim256(dt);
    TOC("simd256")
	cout << "simd256: " << result2 << endl;
	
 
	return 0;
}

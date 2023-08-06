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


float MathMax(const float *input, int size)
{
	float maxVal = input[0];
	for (int i = 1; i < size; i++)
	{
		maxVal = maxVal > input[i] ? maxVal : input[i];
	}
 
	return maxVal;
}
 
float SSEMax(const float *input, int size)
{
	if (input == nullptr)
	{
		printf("input data is null\n");
		return -1;
	}
	int nBlockWidth = 4;
	int cntBlock = size / nBlockWidth;
	int cntRem = size % nBlockWidth;
 
	__attribute__((__aligned__(16))) float output[4];
	__m128 loadData;
	const float *p = input;
 
	__m128 maxVal = _mm_load_ps(p);
	p += nBlockWidth;
 
	for (int i = 1; i < cntBlock; i++)
	{
		loadData = _mm_load_ps(p);
		maxVal = _mm_max_ps(maxVal, loadData);
 
		p += nBlockWidth;
	}
	_mm_store_ps(output, maxVal);
 
	float maxVal_ = output[0];
	for (int i = 1; i < 4; i++)
	{
		maxVal_ = maxVal_ > output[i] ? maxVal_ : output[i];
	}
	for (int i = 0; i < cntRem; i++)
	{
		maxVal_ = maxVal_ > p[i] ? maxVal_ : p[i];
	}
 
	return maxVal_;
}
 
float AVXMax(const float *input, int size)
{
	if (input == nullptr)
	{
		printf("input data is null\n");
		return -1;
	}
	int nBlockWidth = 8;
	int cntBlock = size / nBlockWidth;
	int cntRem = size % nBlockWidth;

	__attribute__((__aligned__(32))) float output[8];
	__m256 loadData;
	const float *p = input;

	__m256 maxVal = _mm256_loadu_ps(p);

	p += nBlockWidth;

	for (int i = 1; i < cntBlock; i++)
	{
		loadData = _mm256_loadu_ps(p);
		maxVal = _mm256_max_ps(maxVal, loadData);
 
		p += nBlockWidth;
	}
	_mm256_store_ps(output, maxVal);

 

	float maxVal_ = output[0];
	for (int i = 1; i < 8; i++)
	{
		maxVal_ = maxVal_ > output[i] ? maxVal_ : output[i];
	}
	for (int i = 0; i < cntRem; i++)
	{
		maxVal_ = maxVal_ > p[i] ? maxVal_ : p[i];
	}
	
	return maxVal_;
}




 
int main(int argc, char* argv[])
{
	TINIT;
    int size = 58;
	float *input = (float *)malloc(sizeof(float) * size);
 
	default_random_engine e;
	uniform_real_distribution<float> u(0, 3); //随机数分布对象 
	for (int i = 0; i < size; i++)
	{
		input[i] = u(e);
		printf("%f ", input[i]);
		if ((i + 1) % 8 == 0)
			printf("\n");
	}
	printf("\n");
 
	int cntLoop = 100000000;
	float org;
    TIC;
	for (int i = 0; i < cntLoop; i++)
		org = MathMax(input, size);
    TOC("org");
	printf("org = %f\t", org);

 

	float sse;
    TIC;
	for (int i = 0; i < cntLoop; i++)
		sse = SSEMax(input, size);
    TOC("sse");
	printf("sse = %f\t", sse);

 

	float avx;
    TIC;
	for (int i = 0; i < cntLoop; i++)
		avx  = AVXMax(input, size);
    TOC("avx");
	printf("avx = %f\t", avx);

	free(input);
 
	return 0;
}
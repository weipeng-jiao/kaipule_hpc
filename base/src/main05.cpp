#include <stdio.h>
#include <iostream>
#include <time.h>
#if _WIN32
#include <intrin.h>
#include <Windows.h>
#endif
#if __linux__
#include <immintrin.h>
#include <sys/time.h>
#endif

using namespace std;

bool isAligned(void* data, int alignment)
{
	// 又是一个经典算法, 参见<Hacker's Delight>  
	return ((uintptr_t)data & (alignment - 1)) == 0;
}


int main()
{
    float* result=NULL;
    int *result_i=NULL;
    float A[16]={0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f,1.1f,1.2f,1.3f,1.4f,1.5f};
    alignas(16) float B[16]={0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f,1.1f,1.2f,1.3f,1.4f,1.5f};
    int C[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    __m128 a,b,c,h;
    __m128i d,e,f;


    if (isAligned(&B[0], 16)) {
		std::cout << "isAligned\n";
	}

    // load
    result=(float*)&A;
    printf("[ref] input:        %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    d=_mm_loadu_si128((__m128i*)C);
    result_i=(int*)&d;
    printf("[sse] _mm_loadu_si128: %d  %d  %d  %d \r\n",result_i[0],result_i[1],result_i[2],result_i[3]);

    e=_mm_lddqu_si128((__m128i*)&C[4]);
    result_i=(int*)&e;
    printf("[sse] _mm_lddqu_si128: %d  %d  %d  %d \r\n",result_i[0],result_i[1],result_i[2],result_i[3]);


    f=_mm_alignr_epi8(d,e,2*sizeof(int));// high <- low
    result_i=(int*)&f;
    printf("[sse] _mm_alignr_epi8: %d  %d  %d  %d \r\n",result_i[0],result_i[1],result_i[2],result_i[3]);
 

    a=_mm_loadu_ps(A);
    result=(float*)&a;
    printf("[sse] _mm_loadu_ps: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_loadu_ps(&A[4]);
    result=(float*)&b;
    printf("[sse] _mm_loadu_ps: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    f=_mm_alignr_epi8(a,b,sizeof(int));// high <- low  <-(b a) 
    result=(float*)&f;
    printf("[sse] _mm_alignr_epi8: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);


    c=_mm_cmpgt_ps(a,b); // a>b
    result=(float*)&c;
    printf("[sse] _mm_cmpgt_ps: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);
    
    h=_mm_blendv_ps(a,b,c); // c?b:a
    result=(float*)&h;
    printf("[sse] _mm_blendv_ps: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);



}
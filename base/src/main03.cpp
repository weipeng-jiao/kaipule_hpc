#include <stdio.h>
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
    int C[16]={0,1,2,3,4,5,6,8,9,10,11,12,13,14,15};

    __m128 a,b,d;
    __m128i c,e;


    if (isAligned(&B[0], 16)) {
		std::cout << "isAligned\n";
	}

    // load
    result=(float*)&A;
    printf("[ref] input:        %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    a=_mm_loadu_ps(A);
    result=(float*)&a;
    printf("[sse] _mm_loadu_ps: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_load_ps(B);
    result=(float*)&b;
    printf("[sse] _mm_load_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_loadr_ps(B);
    result=(float*)&b;
    printf("[sse] _mm_loadr_ps: %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_load1_ps(B);
    result=(float*)&b;
    printf("[sse] _mm_load1_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_load_ss(&B[1]);
    result=(float*)&b;
    printf("[sse] _mm_load_ss:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_loadh_pi(a,(__m64*)&A[0]);
    result=(float*)&b;
    printf("[sse] _mm_loadh_pi:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_loadl_pi(a,(__m64*)&A[0]);
    result=(float*)&b;
    printf("[sse] _mm_loadl_pi:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    c=_mm_loadu_si128((__m128i*)C);
    result_i=(int*)&c;
    printf("[sse] _mm_loadu_si128: %d  %d  %d  %d \r\n",result_i[0],result_i[1],result_i[2],result_i[3]);

    c=_mm_lddqu_si128((__m128i*)&C[4]);
    result_i=(int*)&c;
    printf("[sse] _mm_lddqu_si128: %d  %d  %d  %d \r\n",result_i[0],result_i[1],result_i[2],result_i[3]);

    // set
    b=_mm_setzero_ps();
    result=(float*)&b;
    printf("[sse] _mm_setzero_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_set1_ps(A[1]);
    result=(float*)&b;
    printf("[sse] _mm_set1_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_set_ps(A[0],A[1],A[2],A[3]); // high <- low
    result=(float*)&b;
    printf("[sse] _mm_set_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    b=_mm_setr_ps(A[0],A[1],A[2],A[3]);
    result=(float*)&b;
    printf("[sse] _mm_setr_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    
    a=_mm_setr_ps(1.0f,-1.0f,1.5f,105.5f);
    b=_mm_setr_ps(-5.0f,10.0f,-325.0625f,81.125f);
    d=_mm_insert_ps(a,b,0xb0); // 0xb0 =10 11 0000 , 10 = 2 表示选b[2]=-325.0625f, 11 = 3 表示a[3] , b[2]插入a[3]的位置
    // 1.000000  -1.000000  1.500000  -325.062500
    result=(float*)&d;
    printf("[sse] _mm_insert_ps:  %f  %f  %f  %f \r\n",result[0],result[1],result[2],result[3]);

    e=_mm_insert_epi32(c,0,2);
    result_i=(int*)&e;
    printf("[sse] _mm_insert_epi32: %d  %d  %d  %d \r\n",result_i[0],result_i[1],result_i[2],result_i[3]);

    __m128i v = _mm_set_epi32(3, 2, 1, 0); // initialise v to 4 x 32 bit int values
    int extract = _mm_extract_epi32(v, 3); // extract element 3 mast is 3 if imm8 set 4 is err
    printf("[sse] _mm_extract_epi32: %d \r\n",extract);

    int cnt=_mm_movemask_ps(b); // 1 2 4 8
    printf("[sse] _mm_movemask_ps: %d \r\n",cnt);

}
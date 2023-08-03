#include <iostream>
#include <stdlib.h>
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

// 手写动态内存申请字节对齐函数
void* aligned_malloc(size_t size, int alignment)
{
	// 分配足够的内存, 这里的算法很经典, 早期的STL中使用的就是这个算法  
 
	// 首先是维护FreeBlock指针占用的内存大小  
	const int pointerSize = sizeof(void*);
 
	// alignment - 1 + pointerSize这个是FreeBlock内存对齐需要的内存大小  
	// 前面的例子sizeof(T) = 20, __alignof(T) = 16,  
	// g_MaxNumberOfObjectsInPool = 1000  
	// 那么调用本函数就是alignedMalloc(1000 * 20, 16)  
	// 那么alignment - 1 + pointSize = 19  
	const int requestedSize = size + alignment - 1 + pointerSize;
 
	// 分配的实际大小就是20000 + 19 = 20019  
	void* raw = malloc(requestedSize);
 
	// 这里实Pool真正为对象实例分配的内存地址  
	uintptr_t start = (uintptr_t)raw + pointerSize;
	// 向上舍入操作  
	// 解释一下, __ALIGN - 1指明的是实际内存对齐的粒度  
	// 例如__ALIGN = 8时, 我们只需要7就可以实际表示8个数(0~7)  
	// 那么~(__ALIGN - 1)就是进行舍入的粒度  
	// 我们将(bytes) + __ALIGN-1)就是先进行进位, 然后截断  
	// 这就保证了我是向上舍入的  
	// 例如byte = 100, __ALIGN = 8的情况  
	// ~(__ALIGN - 1) = (1 000)B  
	// ((bytes) + __ALIGN-1) = (1 101 011)B  
	// (((bytes) + __ALIGN-1) & ~(__ALIGN - 1)) = (1 101 000 )B = (104)D  
	// 104 / 8 = 13, 这就实现了向上舍入  
	// 对于byte刚好满足内存对齐的情况下, 结果保持byte大小不变  
	// 记得《Hacker's Delight》上面有相关的计算  
	// 这个表达式与下面给出的等价  
	// ((((bytes) + _ALIGN - 1) * _ALIGN) / _ALIGN)  
	// 但是SGI STL使用的方法效率非常高   
	void* aligned = (void*)((start + alignment - 1) & ~(alignment - 1));
 
	// 这里维护一个指向malloc()真正分配的内存  
	*(void**)((uintptr_t)aligned - pointerSize) = raw;
 
	// 返回实例对象真正的地址  
	return aligned;
}

template<typename T>
void aligned_free(T * aligned_ptr)
{
	if (aligned_ptr)
	{
		free(((T**)aligned_ptr)[-1]);
	}
}
 
bool isAligned(void* data, int alignment)
{
	// 又是一个经典算法, 参见<Hacker's Delight>  
	return ((uintptr_t)data & (alignment - 1)) == 0;
}



// avx demo
int avx2_simple_case()
{
    TINIT;
    __m256 a = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 
                             4.0, 3.0, 2.0, 1.0);
    __m256 b = _mm256_set_ps(18.0, 17.0, 16.0, 15.0, 
                             14.0, 13.0, 12.0, 11.0);
    __m256 c = _mm256_add_ps(a, b);

    float d[8];
    
    TIC;
    _mm256_storeu_ps(d, c);

    std::cout << "result equals " << d[0] << "," << d[1]
              << "," << d[2] << "," << d[3] << ","
              << d[4] << "," << d[5] << "," << d[6] << ","
              << d[7] << std::endl;
    TOC("avx2_simple_case");
    return 0;
}


void serial_vec_add(float *A,float *B,float *C,int len)
{
    for(int i = 0; i < len; i++){
        C[i] = A[i] + B[i];
    }
}

void sse_vec_add(float *A,float *B,float *C,int len)
{
    for (int i = 0; i < len; i += 4)    // 一次计算4个数据，所以要改成+4
    {
        __m128 ra = _mm_loadu_ps(A + i); // ra = {A[i], A[i+1], A[i+2], A[i+3]}
        __m128 rb = _mm_loadu_ps(B + i); // rb = {B[i], B[i+1], B[i+2], B[i+3]}
        __m128 rc = _mm_add_ps(ra, rb);  // rc = ra + rb
        _mm_storeu_ps(C + i, rc);        // C[i~i+3] <= rc
    }

}


void sse_vec_add_opt(float *A,float *B,float *C,int len)
{
    for (int i = 0; i < len; i += 4)
    {
        _mm_storeu_ps(C + i,  _mm_add_ps(_mm_loadu_ps(A + i), _mm_loadu_ps(B + i))); // 压行
    }

}


void sse_vec_add_opt_align(float *A,float *B,float *C,int len)
{
    for (int i = 0; i < len; i += 4)
    {
        _mm_store_ps(C + i,  _mm_add_ps(_mm_load_ps(A + i), _mm_load_ps(B + i))); // 用store和load替换storeu和loadu
    }

}

void avx_vec_add_opt_align(float *A,float *B,float *C,int len)
{
   for (int i = 0; i < len; i += 8) // 循环跨度修改为8
    {
        *(__m256 *)(C + i) = _mm256_add_ps(*(__m256 *)(A + i), *(__m256 *)(B + i)); // 使用256位宽的数据与函数
    }
}

enum alloc_align
{
    NON_ALIGN=0,
    C11_ALIGN,
    C17_ALIGN,
    POSIX_ALIGN,
    INTEL_ALIGN,
    WIN_ALIGN,
};

void vec_add_compare(int fun_opt,int len,alloc_align align_opt,int alignment)
{
    int ret=0;
    float * list_A=NULL;
    float * list_B=NULL;
    float * list_C=NULL;
    int size=len*sizeof(float);
    /* align malloc */
    if(align_opt==NON_ALIGN){
        list_A=(float *)malloc(size);
        list_B=(float *)malloc(size);
        list_C=(float *)malloc(size);
    }
    if(align_opt==C11_ALIGN){
        list_A=(float *)aligned_alloc(alignment, size);
        list_B=(float *)aligned_alloc(alignment, size);
        list_C=(float *)aligned_alloc(alignment, size);
    }
    else if(align_opt==C17_ALIGN){
        list_A=new ((std::align_val_t)alignment) float[size];
        list_B=new ((std::align_val_t)alignment) float[size];
        list_C=new ((std::align_val_t)alignment) float[size];
    }
    else if(align_opt==POSIX_ALIGN){
        ret=posix_memalign((void**)&list_A,alignment, size);
        ret=posix_memalign((void**)&list_B,alignment, size);
        ret=posix_memalign((void**)&list_C,alignment, size);
    }
    else if(align_opt==INTEL_ALIGN){
        list_A=(float *)_mm_malloc(alignment, size);
        list_B=(float *)_mm_malloc(alignment, size);
        list_C=(float *)_mm_malloc(alignment, size);
    }
    #if _WIN32
    else if(align_opt==WIN_ALIGN){
        list_A=(float *)_aligned_malloc(alignment, size);
        list_B=(float *)_aligned_malloc(alignment, size);
        list_C=(float *)_aligned_malloc(alignment, size);
    }
    #endif
    if( alignment!=0 || alignment/2==0) {
        if (isAligned(list_A, alignment)) {
            std::cout << "[success]The buffer is aligned "<<std::endl;
            std::cout << "[success]Aligned coef is "<< alignment <<std::endl;
        }
        else{
            std::cout << "[success]The buffer is Not aligned "<<std::endl;
        }
    }
    else{
        std::cout << "[success]The buffer is non-aligned "<<std::endl;
    }
	
    /* select code run */
    char* msg;
    TINIT;
    TIC;
    switch(fun_opt)
    {
        case 0 :	
            serial_vec_add(list_A,list_B,list_C,len);
            msg="serial_vec_add";
            break;
        case 1 :
            sse_vec_add(list_A,list_B,list_C,len);
            msg="sse_vec_add";
            break;
        case 2 :
            sse_vec_add_opt(list_A,list_B,list_C,len);
            msg="sse_vec_add_opt";
            break;
        case 3 :
            sse_vec_add_opt_align(list_A,list_B,list_C,len);
            msg="sse_vec_add_opt_align";
            break;
        case 4 :
            avx_vec_add_opt_align(list_A,list_B,list_C,len);
            msg="avx_vec_add_opt_align";
            break;
         
        default :
            std::cout << "There is not the function !"<<std::endl;
            break;
    } 
    TOC(msg);


    if(align_opt==NON_ALIGN){
        free(list_A);
        free(list_B);
        free(list_C);
    }
    if(align_opt==C11_ALIGN){
        free(list_A);
        free(list_B);
        free(list_C);
    }
    else if(align_opt==C17_ALIGN){
        list_A=new ((std::align_val_t)alignment) float[size];
        list_B=new ((std::align_val_t)alignment) float[size];
        list_C=new ((std::align_val_t)alignment) float[size];
    }
    else if(align_opt==POSIX_ALIGN){
        free(list_A);
        free(list_B);
        free(list_C);
    }
    else if(align_opt==INTEL_ALIGN){
        _mm_free(list_A);
        _mm_free(list_B);
        _mm_free(list_C);
    }
    #if _WIN32
    else if(align_opt==WIN_ALIGN){
        _aligned_free(list_A);
        _aligned_free(list_B);
        _aligned_free(list_C);
    }
    #endif


    std::cout << "[success]aligned buffer free"<<std::endl;
}



int main()
{
    int len=10000*10000; // 向量长度
    int align_coef=16; // 对齐系数
    // buffer alloc align method
    // NON_ALIGN=0,
    // C11_ALIGN,
    // C17_ALIGN,
    // POSIX_ALIGN,
    // INTEL_ALIGN,
    // WIN_ALIGN,
    vec_add_compare(0,len,C11_ALIGN,align_coef);

    // 0 ref 254 ms
    // 1 sse 181 ms
    // 2 sse 压行 170 ms
    // 3 sse 压行 aligned 170 ms
    // 4 avx2 压行 aligned 148 ms

    return 0;
}








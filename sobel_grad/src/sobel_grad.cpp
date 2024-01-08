#include"sobel_grad.h"
// 测试用头文件
#include<sys/time.h>
#include<stdio.h>
#include <time.h>


// 原始FLOAT版本
void SobelGrad(cv::Mat &src, cv::Point2f* grad_vec, float mod_t)
{
	int width = src.cols;
	int height = src.rows;
	//int widthStep = src->widthStep;

	//unsigned char* src_data = (unsigned char*)src->imageData;
#pragma omp parallel for
	for (int r = 1; r < height - 1; r++)
	{
		unsigned char* src_data_rm1 = src.ptr<unsigned char>(r - 1); // 第r-1行数据指针
		unsigned char* src_data_r = src.ptr<unsigned char>(r - 0); // 第r行数据指针
		unsigned char* src_data_rp1 = src.ptr<unsigned char>(r + 1);// 第r+1行数据指针
		
		for (int c = 1; c < width - 1; c++)
		{
			cv::Point2f temp;
			int gradx = src_data_rm1[c + 1] - src_data_rm1[c - 1] +
				2 * src_data_r[c + 1] - 2 * src_data_r[c - 1] +
				src_data_rp1[c + 1] - src_data_rp1[c - 1];

			int grady = src_data_rp1[c - 1] - src_data_rm1[c - 1] +
				2 * src_data_rp1[c] - 2 * src_data_rm1[c] +
				src_data_rp1[c + 1] - src_data_rm1[c + 1];

			float mod = sqrt(gradx*gradx + grady*grady + 1e-6);
			temp.x = gradx / mod;
			temp.y = grady / mod;
			if (mod < mod_t) // 如果模很小, 梯度向量置为(0,0)
			{
				temp.x = temp.y = 0;
			}

			grad_vec[r*width + c] = temp;
		}
	}

	return;
}

// 原始INT8代码
void SobelGradInt8(cv::Mat &src, Grad2Int8* grad_vec, float mod_t)
{
	int width = src.cols;
	int height = src.rows;
	//int widthStep = src->widthStep;

	//unsigned char* src_data = (unsigned char*)src->imageData;
	//#pragma omp parallel for
	for (int r = 1; r < height - 1; r++)
	{
		unsigned char* src_data_rm1 = src.ptr<unsigned char>(r - 1); // 第r-1行数据指针
		unsigned char* src_data_r = src.ptr<unsigned char>(r - 0); // 第r行数据指针
		unsigned char* src_data_rp1 = src.ptr<unsigned char>(r + 1);// 第r+1行数据指针
		for (int c = 1; c < width - 1; c++)
		{
			cv::Point2f temp;
			int gradx = src_data_rm1[c + 1] - src_data_rm1[c - 1] +
				2 * src_data_r[c + 1] - 2 * src_data_r[c - 1] +
				src_data_rp1[c + 1] - src_data_rp1[c - 1];

			int grady = src_data_rp1[c - 1] - src_data_rm1[c - 1] +
				2 * src_data_rp1[c] - 2 * src_data_rm1[c] +
				src_data_rp1[c + 1] - src_data_rm1[c + 1];

			float mod = sqrt(gradx*gradx + grady*grady + 1e-6);

			temp.x = gradx / mod;
			temp.y = grady / mod;
			Grad2Int8 temp_grad;
			temp_grad.x = int8(temp.x * 127); // multiply 127 instead of 128, note: 128 can cause overflow
			temp_grad.y = int8(temp.y * 127);

			if (mod < mod_t) // 如果模很小, 梯度向量置为(0,0)
			{
				temp_grad.x = temp_grad.y = 0;
			}

			grad_vec[r*width + c] = temp_grad;
		}
	}

	return;
}

// float 优化版本的查找表
float invsqrt_tab_f[1000000];
void init_sobel_table_f(float mod_t)
{	
	int width=1000;
	memset(invsqrt_tab_f,0,sizeof(invsqrt_tab_f));
	int mod_sq_t = mod_t*mod_t;
	int i_sq = 0;
	int j_sq = 0;
	int mod_sq= 0;
	int bool_mod=0;
	for (int i = 0 ; i < width; i++)
	{
		i_sq = i*i;
		for (int j = 0 ; j < width; j++)
		{
			j_sq = j*j;
			mod_sq = i_sq + j_sq;
			bool_mod=mod_sq<mod_sq_t?0:1;
			invsqrt_tab_f[i*width+j]= (1.0/sqrt(mod_sq+1e-6))*bool_mod;
		}	
	} 
}

// float 优化版本
void SobelGrad_opt_f(cv::Mat &src, cv::Point2f* grad_vec, float mod_t)
{


	// sobel 算子 是可分离滤波
	// 卷积核 sobel_x 由2个行列阵外积得到 [1 2 1] * [-1 0 1] y * x
	// -1  0  1
	// -2  0  2
	// -1  0  1
	// 卷积核 sobel_y 由2个行列阵外积得到 [1 0 -1] * [1 2 1] y * x
	// -1 -2 -1
	//  0  0  0
	//  1  2  1
		
	int width = src.cols;
	int height = src.rows;
	cv::Point2f temp;
	unsigned char* src_data_rm1 = NULL; // 第r-1行数据指针
	unsigned char* src_data_r = NULL; // 第r行数据指针
	unsigned char* src_data_rp1 = NULL;// 第r+1行数据指针

	register int gradx =0; // 梯度x
	register int grady =0; // 梯度y

	register int signed_x = 0; // 暂存变量：暂存 greadx >> 31 结果
	register int signed_y = 0; // 暂存变量：暂存 gready >> 31 结果
	register int x_idx = 0; // 查找表下标x
	register int y_idx = 0; // 查找表下标y

	register float inv_mod=1e-6; // 1/sqrt(mod*mod)

    // 重复使用的数据暂存
	int x_tmp_buf[1280][3]={0}; // 临时存储数据的buf
	int y_tmp_buf[1280][3]={0}; // 临时存储数据的buf

	register int x_tmp_d=0; // 暂存变量：暂存当前行卷积结果
	register int y_tmp_d=0; // 暂存变量：暂存当前行卷积结果

	register int num_d=0; // 暂存变量：暂存IDX(i+1)结果 用于确定列卷积的对应的通道数 d:down
	register int num_m=0; // 暂存变量：暂存IDX( i )结果 用于确定列卷积的对应的通道数 m:middle
	register int num_u=0; // 暂存变量：暂存IDX(i-1)结果 用于确定列卷积的对应的通道数 u:upper

	register int temp1=0; //暂存变量：用于解除行内数据依赖 1<2<3 队列形式 暂存
	register int temp2=0; //暂存变量：用于解除行内数据依赖 1<2<3 队列形式 暂存
	register int temp3=0; //暂存变量：用于解除行内数据依赖 1<2<3 队列形式 暂存
	int temp1_once=0; //暂存变量：用于解除行内数据依赖  1<2<3 队列形式 暂存
	int temp2_once=0; //暂存变量：用于解除行内数据依赖  1<2<3 队列形式 暂存
	int temp3_once=0; //暂存变量：用于解除行内数据依赖  1<2<3 队列形式 暂存


	// ------------------ 先计算前 2 行的行内卷积，保存到 buf 中  ----------------------
	src_data_rm1 = src.ptr<unsigned char>(0); // 第0行数据指针
	src_data_r = src.ptr<unsigned char>(1); // 第1行数据指针
	
	temp1=src_data_rm1[0]; 
	temp2=src_data_rm1[1]; 
	temp1_once=src_data_r[0]; 
	temp2_once=src_data_r[1]; 
	for(int j = 1; j < width-1 ; ++j )
	{
		// 前两行计算 grad_x x方向卷积
		// x_tmp_buf[j][0] = src_data_rm1[j+1] - src_data_rm1[j-1];
		// x_tmp_buf[j][1] = src_data_r[j+1] - src_data_r[j-1];

		// 前两行计算 grad_y x方向卷积
		// y_tmp_buf[j][0] = src_data_rm1[j-1]  + src_data_rm1[j] * 2 + src_data_rm1[j+1];
		// y_tmp_buf[j][1] = src_data_r[j-1]  + src_data_r[j] * 2 + src_data_r[j+1];

		temp3=src_data_rm1[j+1];
		temp3_once=src_data_r[j+1];

		// 前两行计算 grad_x x方向卷积
		x_tmp_buf[j][0] = temp3 - temp1;
		x_tmp_buf[j][1] = temp3_once - temp1_once;

		// 前两行计算 grad_y x方向卷积
		y_tmp_buf[j][0] = temp1  + (temp2<<1) + temp3;
		y_tmp_buf[j][1] = temp1_once  + (temp2_once<<1) + temp3_once;
	
		temp1=temp2; 
		temp2=temp3;
		temp1_once=temp2_once; 
		temp2_once=temp3_once;
	}



	// 每算一行新的，就相当凑齐 3 行，可以做行内的卷积，并更新更新暂存buf
	#define IDX(n) ((n)%3) // 取余数
	for(int i = 1; i < height-1; i++ )
	{

		src_data_rp1 = src.ptr<unsigned char>(i + 1);// 第r+1行数据指针

		num_d=IDX(i+1);
		num_m=IDX(i);
		num_u=IDX(i-1);

		temp1=src_data_rp1[0]; 
		temp2=src_data_rp1[1]; 

		for(int j = 1; j < width-1; j++)
		{

			temp3=src_data_rp1[j+1];
			x_tmp_d= x_tmp_buf[j][num_d] = temp3 - temp1;
			gradx = x_tmp_buf[j][num_u] + (x_tmp_buf[j][num_m] << 1) + x_tmp_d; 

			y_tmp_d= y_tmp_buf[j][num_d] = temp1  + (temp2 << 1) + temp3;
			grady = y_tmp_d - y_tmp_buf[j][num_u]; 

			temp1=temp2; 
			temp2=temp3;
		
			signed_x = gradx>>31; // ox11111111 or ox00000000
			signed_y = grady>>31; // ox11111111 or ox00000000
			x_idx = (gradx^signed_x)-signed_x; // 取绝对值
			y_idx = (grady^signed_y)-signed_y; // 取绝对值

			// 查找表0.5ms sqrt1ms
			inv_mod=invsqrt_tab_f[x_idx*1000+y_idx];
			// 0.25ms
			temp.x = gradx * inv_mod;
			temp.y = grady * inv_mod;
			
			grad_vec[i*width + j] = temp;
		}
	}
	return;
}

// int8 优化版本的查找表
float invsqrt_tab_s8[1000000];
void init_sobel_table_s8(float mod_t)
{	
	int width=1000;
	memset(invsqrt_tab_s8,0,sizeof(invsqrt_tab_s8));
	int mod_sq_t = mod_t*mod_t;
	int i_sq = 0;
	int j_sq = 0;
	int mod_sq= 0;
	int bool_mod=0;
	for (int i = 0 ; i < width; i++)
	{
		i_sq = i*i;
		for (int j = 0 ; j < width; j++)
		{
			j_sq = j*j;
			mod_sq = i_sq + j_sq;
			bool_mod=mod_sq<mod_sq_t?0:1;
			invsqrt_tab_s8[i*width+j]=(127.0/sqrt(mod_sq+1e-6))*bool_mod;
		}	
	} 
}

// int8 优化版本
void SobelGrad_opt_s8(unsigned char * src, Grad2Int8* grad_vec, int width, int height)
{


	// sobel 算子 是可分离滤波
	// 卷积核 sobel_x 由2个行列阵外积得到 [1 2 1] * [-1 0 1] y * x
	// -1  0  1
	// -2  0  2
	// -1  0  1
	// 卷积核 sobel_y 由2个行列阵外积得到 [1 0 -1] * [1 2 1] y * x
	// -1 -2 -1
	//  0  0  0
	//  1  2  1
		
	Grad2Int8 temp;
	unsigned char* src_data_rm1 = NULL; // 第r-1行数据指针
	unsigned char* src_data_r = NULL; // 第r行数据指针
	unsigned char* src_data_rp1 = NULL;// 第r+1行数据指针

	register int gradx =0; // 梯度x
	register int grady =0; // 梯度y

	register int signed_x = 0; // 暂存变量：暂存 greadx >> 31 结果
	register int signed_y = 0; // 暂存变量：暂存 gready >> 31 结果
	register int x_idx = 0; // 查找表下标x
	register int y_idx = 0; // 查找表下标y

	register float inv_mod=1e-6; // 1/sqrt(mod*mod)

    // 重复使用的数据暂存
	int x_tmp_buf[1280][3]={0}; // 临时存储数据的buf
	int y_tmp_buf[1280][3]={0}; // 临时存储数据的buf

	register int x_tmp_d=0; // 暂存变量：暂存当前行卷积结果
	register int y_tmp_d=0; // 暂存变量：暂存当前行卷积结果

	register int num_d=0; // 暂存变量：暂存IDX(i+1)结果 用于确定列卷积的对应的通道数 d:down
	register int num_m=0; // 暂存变量：暂存IDX( i )结果 用于确定列卷积的对应的通道数 m:middle
	register int num_u=0; // 暂存变量：暂存IDX(i-1)结果 用于确定列卷积的对应的通道数 u:upper

	register int temp1=0; //暂存变量：用于解除行内数据依赖 1<2<3 队列形式 暂存
	register int temp2=0; //暂存变量：用于解除行内数据依赖 1<2<3 队列形式 暂存
	register int temp3=0; //暂存变量：用于解除行内数据依赖 1<2<3 队列形式 暂存
	int temp1_once=0; //暂存变量：用于解除行内数据依赖  1<2<3 队列形式 暂存
	int temp2_once=0; //暂存变量：用于解除行内数据依赖  1<2<3 队列形式 暂存
	int temp3_once=0; //暂存变量：用于解除行内数据依赖  1<2<3 队列形式 暂存


	// ------------------ 先计算前 2 行的行内卷积，保存到 buf 中  ----------------------
	src_data_rm1 = &src[0]; // 第0行数据指针
	src_data_r = &src[width]; // 第1行数据指针
	
	temp1=src_data_rm1[0]; 
	temp2=src_data_rm1[1]; 
	temp1_once=src_data_r[0]; 
	temp2_once=src_data_r[1]; 
	for(int j = 1; j < width-1 ; ++j )
	{
		// 前两行计算 grad_x x方向卷积
		// x_tmp_buf[j][0] = src_data_rm1[j+1] - src_data_rm1[j-1];
		// x_tmp_buf[j][1] = src_data_r[j+1] - src_data_r[j-1];

		// 前两行计算 grad_y x方向卷积
		// y_tmp_buf[j][0] = src_data_rm1[j-1]  + src_data_rm1[j] * 2 + src_data_rm1[j+1];
		// y_tmp_buf[j][1] = src_data_r[j-1]  + src_data_r[j] * 2 + src_data_r[j+1];

		temp3=src_data_rm1[j+1];
		temp3_once=src_data_r[j+1];

		// 前两行计算 grad_x x方向卷积
		x_tmp_buf[j][0] = temp3 - temp1;
		x_tmp_buf[j][1] = temp3_once - temp1_once;

		// 前两行计算 grad_y x方向卷积
		y_tmp_buf[j][0] = temp1  + (temp2<<1) + temp3;
		y_tmp_buf[j][1] = temp1_once  + (temp2_once<<1) + temp3_once;
	
		temp1=temp2; 
		temp2=temp3;
		temp1_once=temp2_once; 
		temp2_once=temp3_once;
	}



	// 每算一行新的，就相当凑齐 3 行，可以做行内的卷积，并更新更新暂存buf
	#define IDX(n) ((n)%3) // 取余数
	for(int i = 1; i < height-1; i++ )
	{
		src_data_rp1 = &src[(i+1)*width];// 第r+1行数据指针

		num_d=IDX(i+1);
		num_m=IDX(i);
		num_u=IDX(i-1);

		temp1=src_data_rp1[0]; 
		temp2=src_data_rp1[1]; 

		for(int j = 1; j < width-1; j++)
		{

			temp3=src_data_rp1[j+1];
			x_tmp_d= x_tmp_buf[j][num_d] = temp3 - temp1;
			gradx = x_tmp_buf[j][num_u] + (x_tmp_buf[j][num_m] << 1) + x_tmp_d; 

			y_tmp_d= y_tmp_buf[j][num_d] = temp1  + (temp2 << 1) + temp3;
			grady = y_tmp_d - y_tmp_buf[j][num_u]; 

			temp1=temp2; 
			temp2=temp3;
		
			signed_x = gradx>>31; // ox11111111 or ox00000000
			signed_y = grady>>31; // ox11111111 or ox00000000
			x_idx = (gradx^signed_x)-signed_x; // 取绝对值
			y_idx = (grady^signed_y)-signed_y; // 取绝对值

			// 查找表0.5ms sqrt1ms
			inv_mod=invsqrt_tab_s8[x_idx*1000+y_idx];
			// 0.25ms
			temp.x = int8(gradx * inv_mod);
			temp.y = int8(grady * inv_mod);
			//printf("%d\r\n",int(temp.x));
			
			grad_vec[i*width + j] = temp;
		}
	}
	return;
}

// float neon优化版本
void SobelGrad_opt_neon_f(unsigned char * src, cv::Point2f* grad_vec, int width, int height,float mod_t)
{

	unsigned char* src_data_rm1 = NULL; // 第r-1行数据指针
	unsigned char* src_data_r = NULL; // 第r行数据指针
	unsigned char* src_data_rp1 = NULL;// 第r+1行数据指针

	// 暂存buf设置：用来暂存行卷积结果
	short x_tmp_buf[3][1280]={0}; // 临时存储数据的buf
	short y_tmp_buf[3][1280]={0}; // 临时存储数据的buf
	#define IDX(n) ((n)%3) // 取余数，用来确定列卷积通道数 0 1 2
	register int num_u =0; // 暂存IDX(i-1)的值
	register int num_m =0; // 暂存IDX( i )的值
	register int num_d =0; // 暂存IDX(i+1)的值

	// 循环变量设置：每次取8个数据
	int idx_start =8; // 每行循环起始的位置
	int idx_tail=(width%8); // 不能凑够一个矢量的元素个数
	int idx_end = width-idx_tail-8; // 每行循环终止的上限
	bool idx_tail_flag=(idx_tail==0); // 判断行内数据能否被8整除

	// 暂存变量：用于解除行内数据依赖 队列形式 先入先出
	register int temp1=0; // 第1行 队列头部  可能会复用
	register int temp2=0; // 第1行 队列头部  可能会复用
	register int temp3=0; // 第1行 队列头部  可能会复用
	register int temp1_once=0; // 第2行 队列头部 
	register int temp2_once=0; // 第2行 队列头部 
	register int temp3_once=0; // 第2行 队列头部 


	// ------------------ 先计算前 2 行的行内卷积，保存到 buf 中  ----------------------
	src_data_rm1 = &src[0]; // 第0行数据指针
	src_data_r = &src[width]; // 第1行数据指针
	
	temp1=src_data_rm1[0]; 
	temp2=src_data_rm1[1]; 
	temp1_once=src_data_r[0]; 
	temp2_once=src_data_r[1]; 
	for(int j = 1; j < width-1 ; ++j )
	{	
		temp3=src_data_rm1[j+1]; // 队列尾部tail 后进后出
		temp3_once=src_data_r[j+1]; // 队列尾部tail 后进后出

		// 前两行计算 grad_x x方向卷积
		x_tmp_buf[0][j] = temp3 - temp1;
		x_tmp_buf[1][j] = temp3_once - temp1_once;

		// 前两行计算 grad_y x方向卷积
		y_tmp_buf[0][j] = temp1  + ( temp2 << 1 ) + temp3;
		y_tmp_buf[1][j] = temp1_once  + ( temp2_once << 1 ) + temp3_once;
	
		// 队列变换：Out of queue < [temp1<temp2<temp3] <  incoming queue 
		temp1 = temp2; // 头部元素出队列，队列中部元素从temp2移到temp1位置
		temp2 = temp3; // 队列尾部元素从temp3移到temp2位置
		temp1_once = temp2_once; 
		temp2_once = temp3_once; 
	}

	// 处理固定的数据
	int mod_t_sq=int(mod_t*mod_t); // 模阈值的平方
	float32x4_t v_zeros=vmovq_n_f32(0.0); // 0矢量
	int32x4_t mod_t_sq_s32_q= vld1q_dup_s32(&mod_t_sq); // load数据到矢量寄存器

	// 每算一行新的，就相当凑齐 3 行，可以做行内的卷积，并更新更新暂存buf
	for(int i = 1; i < height-1; i++ )
	{
		src_data_rp1 = &src[(i+1)*width];// 第r+1行数据指针

		num_u=IDX(i-1);
		num_d=IDX(i+1);
		num_m=IDX(i);

		//由于通过队列采用矢量的形式解除行内重复访存，行首和行尾需要单独处理，用一个矢量处理不够8个的数
					
		// 1、加载数据 vhead vbody vtail 作为矢量队列 相当与标量temp1-temp3	队列变换：Out of queue < [vhead<vbody<vtail] <  incoming queue 
		uint8x8_t vhead=vld1_u8((uint8_t*)&src_data_rp1[0]); // 加载左边连续的数据 8个 
		uint8x8_t vbody=vld1_u8((uint8_t*)&src_data_rp1[1]); // 加载中间连续的数据 8个
		uint8x8_t vtail=vld1_u8((uint8_t*)&src_data_rp1[2]); // 加载右边连续的数据 8个
	
		int16x8_t x_buf_u=vld1q_s16(&x_tmp_buf[num_u][1]); // 加载列卷积对应的第1个通道的数据 8个
		int16x8_t x_buf_m=vld1q_s16(&x_tmp_buf[num_m][1]); // 加载列卷积对应的第2个通道的数据 8个

		int16x8_t y_buf_u=vld1q_s16(&y_tmp_buf[num_u][1]); // 加载列卷积对应的第1个通道的数据 8个
		int16x8_t y_buf_m=vld1q_s16(&y_tmp_buf[num_m][1]); // 加载列卷积对应的第2个通道的数据 8个

		uint8x8_t src_l_u8_h=vhead; // l:left 行卷积左边的数据 u:uint8 h:half 64bit 4个元素  q:128bit
		uint8x8_t src_m_u8_h=vbody; // m:midle 行卷积中间的数据 u:uint8 h:half 64bit 4个元素  q:128bit
		uint8x8_t src_r_u8_h=vtail; // r:riaght 行卷积右边的数据 u:uint8 h:half 64bit 4个元素  q:128bit

		// 2、卷积运算: 求梯度 gradx和grady
		int16x8_t x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
		int16x8_t v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
		int16x8_t y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
		int16x8_t v_grady =vsubq_s16(y_buf_d,y_buf_u);

		// 3、数学运算：求出 正弦 和 余弦
		
		// 3.1、数据拆分：处理low部分数据 4个
		int16x4_t v_gradx_h=vget_low_s16(v_gradx);
		int16x4_t v_grady_h=vget_low_s16(v_grady);

		int32x4_t mod_sq_s32_q=vmull_s16(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlal_s16(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		float32x4_t invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		uint32x4_t mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q); // 1/sqrt(mod_sq_s32_q)
	
		float32x4x2_t temp_q_lo; // 存储结果
		temp_q_lo.val[0]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_gradx_h)),invmod_f32_q);
		temp_q_lo.val[1]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_grady_h)),invmod_f32_q);
		
		// 3.2、数据拆分：处理high部分数据 4个
		v_gradx_h=vget_high_s16(v_gradx);
		v_grady_h=vget_high_s16(v_grady);

		mod_sq_s32_q=vmull_s16(v_gradx_h, v_gradx_h);  // mod_sq_s32_q = gradx*gradx
		mod_sq_s32_q=vmlal_s16(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q); // 1/sqrt(mod_sq_s32_q)

		float32x4x2_t temp_q_hi; // 存储结果
		temp_q_hi.val[0]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_gradx_h)),invmod_f32_q);
		temp_q_hi.val[1]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_grady_h)),invmod_f32_q);

		// 4、数据写回
		vst2q_f32((float *)&grad_vec[i*width+1], temp_q_lo); // 数据写到内存 交织形式
		vst2q_f32((float *)&grad_vec[i*width+5], temp_q_hi); // 数据写到内存 交织形式
		
		// 发送列卷积对应的第3个通道的数据到内存 8个
		vst1q_s16(&x_tmp_buf[num_d][1], x_buf_d);
		vst1q_s16(&y_tmp_buf[num_d][1], y_buf_d);

		vbody=vld1_u8((uint8_t*)&src_data_rp1[8]); // 加载中间连续的数据 8个
		for(int j=idx_start; j < idx_end; j+=8)
		{	
			// 数据预存储 实际效果看编译器 这里只是提示编译器
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&x_tmp_buf[num_d][j]), "i"(1024));
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&y_tmp_buf[num_d][j]), "i"(1024));
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&grad_vec[i*width+j]), "i"(1024));	
		
			// 1、加载数据
			vtail=vld1_u8((uint8_t*)&src_data_rp1[j+8]); // 加载连续的数据到队列尾部 8个
			
			// 数据拼接
			src_l_u8_h=vext_u8(vhead, vbody, 7); // 拼接
			src_m_u8_h=vbody; 
			src_r_u8_h=vext_u8(vbody, vtail, 1); // 拼接
			vhead=vbody;
			vbody=vtail;
		
			x_buf_u=vld1q_s16(&x_tmp_buf[num_u][j]); // 加载缓存buf 8个
			x_buf_m=vld1q_s16(&x_tmp_buf[num_m][j]); // 加载缓存buf 8个

			y_buf_u=vld1q_s16(&y_tmp_buf[num_u][j]); // 加载缓存buf 8个
			y_buf_m=vld1q_s16(&y_tmp_buf[num_m][j]); // 加载缓存buf 8个
	
			// 2、卷积运算
			x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
			v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
			y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
			v_grady =vsubq_s16(y_buf_d,y_buf_u);

			// 3、数学运算
			// 3.1、处理low部分
			v_gradx_h=vget_low_s16(v_gradx);
			v_grady_h=vget_low_s16(v_grady);

			mod_sq_s32_q=vmull_s16(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
			mod_sq_s32_q=vmlal_s16(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
			invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
			mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
			invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);
		
			temp_q_lo.val[0]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_gradx_h)),invmod_f32_q);
			temp_q_lo.val[1]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_grady_h)),invmod_f32_q);
		
		    // 3.2、处理high部分
			v_gradx_h=vget_high_s16(v_gradx);
			v_grady_h=vget_high_s16(v_grady);

			mod_sq_s32_q=vmull_s16(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
			mod_sq_s32_q=vmlal_s16(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
			invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
			mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
			invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);

			temp_q_hi.val[0]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_gradx_h)),invmod_f32_q);
			temp_q_hi.val[1]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_grady_h)),invmod_f32_q);

			// 4、数据写回
			vst2q_f32((float *)&grad_vec[i*width+j], temp_q_lo); 
			vst2q_f32((float *)&grad_vec[i*width+j+4], temp_q_hi); 

			vst1q_s16(&x_tmp_buf[num_d][j], x_buf_d);
			vst1q_s16(&y_tmp_buf[num_d][j], y_buf_d);		
		}
		// 处理剩余的数据	8+(width%8)个数据
		// 1、加载数据 
		vhead=vld1_u8((uint8_t*)&src_data_rp1[idx_end-2]); // 加载左边连续的数据 8个
		vbody=vld1_u8((uint8_t*)&src_data_rp1[idx_end-1]); // 加载中间连续的数据 8个
		vtail=vld1_u8((uint8_t*)&src_data_rp1[idx_end]); // 加载右边连续的数据 8个

		x_buf_u=vld1q_s16(&x_tmp_buf[num_u][idx_end-1]); // 加载缓存buf 8个
		x_buf_m=vld1q_s16(&x_tmp_buf[num_m][idx_end-1]); // 加载缓存buf 8个

		y_buf_u=vld1q_s16(&y_tmp_buf[num_u][idx_end-1]); // 加载缓存buf 8个
		y_buf_m=vld1q_s16(&y_tmp_buf[num_m][idx_end-1]); // 加载缓存buf 8个

		src_l_u8_h=vhead;
		src_m_u8_h=vbody;
		src_r_u8_h=vtail;

		// 2、卷积运算
		x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
		v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
		y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
		v_grady =vsubq_s16(y_buf_d,y_buf_u);

		// 3、数学运算
		// 3、1处理LOW部分
		v_gradx_h=vget_low_s16(v_gradx);
	    v_grady_h=vget_low_s16(v_grady);

		mod_sq_s32_q=vmull_s16(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlal_s16(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);
	
		temp_q_lo.val[0]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_gradx_h)),invmod_f32_q);
		temp_q_lo.val[1]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_grady_h)),invmod_f32_q);
		
		// 3.2、处理high部分
		v_gradx_h=vget_high_s16(v_gradx);
		v_grady_h=vget_high_s16(v_grady);

		mod_sq_s32_q=vmull_s16(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlal_s16(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);

		temp_q_hi.val[0]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_gradx_h)),invmod_f32_q);
		temp_q_hi.val[1]=vmulq_f32(vcvtq_f32_s32(vmovl_s16(v_grady_h)),invmod_f32_q);

		// 4、数据写回
		vst2q_f32((float *)&grad_vec[i*width+idx_end-1], temp_q_lo); 
		vst2q_f32((float *)&grad_vec[i*width+idx_end+3], temp_q_hi); 
		
		vst1q_s16(&x_tmp_buf[num_d][idx_end-1], x_buf_d);
		vst1q_s16(&y_tmp_buf[num_d][idx_end-1], y_buf_d);
		// 处理不能凑够一个矢量的部分
		if(idx_tail_flag)
		{
			;
		}
		else
		{
			cv::Point2f temp;
			float mod = 1e-6;
			temp1=src_data_rp1[width-idx_tail]; 
			temp2=src_data_rp1[width-idx_tail]; 
			temp3= 0;
			int gradx=0;
			int grady=0;
			for(int k=width-idx_tail;k<width-1;k++)
			{
				
				temp3=src_data_rp1[k+1];
				x_tmp_buf[num_d][k] = temp3 - temp1;
				gradx = x_tmp_buf[num_u][k] + (x_tmp_buf[num_m][k] << 1) + x_tmp_buf[num_d][k]; 

				y_tmp_buf[num_d][k] = temp1  + (temp2 << 1) + temp3;
				grady = y_tmp_buf[num_d][k] - y_tmp_buf[num_u][k]; 
				
				temp1=temp2; 
				temp2=temp3;

				mod = sqrtf(gradx*gradx + grady*grady + 1e-6);
				temp.x = gradx / mod;
				temp.y = grady / mod;
				if (mod < mod_t) // 如果模很小, 梯度向量置为(0,0)
				{
					temp.x = temp.y = 0;
				}
	
				grad_vec[i*width + k] = temp;
			}
		}

	}

	return;
}

#ifdef USE_FP16
// int neon优化版本 中间使用FP16计算
void SobelGrad_opt_neon_s8(unsigned char * src, Grad2Int8* grad_vec, int width, int height,float mod_t)
{

	unsigned char* src_data_rm1 = NULL; // 第r-1行数据指针
	unsigned char* src_data_r = NULL; // 第r行数据指针
	unsigned char* src_data_rp1 = NULL;// 第r+1行数据指针

	// 暂存buf设置：用来暂存行卷积结果
	short x_tmp_buf[3][1280]={0}; // 临时存储数据的buf gradx的行卷积结果
	short y_tmp_buf[3][1280]={0}; // 临时存储数据的buf grady的行卷积结果
	#define IDX(n) ((n)%3) // 取余数，用来确定列卷积通道数 0 1 2
	register int num_u =0; // 暂存IDX(i-1)的值
	register int num_m =0; // 暂存IDX( i )的值
	register int num_d =0; // 暂存IDX(i+1)的值

	// 循环变量设置：每次取8个数据
	int idx_start =8; // 每行循环起始的位置
	int idx_tail=(width%8); // 不能凑够一个矢量的元素个数
	int idx_end = width-idx_tail-8; // 每行循环终止的上限
	bool idx_tail_flag=(idx_tail==0); // 判断行内数据能否被8整除

	// 暂存变量：用于解除行内数据依赖 队列形式 先入先出
	register int temp1=0; // 第1行 队列头部  可能会复用
	register int temp2=0; // 第1行 队列头部  可能会复用
	register int temp3=0; // 第1行 队列头部  可能会复用
	register int temp1_once=0; // 第2行 队列头部 
	register int temp2_once=0; // 第2行 队列头部 
	register int temp3_once=0; // 第2行 队列头部 


	// ------------------ 先计算前 2 行的行内卷积，保存到 buf 中  ----------------------
	src_data_rm1 = &src[0]; // 第0行数据指针
	src_data_r = &src[width]; // 第1行数据指针
	
	temp1=src_data_rm1[0]; 
	temp2=src_data_rm1[1]; 
	temp1_once=src_data_r[0]; 
	temp2_once=src_data_r[1]; 
	for(int j = 1; j < width-1 ; ++j )
	{	
		temp3=src_data_rm1[j+1]; // 队列尾部tail 后进后出
		temp3_once=src_data_r[j+1]; // 队列尾部tail 后进后出

		// 前两行计算 grad_x x方向卷积
		x_tmp_buf[0][j] = temp3 - temp1;
		x_tmp_buf[1][j] = temp3_once - temp1_once;

		// 前两行计算 grad_y x方向卷积
		y_tmp_buf[0][j] = temp1  + ( temp2 << 1 ) + temp3;
		y_tmp_buf[1][j] = temp1_once  + ( temp2_once << 1 ) + temp3_once;
	
		// 队列变换：Out of queue < [temp1<temp2<temp3] <  incoming queue 
		temp1 = temp2; // 头部元素出队列，队列中部元素从temp2移到temp1位置
		temp2 = temp3; // 队列尾部元素从temp3移到temp2位置
		temp1_once = temp2_once; 
		temp2_once = temp3_once; 
	}

	// 处理固定的数据
	int mod_t_sq=int(mod_t*mod_t); // 模阈值的平方
	float32x4_t v_zeros=vmovq_n_f32(0.0); // 0矢量
	int32x4_t mod_t_sq_s32_q= vld1q_dup_s32(&mod_t_sq); // load数据到矢量寄存器

	// 每算一行新的，就相当凑齐 3 行，可以做行内的卷积，并更新更新暂存buf
	for(int i = 1; i < height-1; i++ )
	{
		src_data_rp1 = &src[(i+1)*width];// 第r+1行数据指针

		num_u=IDX(i-1);
		num_d=IDX(i+1);
		num_m=IDX(i);

		//由于通过队列采用矢量的形式解除行内重复访存，行首和行尾需要单独处理，用一个矢量处理不够8个的数
					
		// 1、加载数据 vhead vbody vtail 作为矢量队列 相当于标量temp1-temp3	队列变换：Out of queue < [vhead<vbody<vtail] <  incoming queue 
		uint8x8_t vhead=vld1_u8((uint8_t*)&src_data_rp1[0]); // 加载左边连续的数据 8个 
		uint8x8_t vbody=vld1_u8((uint8_t*)&src_data_rp1[1]); // 加载中间连续的数据 8个
		uint8x8_t vtail=vld1_u8((uint8_t*)&src_data_rp1[2]); // 加载右边连续的数据 8个
	
		int16x8_t x_buf_u=vld1q_s16(&x_tmp_buf[num_u][1]); // 加载列卷积对应的第1个通道的数据 8个
		int16x8_t x_buf_m=vld1q_s16(&x_tmp_buf[num_m][1]); // 加载列卷积对应的第2个通道的数据 8个

		int16x8_t y_buf_u=vld1q_s16(&y_tmp_buf[num_u][1]); // 加载列卷积对应的第1个通道的数据 8个
		int16x8_t y_buf_m=vld1q_s16(&y_tmp_buf[num_m][1]); // 加载列卷积对应的第2个通道的数据 8个

		uint8x8_t src_l_u8_h=vhead; // l:left 行卷积左边的数据 u:uint8 h:half 64bit 4个元素  q:128bit
		uint8x8_t src_m_u8_h=vbody; // m:midle 行卷积中间的数据 u:uint8 h:half 64bit 4个元素  q:128bit
		uint8x8_t src_r_u8_h=vtail; // r:riaght 行卷积右边的数据 u:uint8 h:half 64bit 4个元素  q:128bit

		// 2、卷积运算: 求梯度 gradx和grady
		int16x8_t x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
		int16x8_t v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
		int16x8_t y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
		int16x8_t v_grady =vsubq_s16(y_buf_d,y_buf_u);

		// 3、数学运算：求出 正弦 和 余弦 
		
		// 3.1、数据拆分：处理low部分数据 4个
		int32x4_t v_gradx_q=vmovl_s16(vget_low_s16(v_gradx));
		int32x4_t v_grady_q=vmovl_s16(vget_low_s16(v_grady));


		int32x4_t mod_sq_s32_q=vmulq_s32(v_gradx_q, v_gradx_q);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_q,v_grady_q); // mod_sq_s32_q=gradx_sq+grady*grady
		float32x4_t invmod_f32_lo=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		uint32x4_t mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_lo=vbslq_f32(mask, v_zeros, invmod_f32_lo); // 1/sqrt(mod_sq_s32_q)

	
	
		// 3.2、数据拆分：处理high部分数据 4个
		v_gradx_q=vmovl_s16(vget_high_s16(v_gradx));
		v_grady_q=vmovl_s16(vget_high_s16(v_grady));


		mod_sq_s32_q=vmulq_s32(v_gradx_q, v_gradx_q);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_q,v_grady_q); // mod_sq_s32_q=gradx_sq+grady*grady
		float32x4_t invmod_f32_hi=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_hi=vbslq_f32(mask, v_zeros, invmod_f32_hi); // 1/sqrt(mod_sq_s32_q)

		float16x8_t invmod_f16_q=vmulq_n_f16(vcombine_f16(vcvt_f16_f32(invmod_f32_lo),vcvt_f16_f32(invmod_f32_hi)),127.0);
		int8x8x2_t temp_s8_q;
		temp_s8_q.val[0]=vmovn_s16(vcvtq_s16_f16(vmulq_f16(vcvtq_f16_s16(v_gradx),invmod_f16_q)));
		temp_s8_q.val[1]=vmovn_s16(vcvtq_s16_f16(vmulq_f16(vcvtq_f16_s16(v_grady),invmod_f16_q)));

		vst2_s8((int8_t *)&grad_vec[i*width+1], temp_s8_q); // 数据写到内存 交织形式
		
		
		// 发送列卷积对应的第3个通道的数据到内存 8个
		vst1q_s16(&x_tmp_buf[num_d][1], x_buf_d);
		vst1q_s16(&y_tmp_buf[num_d][1], y_buf_d);

		vbody=vld1_u8((uint8_t*)&src_data_rp1[8]); // 加载中间连续的数据 8个
		for(int j=idx_start; j < idx_end; j+=8)
		{	
			// 数据预存储 实际效果看编译器 这里只是提示编译器
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&x_tmp_buf[num_d][j]), "i"(1024));
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&y_tmp_buf[num_d][j]), "i"(1024));
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&grad_vec[i*width+j]), "i"(1024));	
		
			// 1、加载数据
			vtail=vld1_u8((uint8_t*)&src_data_rp1[j+8]); // 加载连续的数据到队列尾部 8个
			
			// 数据拼接
			src_l_u8_h=vext_u8(vhead, vbody, 7); // 拼接
			src_m_u8_h=vbody; 
			src_r_u8_h=vext_u8(vbody, vtail, 1); // 拼接
			vhead=vbody;
			vbody=vtail;
		
			x_buf_u=vld1q_s16(&x_tmp_buf[num_u][j]); // 加载缓存buf 8个
			x_buf_m=vld1q_s16(&x_tmp_buf[num_m][j]); // 加载缓存buf 8个

			y_buf_u=vld1q_s16(&y_tmp_buf[num_u][j]); // 加载缓存buf 8个
			y_buf_m=vld1q_s16(&y_tmp_buf[num_m][j]); // 加载缓存buf 8个

	
			// 2、卷积运算
			x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
			v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
			y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
			v_grady =vsubq_s16(y_buf_d,y_buf_u);

			// 3、数学运算
			// 3.1、处理low部分
			v_gradx_q=vmovl_s16(vget_low_s16(v_gradx));
			v_grady_q=vmovl_s16(vget_low_s16(v_grady));

			mod_sq_s32_q=vmulq_s32(v_gradx_q, v_gradx_q);  // mod_sq_s32_q= gradx*gradx
			mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_q,v_grady_q); // mod_sq_s32_q=gradx_sq+grady*grady
			invmod_f32_lo=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
			mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
			invmod_f32_lo=vbslq_f32(mask, v_zeros, invmod_f32_lo); // 1/sqrt(mod_sq_s32_q)

		
			// 3.2、数据拆分：处理high部分数据 4个
			v_gradx_q=vmovl_s16(vget_high_s16(v_gradx));
			v_grady_q=vmovl_s16(vget_high_s16(v_grady));


			mod_sq_s32_q=vmulq_s32(v_gradx_q, v_gradx_q);  // mod_sq_s32_q= gradx*gradx
			mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_q,v_grady_q); // mod_sq_s32_q=gradx_sq+grady*grady
			invmod_f32_hi=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
			mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
			invmod_f32_hi=vbslq_f32(mask, v_zeros, invmod_f32_hi); // 1/sqrt(mod_sq_s32_q)

			invmod_f16_q=vmulq_n_f16(vcombine_f16(vcvt_f16_f32(invmod_f32_lo),vcvt_f16_f32(invmod_f32_hi)),127.0); // 127/sqrt(mod_sq_s32_q)
			temp_s8_q.val[0]=vmovn_s16(vcvtq_s16_f16(vmulq_f16(vcvtq_f16_s16(v_gradx),invmod_f16_q)));
			temp_s8_q.val[1]=vmovn_s16(vcvtq_s16_f16(vmulq_f16(vcvtq_f16_s16(v_grady),invmod_f16_q)));
			vst2_s8((int8_t *)&grad_vec[i*width+j], temp_s8_q); 

			vst1q_s16(&x_tmp_buf[num_d][j], x_buf_d);
			vst1q_s16(&y_tmp_buf[num_d][j], y_buf_d);		
		}
		// 处理剩余的数据	8+(width%8)个数据
		// 1、加载数据 
		vhead=vld1_u8((uint8_t*)&src_data_rp1[idx_end-2]); // 加载左边连续的数据 8个
		vbody=vld1_u8((uint8_t*)&src_data_rp1[idx_end-1]); // 加载中间连续的数据 8个
		vtail=vld1_u8((uint8_t*)&src_data_rp1[idx_end]); // 加载右边连续的数据 8个

		x_buf_u=vld1q_s16(&x_tmp_buf[num_u][idx_end-1]); // 加载缓存buf 8个
		x_buf_m=vld1q_s16(&x_tmp_buf[num_m][idx_end-1]); // 加载缓存buf 8个

		y_buf_u=vld1q_s16(&y_tmp_buf[num_u][idx_end-1]); // 加载缓存buf 8个
		y_buf_m=vld1q_s16(&y_tmp_buf[num_m][idx_end-1]); // 加载缓存buf 8个

		src_l_u8_h=vhead;
		src_m_u8_h=vbody;
		src_r_u8_h=vtail;

		// 2、卷积运算
		x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
		v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
		y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
		v_grady =vsubq_s16(y_buf_d,y_buf_u);

		// 3、数学运算
		// 3、1处理LOW部分
		v_gradx_q=vmovl_s16(vget_low_s16(v_gradx));
		v_grady_q=vmovl_s16(vget_low_s16(v_grady));

		mod_sq_s32_q=vmulq_s32(v_gradx_q, v_gradx_q);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_q,v_grady_q); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_lo=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_lo=vbslq_f32(mask, v_zeros, invmod_f32_lo); // 1/sqrt(mod_sq_s32_q)

		// 3.2、数据拆分：处理high部分数据 4个
		v_gradx_q=vmovl_s16(vget_high_s16(v_gradx));
		v_grady_q=vmovl_s16(vget_high_s16(v_grady));

		mod_sq_s32_q=vmulq_s32(v_gradx_q, v_gradx_q);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_q,v_grady_q); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_hi=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_hi=vbslq_f32(mask, v_zeros, invmod_f32_hi); // 1/sqrt(mod_sq_s32_q)

		invmod_f16_q=vmulq_n_f16(vcombine_f16(vcvt_f16_f32(invmod_f32_lo),vcvt_f16_f32(invmod_f32_hi)),127.0);
		temp_s8_q.val[0]=vmovn_s16(vcvtq_s16_f16(vmulq_f16(vcvtq_f16_s16(v_gradx),invmod_f16_q)));
		temp_s8_q.val[1]=vmovn_s16(vcvtq_s16_f16(vmulq_f16(vcvtq_f16_s16(v_grady),invmod_f16_q)));
		vst2_s8((int8_t *)&grad_vec[i*width+idx_end-1], temp_s8_q); 
	
		vst1q_s16(&x_tmp_buf[num_d][idx_end-1], x_buf_d);
		vst1q_s16(&y_tmp_buf[num_d][idx_end-1], y_buf_d);
		// 处理不能凑够一个矢量的部分
		if(idx_tail_flag)
		{
			;
		}
		else
		{
			Grad2Int8 temp;
			float mod = 1e-6;
			temp1=src_data_rp1[width-idx_tail]; 
			temp2=src_data_rp1[width-idx_tail]; 
			temp3= 0;
			int gradx=0;
			int grady=0;
			for(int k=width-idx_tail;k<width-1;k++)
			{
				
				temp3=src_data_rp1[k+1];
				x_tmp_buf[num_d][k] = temp3 - temp1;
				gradx = x_tmp_buf[num_u][k] + (x_tmp_buf[num_m][k] << 1) + x_tmp_buf[num_d][k]; 

				y_tmp_buf[num_d][k] = temp1  + (temp2 << 1) + temp3;
				grady = y_tmp_buf[num_d][k] - y_tmp_buf[num_u][k]; 
				
				temp1=temp2; 
				temp2=temp3;

				mod = sqrtf(gradx*gradx + grady*grady + 1e-6);
				temp.x = int8(gradx * 127.0 / mod); // 输出int8类型
				temp.y = int8(grady * 127.0 / mod); // 输出int8类型
				if (mod < mod_t) // 如果模很小, 梯度向量置为(0,0)
				{
					temp.x = temp.y = 0;
				}
	
				grad_vec[i*width + k] = temp;
			}
		}

	}

	return;
}
#else

// int neon优化版本 中间使用FP32计算
void SobelGrad_opt_neon_s8(unsigned char * src, Grad2Int8* grad_vec, int width, int height,float mod_t)
{

	unsigned char* src_data_rm1 = NULL; // 第r-1行数据指针
	unsigned char* src_data_r = NULL; // 第r行数据指针
	unsigned char* src_data_rp1 = NULL;// 第r+1行数据指针

	// 暂存buf设置：用来暂存行卷积结果
	short x_tmp_buf[3][1280]={0}; // 临时存储数据的buf
	short y_tmp_buf[3][1280]={0}; // 临时存储数据的buf
	#define IDX(n) ((n)%3) // 取余数，用来确定列卷积通道数 0 1 2
	register int num_u =0; // 暂存IDX(i-1)的值
	register int num_m =0; // 暂存IDX( i )的值
	register int num_d =0; // 暂存IDX(i+1)的值

	// 循环变量设置：每次取8个数据
	int idx_start =8; // 每行循环起始的位置
	int idx_tail=(width%8); // 不能凑够一个矢量的元素个数
	int idx_end = width-idx_tail-8; // 每行循环终止的上限
	bool idx_tail_flag=(idx_tail==0); // 判断行内数据能否被8整除

	// 暂存变量：用于解除行内数据依赖 队列形式 先入先出
	register int temp1=0; // 第1行 队列头部  可能会复用
	register int temp2=0; // 第1行 队列头部  可能会复用
	register int temp3=0; // 第1行 队列头部  可能会复用
	register int temp1_once=0; // 第2行 队列头部 
	register int temp2_once=0; // 第2行 队列头部 
	register int temp3_once=0; // 第2行 队列头部 


	// ------------------ 先计算前 2 行的行内卷积，保存到 buf 中  ----------------------
	src_data_rm1 = &src[0]; // 第0行数据指针
	src_data_r = &src[width]; // 第1行数据指针
	
	temp1=src_data_rm1[0]; 
	temp2=src_data_rm1[1]; 
	temp1_once=src_data_r[0]; 
	temp2_once=src_data_r[1]; 
	for(int j = 1; j < width-1 ; ++j )
	{	
		temp3=src_data_rm1[j+1]; // 队列尾部tail 后进后出
		temp3_once=src_data_r[j+1]; // 队列尾部tail 后进后出

		// 前两行计算 grad_x x方向卷积
		x_tmp_buf[0][j] = temp3 - temp1;
		x_tmp_buf[1][j] = temp3_once - temp1_once;

		// 前两行计算 grad_y x方向卷积
		y_tmp_buf[0][j] = temp1  + ( temp2 << 1 ) + temp3;
		y_tmp_buf[1][j] = temp1_once  + ( temp2_once << 1 ) + temp3_once;
	
		// 队列变换：Out of queue < [temp1<temp2<temp3] <  incoming queue 
		temp1 = temp2; // 头部元素出队列，队列中部元素从temp2移到temp1位置
		temp2 = temp3; // 队列尾部元素从temp3移到temp2位置
		temp1_once = temp2_once; 
		temp2_once = temp3_once; 
	}

	// 处理固定的数据
	int mod_t_sq=int(mod_t*mod_t); // 模阈值的平方
	float32x4_t v_zeros=vmovq_n_f32(0.0); // 0矢量
	int32x4_t mod_t_sq_s32_q= vld1q_dup_s32(&mod_t_sq); // load数据到矢量寄存器

	// 每算一行新的，就相当凑齐 3 行，可以做行内的卷积，并更新更新暂存buf
	for(int i = 1; i < height-1; i++ )
	{
		src_data_rp1 = &src[(i+1)*width];// 第r+1行数据指针

		num_u=IDX(i-1);
		num_d=IDX(i+1);
		num_m=IDX(i);

		//由于通过队列采用矢量的形式解除行内重复访存，行首和行尾需要单独处理，用一个矢量处理不够8个的数
					
		// 1、加载数据 vhead vbody vtail 作为矢量队列 想到与标量temp1-temp3	队列变换：Out of queue < [vhead<vbody<vtail] <  incoming queue 
		uint8x8_t vhead=vld1_u8((uint8_t*)&src_data_rp1[0]); // 加载左边连续的数据 8个 
		uint8x8_t vbody=vld1_u8((uint8_t*)&src_data_rp1[1]); // 加载中间连续的数据 8个
		uint8x8_t vtail=vld1_u8((uint8_t*)&src_data_rp1[2]); // 加载右边连续的数据 8个
	
		int16x8_t x_buf_u=vld1q_s16(&x_tmp_buf[num_u][1]); // 加载列卷积对应的第1个通道的数据 8个
		int16x8_t x_buf_m=vld1q_s16(&x_tmp_buf[num_m][1]); // 加载列卷积对应的第2个通道的数据 8个

		int16x8_t y_buf_u=vld1q_s16(&y_tmp_buf[num_u][1]); // 加载列卷积对应的第1个通道的数据 8个
		int16x8_t y_buf_m=vld1q_s16(&y_tmp_buf[num_m][1]); // 加载列卷积对应的第2个通道的数据 8个

		uint8x8_t src_l_u8_h=vhead; // l:left 行卷积左边的数据 u:uint8 h:half 64bit 4个元素  q:128bit
		uint8x8_t src_m_u8_h=vbody; // m:midle 行卷积中间的数据 u:uint8 h:half 64bit 4个元素  q:128bit
		uint8x8_t src_r_u8_h=vtail; // r:riaght 行卷积右边的数据 u:uint8 h:half 64bit 4个元素  q:128bit

		// 2、卷积运算: 求梯度 gradx和grady
		int16x8_t x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
		int16x8_t v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
		int16x8_t y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
		int16x8_t v_grady =vsubq_s16(y_buf_d,y_buf_u);

		// 3、数学运算：求出 正弦 和 余弦
		
		// 3.1、数据拆分：处理low部分数据 4个
		int32x4_t v_gradx_h=vmovl_s16(vget_low_s16(v_gradx));
		int32x4_t v_grady_h=vmovl_s16(vget_low_s16(v_grady));


		int32x4_t mod_sq_s32_q=vmulq_s32(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		float32x4_t invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		uint32x4_t mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q); // 1/sqrt(mod_sq_s32_q)
	
		int16x4x2_t temp_q_lo; // 存储结果
		temp_q_lo.val[0]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_gradx_h,7),v_gradx_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		temp_q_lo.val[1]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_grady_h,7),v_grady_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		//int8x8x2_t vzip_s8()// s16 x 8 -> s8x8x2 vst1_s8_x2
		
		// v_gradx=vmulq_n_s16(v_gradx,127);
		// v_grady=vmulq_n_s16(v_grady,127);
		
		// 3.2、数据拆分：处理high部分数据 4个
		v_gradx_h=vmovl_s16(vget_high_s16(v_gradx));
		v_grady_h=vmovl_s16(vget_high_s16(v_grady));


		mod_sq_s32_q=vmulq_s32(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // mod_sq_s32_q < mod_t_sq_s32_q
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q); // 1/sqrt(mod_sq_s32_q)
	
		int16x4x2_t temp_q_hi; // 存储结果
		temp_q_hi.val[0]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_gradx_h,7),v_gradx_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		temp_q_hi.val[1]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_grady_h,7),v_grady_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4

		// 4、数据写回
		int8x8x2_t temp_int8_h;
		temp_int8_h.val[0]=vqmovn_s16(vcombine_s8(temp_q_lo.val[0],temp_q_hi.val[0])); // s16 x 4 -> s16 x 8 -> s8 x 8
		temp_int8_h.val[1]=vqmovn_s16(vcombine_s8(temp_q_lo.val[1],temp_q_hi.val[1])); // s16 x 4 -> s16 x 8 -> s8 x 8
		vst2_s8((int8_t *)&grad_vec[i*width+1], temp_int8_h); // 数据写到内存 交织形式
		
		
		// 发送列卷积对应的第3个通道的数据到内存 8个
		vst1q_s16(&x_tmp_buf[num_d][1], x_buf_d);
		vst1q_s16(&y_tmp_buf[num_d][1], y_buf_d);

		vbody=vld1_u8((uint8_t*)&src_data_rp1[8]); // 加载中间连续的数据 8个
		for(int j=idx_start; j < idx_end; j+=8)
		{	
			// 数据预存储 实际效果看编译器 这里只是提示编译器
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&x_tmp_buf[num_d][j]), "i"(1024));
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&y_tmp_buf[num_d][j]), "i"(1024));
			__asm__ volatile("prfm pstl2strm, [%0, #(%1)]"::"r"(&grad_vec[i*width+j]), "i"(1024));	
		
			// 1、加载数据
			vtail=vld1_u8((uint8_t*)&src_data_rp1[j+8]); // 加载连续的数据到队列尾部 8个
			
			// 数据拼接
			src_l_u8_h=vext_u8(vhead, vbody, 7); // 拼接
			src_m_u8_h=vbody; 
			src_r_u8_h=vext_u8(vbody, vtail, 1); // 拼接
			vhead=vbody;
			vbody=vtail;
		
			x_buf_u=vld1q_s16(&x_tmp_buf[num_u][j]); // 加载缓存buf 8个
			x_buf_m=vld1q_s16(&x_tmp_buf[num_m][j]); // 加载缓存buf 8个

			y_buf_u=vld1q_s16(&y_tmp_buf[num_u][j]); // 加载缓存buf 8个
			y_buf_m=vld1q_s16(&y_tmp_buf[num_m][j]); // 加载缓存buf 8个
	
			// 2、卷积运算
			x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
			v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
			y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
			v_grady =vsubq_s16(y_buf_d,y_buf_u);

			// 3、数学运算
			// 3.1、处理low部分
			v_gradx_h=vmovl_s16(vget_low_s16(v_gradx));
			v_grady_h=vmovl_s16(vget_low_s16(v_grady));


			mod_sq_s32_q=vmulq_s32(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
			mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
			invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
			mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
			invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);
		
			temp_q_lo.val[0]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_gradx_h,7),v_gradx_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
			temp_q_lo.val[1]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_grady_h,7),v_grady_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		
		    // 3.2、处理high部分
			v_gradx_h=vmovl_s16(vget_high_s16(v_gradx));
			v_grady_h=vmovl_s16(vget_high_s16(v_grady));


			mod_sq_s32_q=vmulq_s32(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
			mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
			invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
			mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
			invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);
		
			temp_q_hi.val[0]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_gradx_h,7),v_gradx_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
			temp_q_hi.val[1]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_grady_h,7),v_grady_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4

			// 4、数据写回
			temp_int8_h.val[0]=vqmovn_s16(vcombine_s8(temp_q_lo.val[0],temp_q_hi.val[0])); // s16 x 4 -> s16 x 8 -> s8 x 8
			temp_int8_h.val[1]=vqmovn_s16(vcombine_s8(temp_q_lo.val[1],temp_q_hi.val[1])); // s16 x 4 -> s16 x 8 -> s8 x 8
			vst2_s8((int8_t *)&grad_vec[i*width+j], temp_int8_h); 

			vst1q_s16(&x_tmp_buf[num_d][j], x_buf_d);
			vst1q_s16(&y_tmp_buf[num_d][j], y_buf_d);		
		}
		// 处理剩余的数据	8+(width%8)个数据
		// 1、加载数据 
		vhead=vld1_u8((uint8_t*)&src_data_rp1[idx_end-2]); // 加载左边连续的数据 8个
		vbody=vld1_u8((uint8_t*)&src_data_rp1[idx_end-1]); // 加载中间连续的数据 8个
		vtail=vld1_u8((uint8_t*)&src_data_rp1[idx_end]); // 加载右边连续的数据 8个

		x_buf_u=vld1q_s16(&x_tmp_buf[num_u][idx_end-1]); // 加载缓存buf 8个
		x_buf_m=vld1q_s16(&x_tmp_buf[num_m][idx_end-1]); // 加载缓存buf 8个

		y_buf_u=vld1q_s16(&y_tmp_buf[num_u][idx_end-1]); // 加载缓存buf 8个
		y_buf_m=vld1q_s16(&y_tmp_buf[num_m][idx_end-1]); // 加载缓存buf 8个

		src_l_u8_h=vhead;
		src_m_u8_h=vbody;
		src_r_u8_h=vtail;

		// 2、卷积运算
		x_buf_d=vreinterpretq_s16_u16(vsubl_u8(src_r_u8_h,src_l_u8_h)); 
		v_gradx =vaddq_s16(vaddq_s16(x_buf_u, x_buf_m),vaddq_s16(x_buf_m, x_buf_d)); 
		y_buf_d =vreinterpretq_s16_u16(vaddq_u16(vaddl_u8(src_l_u8_h, src_m_u8_h),vaddl_u8(src_m_u8_h, src_r_u8_h))); 
		v_grady =vsubq_s16(y_buf_d,y_buf_u);

		// 3、数学运算
		// 3、1处理LOW部分
		v_gradx_h=vmovl_s16(vget_low_s16(v_gradx));
		v_grady_h=vmovl_s16(vget_low_s16(v_grady));


		mod_sq_s32_q=vmulq_s32(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);
	
		temp_q_lo.val[0]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_gradx_h,7),v_gradx_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		temp_q_lo.val[1]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_grady_h,7),v_grady_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		
		// 3.2、处理high部分
		v_gradx_h=vmovl_s16(vget_high_s16(v_gradx));
		v_grady_h=vmovl_s16(vget_high_s16(v_grady));


		mod_sq_s32_q=vmulq_s32(v_gradx_h, v_gradx_h);  // mod_sq_s32_q= gradx*gradx
		mod_sq_s32_q=vmlaq_s32(mod_sq_s32_q,v_grady_h,v_grady_h); // mod_sq_s32_q=gradx_sq+grady*grady
		invmod_f32_q=vrsqrteq_f32(vcvtq_f32_s32(mod_sq_s32_q));
		mask = vcltq_s32(mod_sq_s32_q, mod_t_sq_s32_q);  // v1 < v2
		invmod_f32_q=vbslq_f32(mask, v_zeros, invmod_f32_q);
	
		temp_q_hi.val[0]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_gradx_h,7),v_gradx_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4
		temp_q_hi.val[1]=vqmovn_s32(vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(vshlq_n_s32(v_grady_h,7),v_grady_h)),invmod_f32_q))); // f32 -> S32 -> S16 x 4


		// 4、数据写回
		temp_int8_h.val[0]=vqmovn_s16(vcombine_s8(temp_q_lo.val[0],temp_q_hi.val[0])); // s16 x 4 -> s16 x 8 -> s8 x 8
		temp_int8_h.val[1]=vqmovn_s16(vcombine_s8(temp_q_lo.val[1],temp_q_hi.val[1])); // s16 x 4 -> s16 x 8 -> s8 x 8
		vst2_s8((int8_t *)&grad_vec[i*width+idx_end-1], temp_int8_h); 
	
		vst1q_s16(&x_tmp_buf[num_d][idx_end-1], x_buf_d);
		vst1q_s16(&y_tmp_buf[num_d][idx_end-1], y_buf_d);
		// 处理不能凑够一个矢量的部分
		if(idx_tail_flag)
		{
			;
		}
		else
		{
			Grad2Int8 temp;
			float mod = 1e-6;
			temp1=src_data_rp1[width-idx_tail]; 
			temp2=src_data_rp1[width-idx_tail]; 
			temp3= 0;
			int gradx=0;
			int grady=0;
			for(int k=width-idx_tail;k<width-1;k++)
			{
				
				temp3=src_data_rp1[k+1];
				x_tmp_buf[num_d][k] = temp3 - temp1;
				gradx = x_tmp_buf[num_u][k] + (x_tmp_buf[num_m][k] << 1) + x_tmp_buf[num_d][k]; 

				y_tmp_buf[num_d][k] = temp1  + (temp2 << 1) + temp3;
				grady = y_tmp_buf[num_d][k] - y_tmp_buf[num_u][k]; 
				
				temp1=temp2; 
				temp2=temp3;

				mod = sqrtf(gradx*gradx + grady*grady + 1e-6);
				temp.x = int8(gradx * 127.0 / mod); // 输出int8类型
				temp.y = int8(grady * 127.0 / mod); // 输出int8类型
				if (mod < mod_t) // 如果模很小, 梯度向量置为(0,0)
				{
					temp.x = temp.y = 0;
				}
	
				grad_vec[i*width + k] = temp;
			}
		}

	}

	return;
}



#endif

// int4版本
void SobelGrad4bit(cv::Mat &src, uint8_t *grad_vec, float mod_t, int num_bits)
{
	float scale = (1 << (num_bits - 1)) - 1;// 8bit:127 4bit:7
	int width = src.cols;
	int height = src.rows;
	//int widthStep = src->widthStep;

	//unsigned char* src_data = (unsigned char*)src->imageData;
//#pragma omp parallel for
	for (int r = 1; r < height - 1; r++)
	{
		unsigned char* src_data_rm1 = src.ptr<unsigned char>(r - 1); // 第r-1行数据指针
		unsigned char* src_data_r = src.ptr<unsigned char>(r - 0); // 第r行数据指针
		unsigned char* src_data_rp1 = src.ptr<unsigned char>(r + 1);// 第r+1行数据指针
		for (int c = 1; c < width - 1; c++)
		{
			cv::Point2f temp;
			int gradx = src_data_rm1[c + 1] - src_data_rm1[c - 1] +
				2 * src_data_r[c + 1] - 2 * src_data_r[c - 1] +
				src_data_rp1[c + 1] - src_data_rp1[c - 1];

			int grady = src_data_rp1[c - 1] - src_data_rm1[c - 1] +
				2 * src_data_rp1[c] - 2 * src_data_rm1[c] +
				src_data_rp1[c + 1] - src_data_rm1[c + 1];

			float mod = sqrt(gradx*gradx + grady * grady + 1e-6);

			temp.x = gradx / mod;
			temp.y = grady / mod;

			if (mod < mod_t) // 如果模很小, 梯度向量置为(0,0)
			{
				temp.x = temp.y = 0;
			}

			Grad2Int8 temp_grad;
			temp_grad.x = int8(round(temp.x * scale)); // multiply 127 instead of 128, note: 128 can cause overflow
			temp_grad.y = int8(round(temp.y * scale));



			// 1xxx 0000   0000 0xxx | 0000 1111(0xf)
			// 1xxx 0000   1111 1xxx | 0000 1111(0xf)
			// 0xxx 0000   0000 0xxx | 0000 1111(0xf)
			// 0xxx 0000   1111 1xxx | 0000 1111(0xf)

			temp_grad.x=temp_grad.x<<4;
			temp_grad.y=(temp_grad.y<<4)>>4 | 0xf;
			uint8_t grad=temp_grad.x&temp_grad.y;

			grad_vec[r*width + c] = grad;
		}
	}

	return;
}


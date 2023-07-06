STRINGIFY(
    __kernel void calc_grad(__global uchar  *left_ir_in_ptr, 
                            __global uchar  *right_ir_in_ptr,
                            __global char  *left_grad_ptr,
                            __global char  *right_grad_ptr,
                                    int     mod_thresh,
                                    int     width,
                                    int     height,
                                    int     win_size
                            )
    {
        // 无异步拷贝版本:left和right在不同的线程中算 0.39ms
        /* ------------------ 预处理 ---------------- */
        int gx    = get_global_id(0) << 2;
        int gy    = get_global_id(1);

        __global uchar*  ir_in_ptr = left_ir_in_ptr;
        __global char*  grad_ptr = left_grad_ptr;
    

        // 防止过多计算
        if (gx >= (width<<1) || gy >= height-1 || gy==0) // 图像上下边界不做计算，超出宽度不做计算
        {
            return;
        }

        if (gx >= width) // 此部分线程处理Right Ir
        {
            gx = gx - width;
            ir_in_ptr = right_ir_in_ptr;
            grad_ptr = right_grad_ptr;
        } 

        
        char4 bound_mask = (char4)(0xff); // 边界处理掩码
        if (gx >= (width-4)) // 用于每行的末端数据置零
        {
            bound_mask = (char4)(0xff, 0xff, 0, 0);
        }


        float4 vec_zeros = (float4)(0); // 0
        float4 vec_coef = (float4)(7.0); // 定点系数         
        int4 vec_mod_t_sq = (int4)(mul24(mod_thresh,mod_thresh)); // 模平方阈值
                 
        // 计算load的数据地址 r1表示中间行，r0表示上一行，r2表示下一行
        int r1_idx = mad24(gy, width, gx);
        int r0_idx = r1_idx-width;
        int r2_idx = r1_idx+width;

        // 存储store的偏移地址
        int hf_win = win_size / 2;  
        int stride = mad24(hf_win, 3, width)+1;
        int dst_idx = mad24(gy, stride, gx) + mad24(hf_win, 2, 1); 

        /* ------------------ 处理left ---------------- */
        // 向量加载，每次load行方向的8个元素
        int8 vec_r0 = convert_int8(vload8(0, ir_in_ptr + r0_idx)); 
        int8 vec_r1 = convert_int8(vload8(0, ir_in_ptr + r1_idx));
        int8 vec_r2 = convert_int8(vload8(0, ir_in_ptr + r2_idx));
         
          // y方向卷积
        int8 vec_temp=vec_r0 + (vec_r1 << (int8)(1)) + vec_r2;

        // x方向卷积 : 数据错位拆分
        int4 vec_c0 = vec_temp.lo;
        int4 vec_c1 = (int4)(vec_temp.s1234);
        int4 vec_c2 = (int4)(vec_temp.s2345);

        int4 vec_gradx = vec_c2 - vec_c0;
      
        // y方向卷积
        vec_temp=vec_r2 - vec_r0;

        // x方向卷积 : 数据错位拆分
        vec_c0 = vec_temp.lo;
        vec_c1 = (int4)(vec_temp.s1234);
        vec_c2 = (int4)(vec_temp.s2345);

        int4 vec_grady = vec_c0 +(vec_c1 << (int4)(1)) + vec_c2;

        int4 vec_mod_sq = mul24(vec_gradx, vec_gradx)+mul24(vec_grady, vec_grady); // mod的平方
        float4 vec_invmod = vec_coef * native_rsqrt(convert_float4(vec_mod_sq)); // 1/mod
        vec_invmod =  select(vec_invmod, vec_zeros, vec_mod_sq < vec_mod_t_sq); 
        
        char4 vec_temp_x = convert_char4(convert_int4(round(vec_invmod * convert_float4(vec_gradx))));
        char4 vec_temp_y = convert_char4(convert_int4(round(vec_invmod * convert_float4(vec_grady))));

        vec_temp_x = vec_temp_x << (char4)(4);
        vec_temp_y = vec_temp_y & (char4)(0xf);
        char4 vec_grad = vec_temp_x | vec_temp_y;

        vec_grad = vec_grad & bound_mask; // 通过掩码过滤掉越界的数据
        vstore4(vec_grad, 0, grad_ptr + dst_idx);
    }

)
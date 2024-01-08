# 开发事项
2023/3/17 ：版本1.0 浮点优化版本耗时2.8ms左右，int8类型耗时3.1ms左右，测试平台SM8250
2023/3/27 : 版本1.1 浮点neon版本耗时1.32ms左右
2023/3/28 : 版本1.2 int8 neon版本耗时1.32ms左右，测试条件SM8250 average time of running 2000000 times
2023/4/13 ：版本1.3 int8 opencl版本耗时0.5ms左右，测试平台SM8250 Adreno GPU
2023/9/13 ：版本1.4 int8 opencl版本耗时3.8,2.0,1.3ms左右，测试平台Adreno GPU version,610(qcm6125),620(LITO765G),630(SDM845)
# 编译路径设置
1、更改build.sh中NDK绝对路径
2、更改CMakeLists中opencv绝对路径
3、更改CMakeLists的testbench文件名可选择arm或gpu


注：不使用wapper的dlopen版本

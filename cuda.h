// cuda.h
#ifndef CUDA_H
#define CUDA_H

#include "baseline.h"

// 声明一个将会在 a.cu 文件中定义的函数
// 这个函数是C++和CUDA代码之间的桥梁
void gaussian_elimination_cuda_wrapper(matrix& m);

#endif // CUDA_H
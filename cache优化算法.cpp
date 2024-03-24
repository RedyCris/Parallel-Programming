#include <iostream>
#include <chrono>
using namespace std;
#define ull unsigned long long int
const int N = 5000;
ull matrix[N][N];
ull vector[N];
// 计算矩阵每列与给定向量的内积（cache优化算法）
ull* columnVectorInnerProductCacheOptimized() {
    static ull result[N] = { 0 }; // 内积结果存储数组

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            result[row] += matrix[row][col] * vector[col];
        }
    }
    return result;
}
 
int main() {
    // 对矩阵赋值
    for (int i = 0; i < N; ++i) {
        vector[i] = i;
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = i * N + j; // 这里可以根据你的需求改变赋值逻辑
        }
    }

    
    int iterations = 10; // 重复执行内积计算10000000次

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ull* result = columnVectorInnerProductCacheOptimized();
    }
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "10000000次计算总时间: " << duration.count() << " microseconds" << std::endl;
    cout << endl;

    return 0;
}
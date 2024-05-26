#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;

#define ull unsigned long long int

const ull N = pow(2, 24);
vector<ull> a(N);
int LOOP = 1;

void init()
{
    for (ull i = 0; i < N; i++)
        a[i] = i;
}

void ordinary()
{
    auto start = chrono::high_resolution_clock::now();
    for (int l = 0; l < LOOP; l++)
    {
        ull sum = 0;
        for (int i = 0; i < N; i++)
            sum += a[i];
    }
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::milli> duration = end - start;
    cout << "ordinary: " << duration.count()  << "ms" << endl;
}

void optimize()
{
    auto start = chrono::high_resolution_clock::now();
    for (int l = 0; l < LOOP; l++)
    {
        ull sum1 = 0,sum2=0,sum=0;
        for (int i = 0; i < N - 1; i += 2)
        {
            sum1 += a[i];
            sum2 += a[i + 1];
        }
        sum = sum1 + sum2;
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::milli> duration = end - start;
    cout << "optimize: " << duration.count() << "ms" << endl;
}

int main()
{
    init();
    ordinary();
    optimize();

    return 0;
}

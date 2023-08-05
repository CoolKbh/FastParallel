#include <iostream>
#include <Eigen/Dense>
#include <arm_neon.h>

int main()
{
    Eigen::MatrixXd m1(2, 2), m2(2, 2);
    m1 << 1, 2, 3, 4;
    m2 << 2, 3, 4, 5;
    std::cout << m1 + m2 << std::endl;
}
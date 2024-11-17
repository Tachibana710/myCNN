#include <iostream>

#include <eigen3/Eigen/Dense>

int main(){
    std::cout << "Hello World!" << std::endl;
    Eigen::MatrixXd m(2,2);
    m << 1, 2,
         3, 4;
    std::cout << m << std::endl;
    return 0;
}
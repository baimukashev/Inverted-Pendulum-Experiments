//
// Created by arms on 1/24/19.
//

#ifndef REACTIONWHEEL_EKF_H
#define REACTIONWHEEL_EKF_H
#include <Eigen/Eigen>

class EKFv {
public:
    EKFv(){};
    EKFv(double dt, double Q,double R1,double R2,double p , double v );
    void predict();
    void update(double z,double v);
    void calculate(double z,double v);
    void get_x(double *x);
private:
    Eigen::Matrix2d F;
    Eigen::Vector2d B;
    Eigen::Matrix2d H;
    Eigen::Matrix2d P;
    Eigen::Matrix2d K;
    Eigen::Vector2d x;
    double Q;
    Eigen::Matrix2d R;
    Eigen::Matrix2d S;

};

class EKFv1D : public EKFv{
public:
    EKFv1D(double dt, double Q,double R,double p , double v );
    void update(double z);
    void calculate(double z);
    void predict();
    void get_x(double *x);
private:
    Eigen::Matrix2d F;
    Eigen::Vector2d B;
    Eigen::RowVector2d H;
    Eigen::Matrix2d P;
    Eigen::Vector2d K;
    Eigen::Vector2d x;
    double Q;
    double R;
    double S;
};

#endif //REACTIONWHEEL_EKF_H

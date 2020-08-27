//
// Created by arms on 1/24/19.
//

#include "EKF.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
EKFv::EKFv(double dt, double Q, double R1, double R2, double p, double v ){
    this->Q = Q;
    this->R << R1,0,0,R2;
    this->x(0)=p;
    this->x(1)=v;

    this->P<<0,0,0,0;
    this->F<<1,dt,0,1;
    this->B <<dt*dt/2,dt;
    this->H <<1,0,0,1;

};
void EKFv::predict(){
    Eigen::VectorXd s = F*x;
    x= s;
    P =(F*P)*F.transpose() + (B*Q)*B.transpose();
}
void EKFv::update(double z,double v){
    Eigen::Vector2d hx;
    hx<<z,v;
    Eigen::Vector2d yy = hx-H*x;
    S = R + (H*P)*H.transpose();
    K = P*H.transpose()*S.inverse();
    Eigen::VectorXd s = x+K*yy;
    x=s;
    P = (Eigen::MatrixXd::Identity(2,2) - K*H)*P;
}

void EKFv::calculate(double z,double v) {
    predict();
    update(z,v);
}

void EKFv::get_x(double *x){
    x[0] = this->x(0);
    x[1] = this->x(1);
}

EKFv1D::EKFv1D(double dt, double Q,double R,double p , double v ){
    this->Q =Q;
    this->R = R;
    this->x(0)=p;
    this->x(1)=v;

    this->P<<0,0,0,0;
    this->F<<1,dt,0,1;
    this->B <<dt*dt/2,dt;
    this->H <<1,0;
}

void EKFv1D::update(double z){
    double hx = H*x;
    double yy = z-hx;
    S = R + (H*P)*H.transpose();
    K = P*H.transpose()/S;
    Eigen::VectorXd s = x+K*yy;
    x=s;
    P = (Eigen::MatrixXd::Identity(2,2) - K*H)*P;
}

void EKFv1D::calculate(double z){
    predict();
    update(z);
}

void EKFv1D::get_x(double *x) {
    x[0] = this->x(0);
    x[1] = this->x(1);
}

void EKFv1D::predict(){
    Eigen::VectorXd s = F*x;
    x= s;
    P =(F*P)*F.transpose() + (B*Q)*B.transpose();
}


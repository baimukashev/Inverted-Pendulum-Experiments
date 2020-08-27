#ifndef _PID_H_
#define _PID_H_

class PIDImpl;
class PID
{
    public:
        // Kp -  proportional gain
        // Ki -  Integral gain
        // Kd -  derivative gain
        // dt -  loop interval time
        // max - maximum value of manipulated variable
        // min - minimum value of manipulated variable
        PID( double dt, double max, double min, double offc ,double Kp, double Kd, double Ki );
        PID( double dt, double max, double min, double offc ,double constant,double Kp, double Kd, double Ki );

    // Returns the manipulated variable given a setpoint and current process value
        double calculate( double setpoint, double pv,double velo);
        ~PID();

    private:
        PIDImpl *pimpl;
};

#endif
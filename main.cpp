#include <iostream>
#include <NIDAQmxBase.h>
#include <unistd.h>
#include <Eigen/Eigen>
#include <string.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include "pid.h"
#include "EKF.h"

#include <torch/torch.h>

#include <torch/script.h>

#include <ATen/ATen.h>

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cudnn.h>

#define DAQmxErrChk(functionCall)                   \
        { if(DAQmxFailed(error=(functionCall)) )    \
            {                                       \
                goto Error;                         \
            }                                       \
        }

inline int32_t DAQmxBaseWriteAnalogScalarF64(TaskHandle taskHandle, bool32 autoStart, float64 timeout, float64 value, bool32 *reserved)
{
    return DAQmxBaseWriteAnalogF64(taskHandle, 1, static_cast<bool32>(false), timeout, DAQmx_Val_GroupByChannel, &value, nullptr, reserved);
}

inline void Sleep(int msec)
{
    std::this_thread::sleep_for(std::chrono::microseconds(msec));
}

using namespace std;

TaskHandle taskHandleM1 = nullptr;
TaskHandle taskHandleM2 = nullptr;

TaskHandle taskHandleEncoder1 = nullptr;  // y link // m1 encoder blue
TaskHandle taskHandleEncoder2 = nullptr;  // x static // m2 encoder red
TaskHandle taskHandleEncoder3 = nullptr;  //m1   // static base encoder
TaskHandle taskHandleEncoder4 = nullptr;  //m2   // link encoder

TaskHandle taskHandleGyroPower1 = nullptr;  // link
TaskHandle taskHandleGyroPower2 = nullptr;  // static
TaskHandle taskHandleGyro1 = nullptr;       // gyro1
TaskHandle taskHandleGyro2 = nullptr;       // gyro2

TaskHandle taskHandleCurrent1 = nullptr;
TaskHandle taskHandleCurrent2 = nullptr;

void encoderPosition(double *encoderPos);
double readGyro(TaskHandle gyroTask);

int main() {

    //////////////////////////////////////////////////
    // CHECK MODEL and GPU
    //////////////////////////////////////////////////

    cout << "Checking GPUs: " << endl;

    cout << "CUDA is available: " << torch::cuda::is_available() << endl;
    cout << "GPU Device Number: " << torch::cuda::device_count() << endl;

    cout << "Checking Model: " << endl;

    int wlen = 10;

    auto inputs_sample = torch::ones({ 1, wlen, 6}, at::kCUDA);
    torch::jit::script::Module module = torch::jit::load("/home/arms/CLionProjects/reactionWheelRNN_NEW/models_FF_new/RNN_W5_FF9.pt", at::kCUDA);
    module.to(at::kCUDA);
    auto outputs_sample = module.forward({inputs_sample}).toTensor();

    cout << "Model uploaded " << endl;

    //////////////////////////////////////////////////
    // SIMULATION PARAMS
    //////////////////////////////////////////////////

    double Tfinal = 10;
    double Ts = 0.01;

    bool fault_added = true;

    auto fault_time = 500;

    auto fault_type = 0;  // ## 0 ## 1 ## 2 ## 3

    auto fault_value = 3; // ## 0 ## +pi ## -pi ## last last_value_before_fault[0][0]

    float last_value_before_fault [1][6];


    //////////////////////////////////////////////////
    // sensor setup
    //////////////////////////////////////////////////

    ofstream ferr("error.txt"); //    ofstream outputfile("output.txt");
    ofstream outputfile_plot("output_plot.txt");

    int32 error=0; // to check the error
    char errBuff[2048] = {0}, errBuff2[2048] ={0};   // buffer to store the error
    double initValue[4]={0,0,0,0};    // this values assigned to encoder as initial value
    auto zIndex = static_cast<bool32>(false); // needed to create sensors

    double currents[2] = {0};
    EKFv encoder1(Ts,50,2e-5,20e-2,0.0,0.0);
    EKFv encoder2(Ts,50,2e-5,20e-2,0.0,0.0);
    EKFv1D encoder3(Ts,100,1e-6,0.0,0.0);
    EKFv1D encoder4(Ts,100,1e-6,0.0,0.0);


    std::vector<std::string> devices{"Dev1", "Dev2"};
    for (const auto &dev : devices) {
        if (DAQmxBaseResetDevice(dev.c_str())) {
            cout << "cannot initialize " << dev << endl;
        }
    }

    //encoders
    DAQmxErrChk(DAQmxBaseCreateTask("",&taskHandleEncoder1));
    DAQmxErrChk(DAQmxBaseCreateCIAngEncoderChan( taskHandleEncoder1,
                                                 "Dev1/ctr0",
                                                 "",
                                                 DAQmx_Val_X4,
                                                 zIndex,
                                                 0,
                                                 DAQmx_Val_ALowBLow,
                                                 DAQmx_Val_Radians,
                                                 1024,
                                                 static_cast<float64>(initValue[0]),
                                                 nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleEncoder1)); // start task 1

    DAQmxErrChk(DAQmxBaseCreateTask("",&taskHandleEncoder2));
    DAQmxErrChk(DAQmxBaseCreateCIAngEncoderChan( taskHandleEncoder2,
                                                 "Dev1/ctr1",
                                                 "",
                                                 DAQmx_Val_X4,
                                                 zIndex,
                                                 0,
                                                 DAQmx_Val_ALowBLow,
                                                 DAQmx_Val_Radians,
                                                 1024,
                                                 static_cast<float64>(initValue[1]),
                                                 nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleEncoder2)); // start task 2

    DAQmxErrChk(DAQmxBaseCreateTask("",&taskHandleEncoder3));
    DAQmxErrChk(DAQmxBaseCreateCIAngEncoderChan( taskHandleEncoder3,
                                                 "Dev2/ctr1",
                                                 "",
                                                 DAQmx_Val_X4,
                                                 zIndex,
                                                 0,
                                                 DAQmx_Val_ALowBLow,
                                                 DAQmx_Val_Radians,
                                                 1024,
                                                 static_cast<float64>(initValue[2]),
                                                 nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleEncoder3)); // start task 1

    DAQmxErrChk(DAQmxBaseCreateTask("",&taskHandleEncoder4));
    DAQmxErrChk(DAQmxBaseCreateCIAngEncoderChan( taskHandleEncoder4,
                                                 "Dev2/ctr0",
                                                 "",
                                                 DAQmx_Val_X4,
                                                 zIndex,
                                                 0,
                                                 DAQmx_Val_ALowBLow,
                                                 DAQmx_Val_Radians,
                                                 1024,
                                                 static_cast<float64>(initValue[3]),
                                                 nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleEncoder4)); // start task 4

    // motor 1
    DAQmxErrChk(DAQmxBaseCreateTask("motor1",&taskHandleM1));

    DAQmxErrChk(DAQmxBaseCreateAOVoltageChan(taskHandleM1,"Dev1/ao0","",-10,10,DAQmx_Val_Volts,nullptr));
    DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM1, 0, 0.0, 0, nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleM1));

    // motor 2
    DAQmxErrChk(DAQmxBaseCreateTask("motor2",&taskHandleM2));

    DAQmxErrChk(DAQmxBaseCreateAOVoltageChan(taskHandleM2,"Dev2/ao0","",-10,10,DAQmx_Val_Volts,nullptr));
    DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM2, 0, 0.0, 0, nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleM2));

    // gyros power 1
    DAQmxErrChk(DAQmxBaseCreateTask("", &taskHandleGyroPower1));
    DAQmxErrChk(
            DAQmxBaseCreateAOVoltageChan(taskHandleGyroPower1, "Dev1/ao1", "", -5.0, 5.0, DAQmx_Val_Volts, nullptr));
    double val = 0.0;
    DAQmxErrChk(
            DAQmxBaseWriteAnalogF64(taskHandleGyroPower1, 1, 0, 0.01, DAQmx_Val_GroupByChannel, &val, nullptr, nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleGyroPower1));
    val = 3.0;
    DAQmxErrChk(
            DAQmxBaseWriteAnalogF64(taskHandleGyroPower1, 1, 0, 0.01, DAQmx_Val_GroupByChannel, &val, nullptr, nullptr));


    // gyros power 2
    DAQmxErrChk(DAQmxBaseCreateTask("", &taskHandleGyroPower2));
    DAQmxErrChk(
            DAQmxBaseCreateAOVoltageChan(taskHandleGyroPower2, "Dev2/ao1", "", -5.0, 5.0, DAQmx_Val_Volts, nullptr));
    val = 0.0;
    DAQmxErrChk(
            DAQmxBaseWriteAnalogF64(taskHandleGyroPower2, 1, 0, 0.01, DAQmx_Val_GroupByChannel, &val, nullptr, nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleGyroPower2));
    val = 3.0;
    DAQmxErrChk(
            DAQmxBaseWriteAnalogF64(taskHandleGyroPower2, 1, 0, 0.01, DAQmx_Val_GroupByChannel, &val, nullptr, nullptr));

    //gyros 1
    DAQmxErrChk(DAQmxBaseCreateTask("", &taskHandleGyro1));
    DAQmxErrChk(
            DAQmxBaseCreateAIVoltageChan(taskHandleGyro1, "Dev1/ai2", "", DAQmx_Val_RSE, 0.4, 2.6, DAQmx_Val_Volts,
                                         nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleGyro1));

    // gyros 2
    DAQmxErrChk(DAQmxBaseCreateTask("", &taskHandleGyro2));
    DAQmxErrChk(
            DAQmxBaseCreateAIVoltageChan(taskHandleGyro2, "Dev2/ai2", "", DAQmx_Val_RSE, 0.4, 2.6, DAQmx_Val_Volts,
                                         nullptr));
    DAQmxErrChk(DAQmxBaseStartTask(taskHandleGyro2));



    // CHECK ENCODER

    double encoderPos[4];
    double x[2] = {0.0,0.0},y[2]= {0.0,0.0},z[2]={0.0,0.0},w[2]={0.0,0.0}, zeros[2]= {-0.38656, 0.30373};
    double m1=0,m2=0,a=0,b=0;
    double gyroValue1,gyroValue2;
    std::vector<double> state(6,0.0);

    //Sleep(3e6);
    for(int j =0;j<30;j++){

        //THIS LOOP is to put the encoders to proper locations
        auto t1 = std::chrono::high_resolution_clock::now(); // timingvalueM[1]
        encoderPosition(encoderPos); // read sensor data;

        //motorCurrent(currents,taskHandleCurrent1,taskHandleCurrent2);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()*1e-9; // elapsed time in seconds for computations
        std::cout << 5.0e-1*j<<'\t'<<encoderPos[1]<< '\t'<<'\t'<< encoderPos[0] << std::endl;
        Sleep((5.0e-1 -(int)elapsed)*1e6);

    }

    std::cout<<std::endl;
    std::cout<<"STARTING COUNT DOWN"<<std::endl;

    std::cout<<std::endl;
    for(int j=0;j<4;j++) {
        std::cout << 4 - j << std::endl;
        Sleep(1e6);
    }
    std::cout<<std::endl;



    //////////////////////////////////////////////////
    // MAIN CONTROL LOOP
    //////////////////////////////////////////////////

//    auto input_hist = torch::zeros({ 1, 600, 6}, at::kCPU);
//    auto input_seq = torch::zeros({ 1, 20, 6}, at::kCPU);

//    torch::Tensor tens = torch::rand({1,6});

    float tens [1][6];
    float input_hist [1][1000][6];
    float input_seq [1][wlen][6];

    at::Tensor inputs;

    for(int j=0;j<(int)(Tfinal/Ts);j++){

        auto t1 = std::chrono::high_resolution_clock::now(); // timingvalueM[1]

        encoderPosition(encoderPos); // read sensor data;

        gyroValue1= readGyro(taskHandleGyro1);
        gyroValue2 = readGyro(taskHandleGyro2);

        encoder1.calculate(encoderPos[0],(gyroValue1+0.1082));
        encoder2.calculate(encoderPos[1],(gyroValue2-0.01480));
        encoder3.calculate(encoderPos[2]);
        encoder4.calculate(encoderPos[3]);

        encoder1.get_x(w);
        encoder2.get_x(x);
        encoder3.get_x(y);
        encoder4.get_x(z);

        tens[0][0] = x[0];  //-x[0];
        tens[0][1] = w[0];  // w[0];
        tens[0][2] = x[1];  //-x[1];
        tens[0][3] = w[1];  // w[1];
        tens[0][4] = y[1];  // z[1];
        tens[0][5] = z[1];  // y[1];

        // STORE ARRAY

        for (int dim1 = 0; dim1 < 6; dim1++) {
            input_hist[0][j][dim1] = tens[0][dim1];
        }

        // store values for encoder values in case of stuck
        if (j == (wlen-1)){
            for (int dim1 = 0; dim1 < 6; dim1++) {
                last_value_before_fault[0][dim1] = tens[0][dim1];
            }
        }


        // wait until 20 values are saved
        if (j>=wlen) {

            // APPLY CURRENT to the motor

            // APPLY FAULT

            if((fault_added == true)&&(j >= fault_time)){

                if (fault_type == 0){

                    //cout << "ENCODER 1 " << endl;
                    // position
                    if (fault_value == 0){

                        input_hist[0][j][0] = 0.0;

                    } else if (fault_value == 1){

                        input_hist[0][j][0] = 3.14;

                    } else if (fault_value == 2){

                        input_hist[0][j][0] = -3.14;

                    } else if (fault_value == 3){

                        input_hist[0][j][0] = last_value_before_fault[0][0];
                    }

                    input_hist[0][j][2] = 0.0;

                }
                else if (fault_type == 1){

                    //cout << "ENCODER 2 " << endl;

                    if (fault_value == 0){

                        input_hist[0][j][1] = 0.0;

                    } else if (fault_value == 1){

                        input_hist[0][j][1] = 3.14;

                    } else if (fault_value == 2){

                        input_hist[0][j][1] = -3.14;

                    } else if (fault_value == 3){

                        input_hist[0][j][1] = last_value_before_fault[0][1];
                    }

                    input_hist[0][j][3] = 0.0;

                }
                else if (fault_type == 2){

                    //cout << "ENCODER 3 " << endl;

                    input_hist[0][j][4] = 0.0;

                }
                else if (fault_type == 3){

                    //cout << "ENCODER 4 " << endl;

                    input_hist[0][j][5] = 0.0;

                }
                else{
                    // pass
                }
            }

            // get last 20 values of input_hist
            for (int dim1 = 0; dim1 < wlen; dim1++) {
                for (int dim2 = 0; dim2 < 6; dim2++) {
                // input_seq 1 20 6
                    input_seq[0][dim1][dim2] = input_hist[0][ (j - wlen)  + dim1 ][dim2];

                }
            }

            // PRINT
            for (int dim1 = 0; dim1 < wlen; dim1++) {
                    // input_seq 1 20 6
                   // cout << input_seq[0][dim1][1] << "  ";

            }

            auto solver_time_start = std::chrono::high_resolution_clock::now();

            inputs = torch::from_blob(input_seq, {1,wlen,6});

            inputs = inputs.to(at::kCUDA);

            auto out = module.forward({inputs}).toTensor().cpu();

            auto solver_time_end = std::chrono::high_resolution_clock::now();
            auto solver_time = std::chrono::duration_cast<std::chrono::nanoseconds>(solver_time_end-solver_time_start).count()*1e-9;


            auto access = out.accessor<float,2>();


            m1 = 4*access[0][1];
            m2 = 4*access[0][0];

            cout << solver_time << endl;

            double upper_lim = 5.0;

            if(m1>=upper_lim){
                m1=upper_lim;
            }else if(m1<-upper_lim){
                m1=-upper_lim;
            }

            if(m2>=upper_lim){
                m2=upper_lim;
            }else if(m2<-upper_lim){
                m2=-upper_lim;
            }


//            m1 = 2;
//            m2 = 2;

            DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM1, 0, 0.0, m1 , nullptr)); //0.5   access[0][0] m1
            DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM2, 0, 0.0, m2, nullptr)); //0.5 access[0][1]    m2

        }

        //cout<<Ts*j <<" "<<encoderPos[0]<<" "<<encoderPos[1]<<" "<<encoderPos[2]<<" "<<encoderPos[3]<<" "<<m1<<" "<<m2<<" "<<gyroValue1<<" "<<gyroValue2<<" "<<x[0]<<" "<<x[1]<<" "<<w[0]<<" "<<w[1]<<endl;

        outputfile_plot<<Ts*j <<" "<<encoderPos[0]<<" "<<encoderPos[1]<<" "<<encoderPos[2]<<" "<<encoderPos[3]<<" "<<m1<<" "<<m2<<" "<<gyroValue1<<" "<<gyroValue2<<" "<<x[0]<<" "<<x[1]<<" "<<w[0]<<" "<<w[1]<<" "<< input_hist[0][j][0]<<" "<< input_hist[0][j][1]<<" "<< input_hist[0][j][2]<<" "<< input_hist[0][j][3]<<" "<< input_hist[0][j][4]<<" "<< input_hist[0][j][5] <<" " << y[1]<< " "<<z[1] <<endl;

        //outputfile_plot<<Ts*j <<" "<<encoderPos[0]<<" "<<encoderPos[1]<<" "<<encoderPos[2]<<" "<<encoderPos[3]<<" "<<m1<<" "<<m2<<" "<<gyroValue1<<" "<<gyroValue2<<" "<<x[0]<<" "<<x[1]<<" "<<w[0]<<" "<<w[1] <<endl;

        auto t2 = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()*1e-9; // elapsed time in seconds for computation

        //cout <<  elapsed << endl;

        Sleep((Ts-(int)elapsed)*1e6);

    }

    cout << "LOOP ENDED" << endl;


    //////////////////////////////////////////////////
    // STOP MOTORS                                ////
    //////////////////////////////////////////////////

    for(int j=0;j<1000;j++){


        DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM1, 0, 0.0, 0.0, nullptr)); //0.5
        DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM2, 0, 0.0, 0.0, nullptr)); //0.5
        Sleep(1e3);
    }

    cout <<  "STOP MOTORS !!!" << endl;

    Error :

        cout<<"Error"<<endl;

        DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM1, 0, 0.0, 0, nullptr));
        DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleM2, 0, 0.0, 0, nullptr));
        DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleGyroPower1, 0, 0, 0.0, nullptr));
        DAQmxErrChk(DAQmxBaseWriteAnalogScalarF64(taskHandleGyroPower2, 0, 0, 0.0, nullptr));

        if (DAQmxFailed(error)) {
            cout << "Error!" << endl;
            DAQmxBaseGetExtendedErrorInfo(errBuff, 2048);
            DAQmxBaseGetExtendedErrorInfo(errBuff2, 2048);
        }
        if (taskHandleEncoder1 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleEncoder1);  // stop task 1
            DAQmxBaseClearTask(taskHandleEncoder1); // stop task 1
        }
        if (taskHandleEncoder2 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleEncoder2);  // stop task 2
            DAQmxBaseClearTask(taskHandleEncoder2); // stop task 2
        }

        if (taskHandleEncoder3 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleEncoder3);  // stop task 3
            DAQmxBaseClearTask(taskHandleEncoder3); // stop task 3
        }

        if (taskHandleEncoder4 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleEncoder4);  // stop task 4
            DAQmxBaseClearTask(taskHandleEncoder4); // stop task 4
        }

        if (taskHandleM1 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleM1);  // stop task 1
            DAQmxBaseClearTask(taskHandleM1); // stop task 1
        }

        // motor 2

        if (taskHandleM2 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleM2);  // stop task 1
            DAQmxBaseClearTask(taskHandleM2); // stop task 1
        }

        if (taskHandleGyroPower1 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleGyroPower1);  // stop task 1
            DAQmxBaseClearTask(taskHandleGyroPower1); // stop task 1
        }

        if (taskHandleGyroPower2 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleGyroPower2);  // stop task 1
            DAQmxBaseClearTask(taskHandleGyroPower2); // stop task 1
        }

        if (taskHandleGyro1 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleGyro1);  // stop task 1
            DAQmxBaseClearTask(taskHandleGyro1); // stop task 1
        }

        if (taskHandleGyro2 != nullptr)
        {
            DAQmxBaseStopTask(taskHandleGyro2);  // stop task 1
            DAQmxBaseClearTask(taskHandleGyro2); // stop task 1
        }

    if(DAQmxFailed(error))

        ferr<< "DAQmxBase Error " << error << ": " << errBuff << endl;
        ferr.close();
        //outputfile.close();
        outputfile_plot.close();
        return 0;

    // END OF MAIN


}

void encoderPosition(double *encoderPos){

    float64 data;
    DAQmxBaseReadCounterScalarF64(taskHandleEncoder1,-1.0,&data, nullptr);
    encoderPos[0] = data;
    DAQmxBaseReadCounterScalarF64(taskHandleEncoder2,-1.0,&data, nullptr);
    encoderPos[1] = data;
    DAQmxBaseReadCounterScalarF64(taskHandleEncoder3,-1.0,&data, nullptr);
    encoderPos[2] = data;
    DAQmxBaseReadCounterScalarF64(taskHandleEncoder4,-1.0,&data, nullptr);
    encoderPos[3] = data;
}


double readGyro(TaskHandle gyroTask) {
    float64 timeout = 0.01;
    float64 readArr[1];
    int32 sampsPerChanRead;
    constexpr double offset = 2.274;
    constexpr double coef = 0.7;
    DAQmxBaseReadAnalogF64(gyroTask, 1, timeout, DAQmx_Val_GroupByChannel, readArr, 1, &sampsPerChanRead, nullptr);
    return 9*(readArr[0]-offset)/coef;
}
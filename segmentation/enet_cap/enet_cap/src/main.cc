/**
* Copyright (C) 2016-2017 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>

using namespace std;
using namespace std::chrono;
using namespace cv;
mutex m;
unsigned long long num_frames=0;
// constant for segmentation network
#define KERNEL_CONV "segmentation_0"
#define CONV_INPUT_NODE "ConvNd_1"
#define MAXPOOL_INPUT_NODE "MaxPool2d_1"									
#define CONV_OUTPUT_NODE "ConvNd_91"

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64, 35, 70, 102, 153, 153, 170, 220, 142, 251, 
                    130, 20, 0, 0, 0, 60, 80, 0, 11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

// comparison algorithm for priority_queue
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

// input video
// VideoCapture video;
VideoCapture capture(0);

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;
bool is_running_2 = false;
bool is_running_3 = false;
bool is_running_4 = false;
bool is_displaying = true;

queue<pair<int, Mat>> read_queue;                                               // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare> display_queue;  // display queue
int read_index = 0;                                                             // frame index of input video
int display_index = 0;                                                          // frame index to display


int dpuSetInputImageWithScale(DPUTask *task, const char* nodeName, const cv::Mat &image, float *mean, float scale, int idx)
{
    int value;
    int8_t *inputAddr;
    unsigned char *resized_data;
    cv::Mat newImage;
    float scaleFix;
    int height, width, channel;

    height = dpuGetInputTensorHeight(task, nodeName, idx);
    width = dpuGetInputTensorWidth(task, nodeName, idx);
    channel = dpuGetInputTensorChannel(task, nodeName, idx);

    if (height == image.rows && width == image.cols) {
        newImage = image;
    } else {
        newImage = cv::Mat (height, width, CV_8SC3,
                    (void*)dpuGetInputTensorAddress(task, nodeName, idx));
        cv::resize(image, newImage, newImage.size(), 0, 0, cv::INTER_LINEAR);
    }
    resized_data = newImage.data;

    inputAddr = dpuGetInputTensorAddress(task, nodeName, idx);
    scaleFix = dpuGetInputTensorScale(task, nodeName, idx);

    scaleFix = scaleFix*scale;

    if (newImage.channels() == 1) {
        for (int idx_h=0; idx_h<height; idx_h++) {
            for (int idx_w=0; idx_w<width; idx_w++) {
                for (int idx_c=0; idx_c<channel; idx_c++) {
                    value = *(resized_data+idx_h*width*channel+idx_w*channel+idx_c);
                    value = (int)((value - *(mean+idx_c)) * scaleFix);
                    inputAddr[idx_h*newImage.cols+idx_w] = (char)value;
                }
            }
        }
    } else {
        dpuProcessNormalizion(inputAddr, newImage.data, newImage.rows, newImage.cols, mean, scaleFix, newImage.step1());
    }
    return 0;
}


/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(DPUTask *task, bool &is_running) {
    // initialize the task's parameters
    DPUTensor *conv_in_tensor = dpuGetInputTensor(task, CONV_INPUT_NODE);
    int inHeight = dpuGetTensorHeight(conv_in_tensor);
    int inWidth = dpuGetTensorWidth(conv_in_tensor);

    DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(conv_out_tensor);
    int outWidth = dpuGetTensorWidth(conv_out_tensor);
    int8_t *outTensorAddr = dpuGetTensorAddress(conv_out_tensor);

    // Run detection for images in read queue
//    while (is_running) {
        // Get an image from read queue
        int index;
        Mat img;
        if (read_queue.empty() & !is_reading) {
//            mtx_read_queue.unlock();
             is_running = false;
             return;
        } else {
            index = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
//            mtx_read_queue.unlock();
        }

		float mean[3]={73.0,82.0,72.0};
		float scale = 0.022;
        // Set image into CONV Task with mean value
		dpuSetInputImageWithScale(task, (char *)MAXPOOL_INPUT_NODE, img, mean, scale,0);							   
		dpuSetInputImageWithScale(task, (char *)CONV_INPUT_NODE, img, mean, scale,0);

        // Run CONV Task on DPU
        dpuRunTask(task);

        Mat segMat(outHeight, outWidth, CV_8UC3);
        Mat showMat(inHeight, inWidth, CV_8UC3);
        for (int row = 0; row < outHeight; row++) {
            for (int col = 0; col < outWidth; col++) {
                int i = row * outWidth * 19 + col * 19;
                auto max_ind = max_element(outTensorAddr + i, outTensorAddr + i + 19);
                int posit = distance(outTensorAddr + i, max_ind);
                segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
            }
        }

        // resize to original scale and overlay for displaying
        resize(segMat, showMat, Size(inWidth, inHeight), 0, 0, INTER_NEAREST);
        for (int i = 0; i < showMat.rows * showMat.cols * 3; i++) {
            img.data[i] = img.data[i] * 0.4 + showMat.data[i] * 0.6;
        }

//        cout << "img.size:" << img.size() << endl;
//        cout << "showMat:" << showMat.size() << endl; 
//        Put image into display queue
        display_queue.push(make_pair(index, img));
//    }
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool &is_reading) {
       if (capture.isOpened())
       {
          Mat img;
          capture >> img;
          cv::resize(img, img, cv::Size(512,256), 0, 0, cv::INTER_LINEAR);          
          num_frames++;
          read_queue.push(make_pair(read_index++, img));
       }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool &is_displaying) {
//    while (is_displaying) {
        if (display_queue.empty()) {
            if (is_running_1 || is_running_2 || is_running_3 || is_running_4) {
                usleep(20);
            } else {
                is_displaying = false;
//                break;
                return;
            }
        } else if (display_index == display_queue.top().first) {
            // Display image
            imshow("Segmentation Display", display_queue.top().second);
            display_index++;
            display_queue.pop();
            if (waitKey(1) == 'q') {
                is_reading = false;
                is_running_1 = false;
                is_running_2 = false;
				is_running_3 = false;
				is_running_4 = false;
                is_displaying = false;
//                break;
        return;
            }
        }
//    }
}

/**
 * @brief Entry for runing Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char **argv) {
    // DPU Kernels/Tasks for runing SSD

    double frame_width = 0;        
    double frame_height = 0; 
    frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    cout << "Camera frame_width:" << frame_width << endl;
    cout << "Camera frame_height:" << frame_height << endl;
//    fps = capture.get(cv::CAP_PROP_FPS);
//    frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    Mat frame;
    DPUKernel *kernel_conv;
    DPUTask *task_conv_1, *task_conv_2;
  	auto total_time=0;
    string file_name = "camera";
    // Check args
//    if (argc != 1) {
//        cout << "Usage of segmentation demo: ./segmentaion file_name[string]" << endl;
//        cout << "\tfile_name: path to your video file" << file_name << endl;
//        return -1;
//    }

    // Attach to DPU driver and prepare for runing
    cout << "Display source" << file_name << endl;
    cout << "Pls press ESC when you want to quit" << endl;
    dpuOpen();
    // Create DPU Kernels and Tasks for CONV Nodes in SSD
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);
    task_conv_2 = dpuCreateTask(kernel_conv, 0);
//  Initializations
//  string file_name = argv[1];

    // Run tasks for SSD
	auto _start = system_clock::now();
  cv::namedWindow( "Segmentation Display", cv::WINDOW_AUTOSIZE );
  while (is_displaying) {
    Read(is_reading);
    runSegmentation(task_conv_1, is_running_1);
    Display(is_displaying);
    if(waitKey(900)==27)break;
    //waitKey(500);
    //Sleep(500); 
    }

//  array<thread, 3> threads = {thread(Read, ref(is_reading)),
//                                thread(runSegmentation, task_conv_1, ref(is_running_1)),
//                                thread(Display, ref(is_displaying))};
//
//  for (int i = 0; i < 3; ++i) {
//       threads[i].join();
//    }
	auto _end = system_clock::now();
	auto duration = (duration_cast<microseconds>(_end - _start)).count();
	total_time+=duration;
    cout << "Duration: " << duration/1000000.0  << " seconds" << endl;
	cout << "Performance: " << num_frames*1000000.0/duration << " FPS" << endl;
    // Destroy DPU Tasks and Kernels and free resources
    dpuDestroyTask(task_conv_1);
    dpuDestroyTask(task_conv_2);
    dpuDestroyKernel(kernel_conv);
    // Detach from DPU driver and release resources
    dpuClose();

    capture.release();

    return 0;
}

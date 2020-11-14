/**
* 漏 Copyright (C) 2016-2017 Xilinx, Inc
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

#include <deque>
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <sys/time.h>
#include <cstdio>
#include <iomanip>
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
//float means[3];

string frankfurt_images = "../Cityscapes/val/frankfurt/";
string lindau_images = "../Cityscapes/val/lindau/";
string munster_images = "../Cityscapes/val/munster/";
string append_filename = "gtFine_colorIds.png";
string path = "../Cityscapes/val/";
//float means[3]={73.0,82.0,72.0}; 

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;

queue<pair<string, Mat>> read_queue;     // read queue
mutex mtx_read_queue;     	 			 // mutex of read queue                                  


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
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
vector<string> images;
void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
                (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }
    cout << images.size() << endl;
    sort(images.begin(), images.end());
    closedir(dir);
}

/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(DPUTask *task, string filename, Mat img, int argc) {
    // initialize the task's parameters
    DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(conv_out_tensor);
    int outWidth = dpuGetTensorWidth(conv_out_tensor);
    int8_t *outTensorAddr = dpuGetTensorAddress(conv_out_tensor);
//    float mean[3]={73.0,82.0,72.0};
//    float mean[3]={104.0,84.0,86.0};
//    float mean[3]={109.0,87.0,87.0};
//    float mean[3]={116.0,102.0,102.0};
    float mean[3];
    if (argc == 1) {
       mean[0]=73.0;mean[1]=82.0;mean[2]=72.0;
    } else {
//       mean[0]=116.0;mean[1]=102.0;mean[2]=102.0;}
      mean[0]=104.0;mean[1]=84.0;mean[2]=86.0;}
    float scale = 0.022;

//     Run detection for images in read queue
    cv::namedWindow( "Display", cv::WINDOW_AUTOSIZE );

//     Set image into CONV Task with mean value
//        cout << "size:" << img.size << endl;   
	      resize(img, img, Size(outWidth, outHeight), 0, 0, INTER_NEAREST);		
//        cout << "size:" << img.size << endl;         
    cout << "mean = " << *mean << *(mean+1)<< *(mean+2)<< endl;   
	  dpuSetInputImageWithScale(task, (char *)MAXPOOL_INPUT_NODE, img, mean, scale,0);							   
 	  dpuSetInputImageWithScale(task, (char *)CONV_INPUT_NODE, img, mean, scale,0);
    // Run CONV Task on DPU
    dpuRunTask(task);

    Mat segMat(outHeight, outWidth, CV_8UC3);
    int inHeight = 512;
    int inWidth = 1024;
    Mat showMat(inHeight, inWidth, CV_8UC3);
    
    for (int row = 0; row < outHeight; row++) {
        for (int col = 0; col < outWidth; col++) {
            int i = row * outWidth * 19 + col * 19;
            auto max_ind = max_element(outTensorAddr + i, outTensorAddr + i + 19);
            int posit = distance(outTensorAddr + i, max_ind);
//                segMat.at<unsigned char>(row, col) = (unsigned char)(posit); //create a grayscale image with the class
            segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
        }
    }
//        resize(segMat, segMat, Size(2048,1024),0,0, INTER_NEAREST);
//        图像尺寸
//        cout << "size:" << segMat.size << endl;   
//        cout << "segMat:" << segMat.at<Vec3b>(0,0) << endl;      
//        resize(segMat, showMat, Size(inWidth, inHeight), 0, 0, INTER_NEAREST);
//        cv::Mat image = cv::imread("img.jpg");   

    for (int i = 0; i < segMat.rows * segMat.cols * 3; i++) {
        segMat.data[i] = img.data[i] * 0.2 + segMat.data[i] * 0.8;}
    cv::imshow( "Display", segMat );  

//        cv::imshow("Display", img);  
//        cv::waitKey(1000);
//        cv::destroyWindow("Display window");
//        image.release();        
//		    cvMoveWindow(filenamestr,200,200); //【6】将显示窗口固定在(200,200)这个位置显示都进来的图片

//        imwrite(filename, segMat); 
//    }
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(DPUTask *task, int argc) {
    cout <<"...This routine assumes all images fit into DDR Memory..." << endl;
    cout <<"...Reading Images..." << endl;
    Mat img;
    string img_result_name;
//    ListImages(frankfurt_images, images);
//        if (images.size() == 0) {
//            cerr << "\nError: No images exist in " << frankfurt_images << endl;
//            return;
//        }   else {
//            cout << "total Frankfurt images : " << images.size() << endl;
//        }
//        
//        for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
//                cout << frankfurt_images + images.at(img_idx) << endl;
//                img = imread(frankfurt_images + images.at(img_idx));                                                 
//                img_result_name = "results/frankfurt/" + images.at(img_idx);
//                img_result_name.erase(img_result_name.end()-15,img_result_name.end());
//                img_result_name.append(append_filename);
//                read_queue.push(make_pair(img_result_name, img));
//            }
//    images.clear();
    ListImages(lindau_images, images);
    if (images.size() == 0) {
        cerr << "\nError: No images exist in " << lindau_images << endl;
        return;
    }   else {
//        cout << "total Lindau images : " << images.size() << endl;
    }
    for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
//            cout << lindau_images + images.at(img_idx) << endl;
            img = imread(lindau_images + images.at(img_idx));                                                     
            img_result_name = "results/lindau/" + images.at(img_idx);
            img_result_name.erase(img_result_name.end()-15,img_result_name.end());
            img_result_name.append(append_filename);
//            cout << img_result_name << endl;
            runSegmentation(task, img_result_name, img, argc);
            if(waitKey(300)==27) while(getchar() != '\n');//getchar();
            read_queue.push(make_pair(img_result_name, img));
        }
    images.clear();
//    ListImages(munster_images, images);
//    if (images.size() == 0) {
//        cerr << "\nError: No images exist in " << lindau_images << endl;
//        return;
//    }   else {
//        cout << "total Munster images : " << images.size() << endl;
//    }
//    for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
//            cout << munster_images + images.at(img_idx) << endl;
//            img = imread(munster_images + images.at(img_idx));                                                     
//            img_result_name = "results/munster/" + images.at(img_idx);
//            img_result_name.erase(img_result_name.end()-15,img_result_name.end());
//            img_result_name.append(append_filename);
//            read_queue.push(make_pair(img_result_name, img));
//        }
//    images.clear();
    cout << "...if you want to quit, pls press ESC twice..." << endl;
}     

/**
 * @brief Entry for runing Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */

 
int main(int argc, char **argv) {
    
    if (argc == 2) {
        lindau_images = path + argv[1] + "/";
        cout << " directory : " << lindau_images << endl;
    }   else {
        cout << " directory : " << lindau_images << endl;
    }
 
    DPUKernel *kernel_conv;
    DPUTask *task_conv_1;

    // Attach to DPU driver and prepare for runing
    dpuOpen();
    // Create DPU Kernels and Tasks for CONV Nodes 
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);

    Read(task_conv_1, argc);
//    while(1){
//       array<thread, 1> threads = {thread(runSegmentation, task_conv_1, ref(is_running_1))};
//   
//       for (int i = 0; i < 1; ++i) {
//           threads[i].join();
//       }
//         cout << "one run completed" << endl;
//         if(waitKey(300)==27)break;
//    }
    
    // Destroy DPU Tasks and Kernels and free resources
    dpuDestroyTask(task_conv_1);
    dpuDestroyKernel(kernel_conv);
    // Detach from DPU driver and release resources
    dpuClose();

    return 0;
}

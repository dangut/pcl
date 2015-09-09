/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "evaluation.h"

#include<iostream>

using namespace pcl::gpu;
using namespace std;


//TUM KInect
//const float Evaluation::fx = 525.0f;
//const float Evaluation::fy = 525.0f;
//const float Evaluation::cx = 319.5f;
//const float Evaluation::cy = 239.5f;

//fr3
//const float Evaluation::fx = 535.40f;
//const float Evaluation::fy = 539.2f;
//const float Evaluation::cx = 320.1f;
//const float Evaluation::cy = 247.6f;

//////My asus (Daniel)
//const float Evaluation::fx = 543.78f;
//const float Evaluation::fy = 543.78f;
//const float Evaluation::cx = 313.45f;
//const float Evaluation::cy = 235.00f;

//////Handa
const float Evaluation::fx = 481.20f;
const float Evaluation::fy = -480.0f;
const float Evaluation::cx = 319.5f;
const float Evaluation::cy = 239.5f;

#ifndef HAVE_OPENCV

struct Evaluation::Impl {};

Evaluation::Evaluation(const std::string&) { cout << "Evaluation requires OpenCV. Please enable it in cmake-file" << endl; exit(0); }
void Evaluation::setMatchFile(const std::string&) { }
bool Evaluation::grab (double stamp, pcl::gpu::PtrStepSz<const RGB>& rgb24) { return false; }
bool Evaluation::grab (double stamp, pcl::gpu::PtrStepSz<const unsigned short>& depth) { return false; }
bool Evaluation::grab (double stamp, pcl::gpu::PtrStepSz<const unsigned short>& depth, pcl::gpu::PtrStepSz<const RGB>& rgb24) { return false; }
void Evaluation::saveAllPoses(const pcl::gpu::kinfuRGBD::KinfuTracker& kinfu, int frame_number, const std::string& logfile) const {}

#else

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<fstream>

using namespace cv;

struct Evaluation::Impl
{
   Mat depth_buffer;
   Mat rgb_buffer;
};



Evaluation::Evaluation(const std::string& folder) : folder_(folder), visualization_(false)
{   
  impl_.reset( new Impl() );

  if (folder_[folder_.size() - 1] != '\\' && folder_[folder_.size() - 1] != '/')
      folder_.push_back('/');

  
  cout << "Initializing evaluation from folder: " << folder_ << endl;
  string depth_file = folder_ + "depth_associated.txt";
  string rgb_file = folder_ + "rgb_associated.txt";
  
  readFile(depth_file, depth_stamps_and_filenames_);
  readFile(rgb_file, rgb_stamps_and_filenames_);  

  cout << "Associate: " << folder_ << endl;
  associate_depth_rgb(depth_file, rgb_file);
  

  //string associated_file = folder_ + "associated.txt";
}

void Evaluation::associate_depth_rgb(const std::string& file_depth, const std::string& file_rgb)
{
  char buffer[4096];

  string full_depth = file_depth;
  string full_rgb = file_rgb;

  ifstream iff_depth(full_depth.c_str());
  ifstream iff_rgb(full_rgb.c_str());

  if(!iff_depth || !iff_rgb)
  {
    cout << "Can't read rgbd" << file_depth << endl;
    //exit(1);
    return;
  }
  // ignore three header lines
  iff_depth.getline(buffer, sizeof(buffer));
  iff_depth.getline(buffer, sizeof(buffer));
  iff_depth.getline(buffer, sizeof(buffer));

  // ignore three header lines
  iff_rgb.getline(buffer, sizeof(buffer));
  iff_rgb.getline(buffer, sizeof(buffer));
  iff_rgb.getline(buffer, sizeof(buffer));
  accociations_.clear();  
  while (!iff_depth.eof() || !iff_rgb.eof())
  {
    Association acc;    
    iff_depth >> acc.time1 >> acc.name1;
    iff_rgb  >> acc.time2 >> acc.name2;
    accociations_.push_back(acc);
  }    
}

void Evaluation::setMatchFile(const std::string& file)
{
  string full = folder_ + file;
  ifstream iff(full.c_str());  
  std::cout << full << std::endl;
  if(!iff)
  {
    cout << "Can't read " << file << endl;
    exit(1);
  }

  accociations_.clear();  
  while (!iff.eof())
  {
    Association acc;    
    iff >> acc.time1 >> acc.name1 >> acc.time2 >> acc.name2;
    accociations_.push_back(acc);
  }  
}

void Evaluation::readFile(const string& file, vector< pair<double,string> >& output)
{
  char buffer[4096];
  vector< pair<double,string> > tmp;
  
  ifstream iff(file.c_str());
  if(!iff)
  {
    cout << "Can't read" << file << endl;
    //exit(1);
    return;
  }

  // ignore three header lines
  iff.getline(buffer, sizeof(buffer));
  iff.getline(buffer, sizeof(buffer));
  iff.getline(buffer, sizeof(buffer));
	
  // each line consists of the timestamp and the filename of the depth image
  while (!iff.eof())
  {
    double time; string name;
    iff >> time >> name;
    tmp.push_back(make_pair(time, name));
  }
  tmp.swap(output);
}
  
bool Evaluation::grab (double stamp, PtrStepSz<const RGB>& rgb24)
{  
  size_t i = static_cast<size_t>(stamp); // temporary solution, now it expects only index
  size_t total = accociations_.empty() ? rgb_stamps_and_filenames_.size() : accociations_.size();
  cout << "Grabbing" << endl;
  if ( i>= total)
      return false;
  
  string file = folder_ + (accociations_.empty() ? rgb_stamps_and_filenames_[i].second : accociations_[i].name2);

  cv::Mat bgr = cv::imread(file);
  if(bgr.empty())
      return false;     
      
  cv::cvtColor(bgr, impl_->rgb_buffer, CV_BGR2RGB);
  
  rgb24.data = impl_->rgb_buffer.ptr<RGB>();
  rgb24.cols = impl_->rgb_buffer.cols;
  rgb24.rows = impl_->rgb_buffer.rows;
  rgb24.step = impl_-> rgb_buffer.cols*3*sizeof(unsigned char);

  if (visualization_)
  {			    
	cv::imshow("Color channel", bgr);
	cv::waitKey(3);
  }
   cout << "End Grabbing" << file << endl;
  return true;  
}

bool Evaluation::grab (double stamp, PtrStepSz<const unsigned short>& depth)
{  
  size_t i = static_cast<size_t>(stamp); // temporary solution, now it expects only index
  size_t total = accociations_.empty() ? depth_stamps_and_filenames_.size() : accociations_.size();

  if ( i>= total)
      return false;

  string file = folder_ + (accociations_.empty() ? depth_stamps_and_filenames_[i].second : accociations_[i].name1);
  string str_png(".png");
  file.replace(file.find(str_png),str_png.length(),".png");
  
  cv::Mat d_img = cv::imread(file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  if(d_img.empty())
      return false;
   
  if (d_img.elemSize() != sizeof(unsigned short))
  {
    cout << "Image was not opend in 16-bit format. Please use OpenCV 2.3.1 or higher" << endl;
    exit(1);
  }

  // Datasets are with factor 5000 (pixel to m) 
  // http://cvpr.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
    
  d_img.convertTo(impl_->depth_buffer, d_img.type(), 0.2);
  depth.data = impl_->depth_buffer.ptr<ushort>();
  depth.cols = impl_->depth_buffer.cols;
  depth.rows = impl_->depth_buffer.rows;
  depth.step = impl_->depth_buffer.cols*sizeof(ushort); // 1280 = 640*2

  if (visualization_)
  {			
    cv::Mat scaled = impl_->depth_buffer/5000.0*65535;	
	cv::imshow("Depth channel", scaled);
	cv::waitKey(3);
  }
  return true;
}

bool Evaluation::grab (double stamp, PtrStepSz<const unsigned short>& depth, PtrStepSz<const RGB>& rgb24)
{
  if (accociations_.empty())
  {
    cout << "Please set match file" << endl;
    exit(0);
  }

  size_t i = static_cast<size_t>(stamp); // temporary solution, now it expects only index

  if ( i>= accociations_.size())
      return false;

  string depth_file = folder_ + accociations_[i].name1;
  string str_png(".png");
  //depth_file.replace(depth_file.find(str_png),str_png.length(),".ppm");
  string color_file = folder_ + accociations_[i].name2;
  //color_file.replace(color_file.find(str_png),str_png.length(),".ppm");
  
  cv::Mat d_img = cv::imread(depth_file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  if(d_img.empty())
      return false;
   
  if (d_img.elemSize() != sizeof(unsigned short))
  {
    cout << "Image was not opend in 16-bit format. Please use OpenCV 2.3.1 or higher" << endl;
    exit(1);
  }

  // Datasets are with factor 5000 (pixel to m) 
  // http://cvpr.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
     
  d_img.convertTo(impl_->depth_buffer, d_img.type(), 0.2);
  depth.data = impl_->depth_buffer.ptr<ushort>();
  depth.cols = impl_->depth_buffer.cols;
  depth.rows = impl_->depth_buffer.rows;
  depth.step = impl_->depth_buffer.cols*depth.elemSize(); // 1280 = 640*2

  cv::Mat bgr = cv::imread(color_file);
  if(bgr.empty())
      return false;     
      
  cv::cvtColor(bgr, impl_->rgb_buffer, CV_BGR2RGB);
  
  rgb24.data = impl_->rgb_buffer.ptr<RGB>();
  rgb24.cols = impl_->rgb_buffer.cols;
  rgb24.rows = impl_->rgb_buffer.rows;
  rgb24.step = impl_-> rgb_buffer.cols*3*sizeof(unsigned char);

  return true;  
}

void Evaluation::saveAllPoses(const pcl::gpu::kinfuRGBD::KinfuTracker& kinfu, int frame_number, const std::string& poses_logfile, const std::string& misc_logfile) const
{   
  size_t total = accociations_.empty() ? depth_stamps_and_filenames_.size() : accociations_.size();

  if (frame_number < 0)
      frame_number = (int)total;

  frame_number = std::min(frame_number, (int)kinfu.getNumberOfPoses());

  cout << "Writing " << frame_number << " poses to " << poses_logfile << endl;
  
  ofstream poses_path_file_stream(poses_logfile.c_str());
  poses_path_file_stream.setf(ios::fixed,ios::floatfield);
  
  ofstream misc_path_file_stream(misc_logfile.c_str());
  misc_path_file_stream.setf(ios::fixed,ios::floatfield);
  
  float mean_vis_odo_time = 0.f;
  float std_vis_odo_time = 0.f;
  float max_vis_odo_time = 0.f;
  
  for(int i = 0; i < frame_number; ++i)
  {
    mean_vis_odo_time += kinfu.getVisOdoTime(i)/frame_number;
    
    if  (kinfu.getVisOdoTime(i) > max_vis_odo_time)
      max_vis_odo_time = kinfu.getVisOdoTime(i);
  }
  
  for(int i = 0; i < frame_number; ++i)
    std_vis_odo_time = (kinfu.getVisOdoTime(i) - mean_vis_odo_time)*(kinfu.getVisOdoTime(i) - mean_vis_odo_time) / frame_number;
   
  std_vis_odo_time = sqrt(std_vis_odo_time);
  
  misc_path_file_stream << "Mean time per frame: " << mean_vis_odo_time << std::endl
                         << "Std time per frame: " << std_vis_odo_time << std::endl
                         << "Max time per frame: " << max_vis_odo_time << std::endl;
                         
  std::cout << "Mean time per frame: " << mean_vis_odo_time << std::endl
            << "Std time per frame: " << std_vis_odo_time << std::endl
            << "Max time per frame: " << max_vis_odo_time << std::endl;
            
  misc_path_file_stream << "timestamp" << " " 
                        << "chiTest" << " " 
                        << "timeVisodo" << " "
                        << "condition_number" << " " 
                        << "visibility_ratio" << " " 
                        << "error_sigmas_int" << " " 
                        << "error_sigmas_depth" << " " 
                        << "error_biases_int" << " " 
                        << "error_biases_depth" << " " 
                        << "error_RMSE" << " " 
                        << "trans_maxerr" << " " 
                        << "rot_maxerr"   << " " 
                        << "odo_odoKF_indexes"    << " "     
                        << "odo_curr_indexes"  << " "
                        << endl;

  for(int i = 0; i < frame_number; ++i)
  {
    Eigen::Affine3f pose = kinfu.getCameraPose(i);
    Eigen::Quaternionf q(pose.rotation());
    Eigen::Vector3f t = pose.translation();
    
    float chi_test = kinfu.getChiTest(i);
    float vis_odo_time = kinfu.getVisOdoTime(i);

    double stamp = accociations_.empty() ? depth_stamps_and_filenames_[i].first : accociations_[i].time1;

    poses_path_file_stream << stamp << " ";
    poses_path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
    poses_path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    
      
  
    misc_path_file_stream << stamp << " " 
                          << chi_test << " " 
                          << vis_odo_time << " " 
                          << kinfu.condition_numbers_[i] << " " 
                          << kinfu.visibility_ratio_[i] << " " 
                          << kinfu.error_sigmas_int_[i] << " " 
                          << kinfu.error_sigmas_depth_[i] << " " 
                          << kinfu.error_biases_int_[i] << " " 
                          << kinfu.error_biases_depth_[i] << " " 
                          << kinfu.error_RMSE_[i] << " " 
                          << std::setprecision(16) << kinfu.trans_maxerr_[i] << " " 
                          << std::setprecision(16) << kinfu.rot_maxerr_[i]   << " " 
                          << kinfu.odo_odoKF_indexes_[i]    << " "     
                          << kinfu.odo_curr_indexes_[i]  << " "
                          << endl;
  }
}


#endif /* HAVE_OPENCV */

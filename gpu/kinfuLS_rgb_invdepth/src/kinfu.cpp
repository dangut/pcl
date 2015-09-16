/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
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
 */

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfuLS_rgb_invdepth/kinfu.h>


#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV
  
  //~ #include <opencv2/gpu/gpu.hpp>
  //~ #include <pcl/gpu/utils/timers_opencv.hpp>
#endif

using namespace std;
using namespace pcl::device::kinfuRGBD;

using pcl::device::kinfuRGBD::device_cast;
//using Eigen::AngleAxisf;
//using Eigen::Array3f;
using Eigen::Vector3i;
//using Eigen::Vector3f;

//typedef double float_trafos;

//typedef Eigen::Matrix<float_trafos,6,1> Vector6f;
//typedef Eigen::Matrix<float_trafos,3,1> Vector3f;
//typedef Eigen::Matrix<float_trafos,3,3> Matrix6f;
//typedef Eigen::Matrix<float_trafos,6,1> Vector6f;


      
      
namespace pcl
{
  namespace gpu
  {
    namespace kinfuRGBD
    {
      
      typedef double float_trafos;          

      typedef Eigen::Matrix<float_trafos, 3, 3, Eigen::RowMajor> Matrix3ft;
      typedef Eigen::Matrix<float_trafos,3,1> Vector3ft;
      typedef Eigen::Matrix<float_trafos,6,1> Vector6ft;
      typedef Eigen::Matrix<float_trafos, 4, 4, Eigen::RowMajor> Matrix4ft;
      typedef Eigen::Matrix<float_trafos, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXft;
      
      typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f;
      typedef Eigen::Matrix<float,3,1> Vector3f;
      typedef Eigen::Matrix<float,6,1> Vector6f;
      typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4f;
      typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf;
    
      Matrix3ft forceOrthogonalisation(const Matrix3ft& matrix);
      
      Vector3ft rodrigues2(const Matrix3ft& matrix);  
      
      Vector6ft logMap(const Matrix3ft& R, const Vector3ft& trans);   

      Matrix4ft expMap(const Vector3ft& omega_t, const Vector3ft& v_t);  
      
      Matrix3ft expMapRot(const Vector3ft& omega_t);

      Eigen::Matrix3d skew(const Eigen::Vector3d& w);       
    }
  }
}

    

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::kinfuRGBD::KinfuTracker::KinfuTracker (const Vector3f &volume_size, const float shiftingDistance, int optim_dim, int Mestimator, int motion_model, int depth_error_type, int sigma_estimator, int weighting, int warping, int max_keyframe_count, int finest_level, int termination, float visratio, int image_filtering, int rows, int cols) : 
  cyclical_( DISTANCE_THRESHOLD, VOLUME_SIZE, VOLUME_X), rows_(rows), cols_(cols),  global_time_(0),  integration_metric_threshold_(0.f), perform_last_scan_ (false), finished_(false), lost_ (false), optim_dim_(optim_dim), Mestimator_(Mestimator), motion_model_(motion_model), depth_error_type_(depth_error_type), sigma_estimator_(sigma_estimator), weighting_(weighting), warping_(warping), max_keyframe_count_(max_keyframe_count), finest_level_(finest_level), termination_(termination), visibility_ratio_threshold_(visratio), image_filtering_(image_filtering)
{
  //const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution (VOLUME_X, VOLUME_Y, VOLUME_Z);

  volume_size_ = volume_size(0);
  std::cout << "Memory usage BEFORE creating volume" << std::endl;
  showGPUMemoryUsage();
  tsdf_volume_ = TsdfVolume::Ptr ( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize (volume_size);
  std::cout << "Memory usage AFTER creating volume" << std::endl;
  showGPUMemoryUsage();
  shifting_distance_ = shiftingDistance;

  // set cyclical buffer values
  cyclical_.setDistanceThreshold (shifting_distance_);
  cyclical_.setVolumeSize (volume_size_, volume_size_, volume_size_);
  
  setDepthIntrinsics (FOCAL_LENGTH, FOCAL_LENGTH, CENTER_X, CENTER_Y, BASELINE);
  
  init_Rcam_ = Matrix3ft::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size.cast<float_trafos>() * 0.5f - Vector3ft (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 10, 10, 0, 0};
  std::copy (iters, iters + LEVELS, visodo_iterations_);

  
  const float default_tranc_dist = 0.03f; //meters
  
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

  allocateBuffers (rows, cols);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);
  chi_tests_.reserve (30000);
  vis_odo_times_.reserve(30000);
  
  condition_numbers_.reserve(30000);
  visibility_ratio_.reserve(30000);
  error_sigmas_int_.reserve(30000);
  error_sigmas_depth_.reserve(30000);
  error_biases_int_.reserve(30000);
  error_biases_depth_.reserve(30000);
  error_RMSE_.reserve(30000);
  trans_maxerr_.reserve(30000);
  rot_maxerr_.reserve(30000);  
  odo_rmats_.reserve(30000);
  odo_tvecs_.reserve(30000);
  //odo_covmats_.reserve(30000);
        
  odo_odoKF_indexes_.reserve(30000);      
  odo_curr_indexes_.reserve(30000);

  reset ();
  
  // initialize cyclical buffer
  cyclical_.initBuffer(tsdf_volume_);
  
  last_estimated_rotation_= Matrix3ft::Identity ();
  last_estimated_translation_= volume_size.cast<float_trafos>() * 0.5f - Vector3ft (0, 0, volume_size (2) / 2 * 1.2f);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy, float baseline)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy;  
  baseline_ = (baseline == -1) ? 50.f : baseline;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::setInitialCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ().cast<float_trafos>();
  init_tcam_ = pose.translation ().cast<float_trafos>();
  //reset (); // (already called in constructor)
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;  
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuRGBD::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuRGBD::KinfuTracker::rows ()
{
  return (rows_);
}

void
pcl::gpu::kinfuRGBD::KinfuTracker::extractAndSaveWorld ()
{
  
  //extract current volume to world model
  PCL_INFO("Extracting current volume...");
  cyclical_.checkForShift(tsdf_volume_, getCameraPose (), volume_size_, true, true, true); // this will force the extraction of the whole cube.
  PCL_INFO("Done\n");
  
  finished_ = true; 
  
  int cloud_size = 0;
  
  cloud_size = cyclical_.getWorldModel ()->getWorld ()->points.size();
  
  char world_filename[32];
  sprintf (world_filename, "world_%06d.pcd", (int) global_time_);  
  
  if (cloud_size <= 0)
  {
    PCL_WARN ("World model currently has no points. Skipping save procedure.\n");
    return;
  }
  else
  {
    PCL_INFO ("Saving current world to world.pcd with %d points.\n", cloud_size);
    pcl::io::savePCDFile<pcl::PointXYZI> (world_filename, *(cyclical_.getWorldModel ()->getWorld ()), true);
    return;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::reset ()
{
  cout << "in reset function!" << std::endl;
  
  if (global_time_)
    PCL_WARN ("Reset\n");
    
  // dump current world to a pcd file
  /*
  if (global_time_)
  {
    PCL_INFO ("Saving current world to current_world.pcd\n");
    pcl::io::savePCDFile<pcl::PointXYZI> ("current_world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
    // clear world model
    cyclical_.getWorldModel ()->reset ();
  }
  */
  
  // clear world model
  cyclical_.getWorldModel ()->reset ();
  
  
  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();
  chi_tests_.clear();
  vis_odo_times_.clear();
  condition_numbers_.clear();
  visibility_ratio_.clear();
  error_sigmas_int_.clear();
  error_sigmas_depth_.clear();
  error_biases_int_.clear();
  error_biases_depth_.clear();
  error_RMSE_.clear();
  trans_maxerr_.clear();
  rot_maxerr_.clear();  
  odo_rmats_.clear();
  odo_tvecs_.clear();
  odo_odoKF_indexes_.clear(); 
  odo_curr_indexes_.clear();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);
  chi_tests_.push_back(0.f);
  vis_odo_times_.push_back(0.f);
  condition_numbers_.push_back(0.f);
  visibility_ratio_.push_back(0.f);
  error_sigmas_int_.push_back(0.f);
  error_sigmas_depth_.push_back(0.f);
  error_biases_int_.push_back(0.f);
  error_biases_depth_.push_back(0.f);
  error_RMSE_.push_back(0.f);
  trans_maxerr_.push_back(0.f);
  rot_maxerr_.push_back(0.f);  
  odo_rmats_.push_back(init_Rcam_);
  odo_tvecs_.push_back(init_tcam_);
  odo_odoKF_indexes_.push_back(0);      
  odo_curr_indexes_.push_back(0);  

  tsdf_volume_->reset ();
  
  // reset cyclical buffer as well
  cyclical_.resetBuffer (tsdf_volume_);
  
  if (color_volume_) // color integration mode is enabled
    color_volume_->reset ();    
  
  // reset estimated pose
  last_estimated_rotation_= Matrix3ft::Identity ();
  last_estimated_translation_= Vector3ft (volume_size_, volume_size_, volume_size_) * 0.5f - Vector3ft (0, 0, volume_size_ / 2 * 1.2f);
  
  
  lost_=false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::allocateBuffers (int rows, int cols)
{    

  std::cout << "Memory usage BEFORE allocating buffers" << std::endl;
  showGPUMemoryUsage();
  depths_curr_.resize (LEVELS);
  intensities_curr_.resize(LEVELS);

  depths_prev_.resize (LEVELS);
  intensities_prev_.resize(LEVELS);
  
  depths_keyframe_.resize (LEVELS);
  depths_keyframe_filtered_.resize (LEVELS);
  intensities_keyframe_filtered_.resize (LEVELS);
  intensities_keyframe_.resize(LEVELS);

  xGradsInt_keyframe_.resize(LEVELS);
  yGradsInt_keyframe_.resize(LEVELS);
  xGradsDepth_keyframe_.resize(LEVELS);
  yGradsDepth_keyframe_.resize(LEVELS);

  warped_depths_curr_.resize(LEVELS);
  warped_intensities_curr_.resize(LEVELS);  
  warped_level_depths_curr_.resize(LEVELS);
  warped_level_intensities_curr_.resize(LEVELS);  

  res_intensities_.resize(LEVELS);
  res_depths_.resize(LEVELS);
  
  vmaps_g_prev_.resize (LEVELS);
  nmaps_g_prev_.resize (LEVELS);
  
  vmaps_g_prev_[0].create (rows*3, cols);
  
  projected_transformed_points_.create(rows*2,cols);
  depth_warped_in_curr_.create(rows,cols);
  
  depths_prefilt_.create(rows,cols);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    intensities_curr_[i].create (pyr_rows, pyr_cols);
    depths_curr_[i].create (pyr_rows, pyr_cols);

    intensities_prev_[i].create (pyr_rows, pyr_cols);
    depths_prev_[i].create (pyr_rows, pyr_cols);  
    
    intensities_keyframe_[i].create (pyr_rows, pyr_cols);
    depths_keyframe_[i].create (pyr_rows, pyr_cols);  
    depths_keyframe_filtered_[i].create (pyr_rows, pyr_cols);  
    intensities_keyframe_filtered_[i].create (pyr_rows, pyr_cols);  

    xGradsInt_keyframe_[i].create (pyr_rows, pyr_cols);
    yGradsInt_keyframe_[i].create (pyr_rows, pyr_cols);
    xGradsDepth_keyframe_[i].create (pyr_rows, pyr_cols);
    yGradsDepth_keyframe_[i].create (pyr_rows, pyr_cols); 
    
    warped_depths_curr_[i].create(pyr_rows, pyr_cols);
    warped_intensities_curr_[i].create(pyr_rows, pyr_cols);
    warped_level_depths_curr_[i].create(pyr_rows, pyr_cols);
    warped_level_intensities_curr_[i].create(pyr_rows, pyr_cols);
    
    res_intensities_[i].create(pyr_rows * pyr_cols);
    res_depths_[i].create(pyr_rows * pyr_cols);

    //With this Vis-Odo approach Vmaps and nmaps are just used for visualisation.
    //if we select a coarse pyramid level raycast will take less time
    if (i==PYR_RAYCAST)
    {
      vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
      nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
    }
  }  
  
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  gbuf_.create (27, 20*60);
  sumbuf_.create (27);
  
  std::cout << "Memory usage AFTER allocating buffers" << std::endl;
  showGPUMemoryUsage();
}

inline void 
pcl::gpu::kinfuRGBD::KinfuTracker::convertTransforms (Matrix3ft& rotation_in_1, Matrix3ft& rotation_in_2, Vector3ft& translation_in_1, Vector3ft& translation_in_2, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out_1, float3& translation_out_2)
{
  Matrix3f rotation_in_1f = rotation_in_1.cast<float>();
  Matrix3f rotation_in_2f = rotation_in_2.cast<float>();
  Vector3f translation_in_1f = translation_in_1.cast<float>();
  Vector3f translation_in_2f = translation_in_2.cast<float>();
  
  rotation_out_1 = device_cast<Mat33> (rotation_in_1f);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2f);
  translation_out_1 = device_cast<float3>(translation_in_1f);
  translation_out_2 = device_cast<float3>(translation_in_2f);
}

inline void 
pcl::gpu::kinfuRGBD::KinfuTracker::convertTransforms (Matrix3ft& rotation_in_1, Matrix3ft& rotation_in_2, Vector3ft& translation_in, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out)
{
  Matrix3f rotation_in_1f = rotation_in_1.cast<float>();
  Matrix3f rotation_in_2f = rotation_in_2.cast<float>();
  Vector3f translation_in_f = translation_in.cast<float>();
  
  rotation_out_1 = device_cast<Mat33> (rotation_in_1f);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2f);
  translation_out = device_cast<float3>(translation_in_f);
}

inline void 
pcl::gpu::kinfuRGBD::KinfuTracker::convertTransforms (Matrix3ft& rotation_in, Vector3ft& translation_in, Mat33& rotation_out, float3& translation_out)
{
  Matrix3f rotation_in_f = rotation_in.cast<float>();
  Vector3f translation_in_f = translation_in.cast<float>();
  
  rotation_out = device_cast<Mat33> (rotation_in_f);  
  translation_out = device_cast<float3>(translation_in_f);
}

inline void 
pcl::gpu::kinfuRGBD::KinfuTracker::prepareImages (const DepthMap& depth_raw, const View& colors_raw, const Intr& cam_intrinsics)
{  
  computeIntensity (colors_raw, intensities_curr_[0]);

  if (depth_error_type_ == DEPTH)
    convertDepth2Float(depth_raw, depths_curr_[0]);
  else if (depth_error_type_ == INV_DEPTH)
    convertDepth2InvDepth(depth_raw, depths_curr_[0]);
    
  //if (depth_error_type_ == DEPTH)
    //convertDepth2Float(depth_raw, depths_prefilt_);
  //else if (depth_error_type_ == INV_DEPTH)
    //convertDepth2InvDepth(depth_raw, depths_prefilt_);
    
  //bilateralFilter(depths_prefilt_, depths_curr_[0], 0.05f);
  
  for (int i = 1; i < LEVELS; ++i)
  {
    pyrDownIntensity (intensities_curr_[i-1], intensities_curr_[i]);
    pyrDownDepth (depths_curr_[i-1], depths_curr_[i]);
  }   
}

inline void
pcl::gpu::kinfuRGBD::KinfuTracker::saveCurrentImages  ()
{  
  for (int i = 0; i < LEVELS; ++i)
  {   
    copyImages (depths_curr_[i], intensities_curr_[i], depths_prev_[i], intensities_prev_[i]);
  }  
}


inline void
pcl::gpu::kinfuRGBD::KinfuTracker::saveCurrentImagesAsKeyframes  ()
{
  for (int i = 0; i < LEVELS; ++i)
  {   
    copyImages (depths_curr_[i], intensities_curr_[i], depths_keyframe_[i], intensities_keyframe_[i]);
  }  
}


inline void
pcl::gpu::kinfuRGBD::KinfuTracker::savePreviousImagesAsKeyframes  ()
{
  for (int i = 0; i < LEVELS; ++i)
  {   
    copyImages (depths_prev_[i], intensities_prev_[i], depths_keyframe_[i], intensities_keyframe_[i]);
  }  
}



inline bool 
pcl::gpu::kinfuRGBD::KinfuTracker::estimateVisualOdometry  (const Intr cam_intrinsics, Matrix3ft& resulting_rotation , Vector3ft& resulting_translation)
{ 

  pcl::StopWatch t4;

  Matrix3ft previous_rotation = resulting_rotation;
  Vector3ft previous_translation = resulting_translation;
  
  Matrix3ft current_rotation;
  Vector3ft current_translation;
  
   //std::cout << "previous_rotation" << previous_rotation << std::endl
            //<< "previous_translation" << previous_translation << std::endl;

  //Initialize rotation and translation as if velocity was kept constant    
  if ((global_time_ > 1) && (motion_model_ == CONSTANT_VELOCITY))
  {
    Matrix3ft delta_rot_prev;
    Vector3ft delta_trans_prev;
    
    ////////////////This is not robust to changes in frame rate
    //Matrix3ft Rm2_inv = rmats_[global_time_-2].inverse();
    //Vector3ft   t_m2     = tvecs_[global_time_-2];
    //Matrix3ft Rm1 = rmats_[global_time_-1];
    //Vector3ft   t_m1 = tvecs_[global_time_-1];

    //delta_rot_prev = Rm2_inv*Rm1;
    //delta_trans_prev = Rm2_inv*(t_m1 - t_m2);
    //delta_rot_prev = forceOrthogonalisation(delta_rot_prev);
    
    
    //std::cout << "Trafo conc" << std::endl
              //<< "delta rot_prev: " << delta_rot_prev << std::endl
              //<< "delta trans prev: " << delta_trans_prev << std::endl;
    ////////////////////
    
    ///////////////This one is robust to changes in frame rate. estimate velocity at the end of odometry estmate
    Vector3ft v_t = velocity_*delta_t_;
    Vector3ft omega_t = omega_*delta_t_;
    Matrix4ft trafo_const_vel = expMap(omega_t, v_t);
    
    delta_rot_prev = trafo_const_vel.block(0,0,3,3);
    delta_trans_prev = trafo_const_vel.block(0,3,3,1);    
    ///////////////
    
    current_translation = previous_rotation*delta_trans_prev + previous_translation;
    current_rotation = previous_rotation*delta_rot_prev;
  }
  else
  {
    current_translation = previous_translation;
    current_rotation = previous_rotation;
  }

  ///////////////////////////////////////////////
  float timeBuildSystem = 0.f;
  float timeWarping = 0.f;
  float timeSigma = 0.f;
  float timeError = 0.f;
  float timeChiSquare = 0.f;
  float timeGradients = 0.f;
  float timeFilters = 0.f;

  Eigen::Matrix<double, B_SIZE, B_SIZE, Eigen::RowMajor> A_total;
  Eigen::Matrix<double, B_SIZE, 1> b_total;

  float sigma_int_ref, sigma_depth_ref;
  
  float chi_test = 1.f;
  sigma_int_ref = 5.f;
  sigma_depth_ref = 0.0025f;
  float sigma_int, sigma_depth;
  float bias_int, bias_depth;
  
  bool last_iter_flag = false;
  
  if (image_filtering_ == FILTER_GRADS)  
  {
    timeFilters += bilateralFilter(depths_keyframe_[0], depths_keyframe_filtered_[0], 2.f*sigma_depth_ref);
    timeFilters += bilateralFilter(intensities_keyframe_[0], intensities_keyframe_filtered_[0], sigma_int_ref);
    
    for (int i = 1; i < LEVELS; ++i)
    {     
        timeWarping+=pyrDownDepth (depths_keyframe_filtered_[i-1], depths_keyframe_filtered_[i]);
        timeWarping+=pyrDownIntensity (intensities_keyframe_filtered_[i-1], intensities_keyframe_filtered_[i]);
    }
  }
    
  for (int level_index = LEVELS-1; level_index>=finest_level_; --level_index)
  {
    int iter_num = visodo_iterations_[level_index];//+(level_index==finest_level_);
    float chi_square_prev = 1.f;
    float chi_square = 1.f;
    float Ndof = (float) ( (rows_ >> level_index) * (cols_ >> level_index) );
    float chi_test = 1.f;  
    float RMSE;
    float RMSE_prev;
    
    //KF intensity and depth/inv_depth  
    IntensityMapf& intensity_keyframe = intensities_keyframe_[level_index];
    DepthMapf& depth_keyframe = depths_keyframe_[level_index];     
    
    DepthMapf& depth_keyframe_filtered = depths_keyframe_filtered_[level_index];
    IntensityMapf& intensity_keyframe_filtered = intensities_keyframe_filtered_[level_index];
    
    //We need gradients on the KF maps. 
    GradientMap& xGradInt_keyframe = xGradsInt_keyframe_[level_index];
    GradientMap& yGradInt_keyframe = yGradsInt_keyframe_[level_index];
    GradientMap& xGradDepth_keyframe = xGradsDepth_keyframe_[level_index];
    GradientMap& yGradDepth_keyframe = yGradsDepth_keyframe_[level_index];    

    //Residuals between warped_{}_curr and {}_keyframe
    DeviceArray<float>&  res_intensity = res_intensities_[level_index];
    DeviceArray<float>&  res_depth = res_depths_[level_index];

    {
      //ScopeTime t5 ("estimateGrads");            
      if (image_filtering_ == FILTER_GRADS)  
      {
        timeGradients += computeGradientIntensity(intensity_keyframe_filtered, xGradInt_keyframe, yGradInt_keyframe);
        timeGradients += computeGradientDepth(depth_keyframe_filtered, xGradDepth_keyframe, yGradDepth_keyframe);
      }
      else
      {
        timeGradients += computeGradientIntensity(intensity_keyframe, xGradInt_keyframe, yGradInt_keyframe);
        timeGradients += computeGradientDepth(depth_keyframe, xGradDepth_keyframe, yGradDepth_keyframe); 
      }
    }
    
    Matrix3ft cam_rot_incremental_inv;
    Matrix3ft cam_rot_incremental;
    Vector3ft cam_trans_incremental;
    
    Matrix3ft current_rotation_prov;
    Vector3ft current_translation_prov;

    // run optim for iter_num iterations (return false when lost)
    for (int iter = 0; iter < iter_num; ++iter)
    { 
      Vector3f init_vector = Vector3f::Zero();
      float3 device_current_delta_trans = device_cast<float3>(init_vector);
      float3 device_current_delta_rot = device_cast<float3>(init_vector);
      Matrix3ft inv_current_rotation = current_rotation.inverse();
      Vector3ft inv_current_translation = - current_rotation.inverse()*current_translation;      
      
      Matrix3f inv_current_rotation_f = inv_current_rotation.cast<float>();
      Vector3f inv_current_translation_f = inv_current_translation.cast<float>();
      Matrix3f current_rotation_f = current_rotation.cast<float>();  
      
      last_iter_flag =  ((level_index == finest_level_) && (iter ==  (iter_num-1)));
      
      if ((warping_ == WARP_FIRST)) //Expensive warping (warp in lvl 0, pyr down every iter)
      {
        //std::cout << "warp fisrt (on lvl 0 always) then pyr down warped images" << std::endl;
        if (depth_error_type_ == DEPTH)
        {
          timeWarping += warpIntensityWithTrafo3D  (intensities_curr_[0], warped_intensities_curr_[0], depths_keyframe_[0], 
                                                    device_cast<Mat33> (inv_current_rotation_f),  
                                                    device_cast<float3> (inv_current_translation_f), 
                                                    cam_intrinsics(0));

          timeWarping += warpDepthWithTrafo3D  (depths_curr_[0], warped_depths_curr_[0], depths_keyframe_[0], 
                                                device_cast<Mat33> (inv_current_rotation_f), 
                                                device_cast<float3> (inv_current_translation_f), 
                                                cam_intrinsics(0),
                                                device_cast<Mat33> (current_rotation_f));
        }
        else if (depth_error_type_ == INV_DEPTH)
        {
          timeWarping += warpIntensityWithTrafo3DInvDepth  (intensities_curr_[0], warped_intensities_curr_[0], depths_keyframe_[0], 
                                                            device_cast<Mat33> (inv_current_rotation_f),  
                                                            device_cast<float3> (inv_current_translation_f), 
                                                            cam_intrinsics(0));

          timeWarping += warpInvDepthWithTrafo3D  (depths_curr_[0], warped_depths_curr_[0], depths_keyframe_[0], 
                                                   device_cast<Mat33> (inv_current_rotation_f), 
                                                   device_cast<float3> (inv_current_translation_f), 
                                                   cam_intrinsics(0),
                                                   device_cast<Mat33> (current_rotation_f));
        }
        
        for (int i = 1; i < (level_index + 1); ++i)
        {     
          timeWarping+=pyrDownIntensity (warped_intensities_curr_[i-1], warped_intensities_curr_[i]);
          timeWarping+=pyrDownDepth (warped_depths_curr_[i-1], warped_depths_curr_[i]); 
        }
      
      }
      else //(warping_ == PYR_FIRST) //Cheap warping (pyr down every level, warp on curr level)
      {
        //std::cout << "pyr down first (at start of level loop), then warping" << std::endl;
        if (depth_error_type_ == DEPTH)
        {
          timeWarping += warpIntensityWithTrafo3D  (intensities_curr_[level_index], warped_intensities_curr_[level_index], depths_keyframe_[level_index], 
                                                    device_cast<Mat33> (inv_current_rotation_f),  
                                                    device_cast<float3> (inv_current_translation_f), 
                                                    cam_intrinsics(level_index));

          timeWarping += warpDepthWithTrafo3D  (depths_curr_[level_index], warped_depths_curr_[level_index], depths_keyframe_[level_index], 
                                                device_cast<Mat33> (inv_current_rotation_f), 
                                                device_cast<float3> (inv_current_translation_f), 
                                                cam_intrinsics(level_index),
                                                device_cast<Mat33> (current_rotation_f));
        }
        else if (depth_error_type_ == INV_DEPTH)
        {
          timeWarping += warpIntensityWithTrafo3DInvDepth  (intensities_curr_[level_index], warped_intensities_curr_[level_index], depths_keyframe_[level_index], 
                                                            device_cast<Mat33> (inv_current_rotation_f),  
                                                            device_cast<float3> (inv_current_translation_f), 
                                                            cam_intrinsics(level_index));

          timeWarping += warpInvDepthWithTrafo3D  (depths_curr_[level_index], warped_depths_curr_[level_index], depths_keyframe_[level_index], 
                                                   device_cast<Mat33> (inv_current_rotation_f), 
                                                   device_cast<float3> (inv_current_translation_f), 
                                                   cam_intrinsics(level_index),
                                                   device_cast<Mat33> (current_rotation_f));
        } 
      }
      
      //if (last_iter_flag == true)
        //break;

      IntensityMapf& warped_intensity_curr = warped_intensities_curr_[level_index];
      DepthMapf& warped_depth_curr = warped_depths_curr_[level_index];
      
      
      timeError += computeError (warped_intensity_curr, intensity_keyframe, res_intensity); 
      timeError += computeError (warped_depth_curr, depth_keyframe, res_depth);
      
      
      if (termination_ == CHI_SQUARED)
      {
        timeChiSquare += computeChiSquare (res_intensity, res_depth, sigma_int_ref, sigma_depth_ref, Mestimator_, chi_square, chi_test, Ndof);
        RMSE = sqrt(chi_square)/sqrt(Ndof);
        //std::cout << "lvl " << level_index << " iter " << iter << ", chiSq: " << chi_square << ",Ndof: " << Ndof << ", RMSE: "<< RMSE   << std::endl;
      
        if (!(iter == 0))
        {
          
          if (RMSE > RMSE_prev) //undo the previous increment and end iters at curr pyr
          {
            current_translation = cam_rot_incremental_inv*(current_translation - cam_trans_incremental);
            current_rotation = cam_rot_incremental_inv*current_rotation;
            std::cout << "Break in pyr " << level_index << " at iteration " << iter << std::endl; 
            break;
          }
          
          float rel_diff = abs(RMSE - RMSE_prev) / RMSE_prev;
          
          
          if (rel_diff < 0.0001) //end iters at curr pyr
          {
            std::cout << "Break in pyr " << level_index << " at iteration " << iter << std::endl; 
            break;
          }
        }     
        
        chi_square_prev = chi_square;
        RMSE_prev = RMSE;       
      }   

      sigma_int = 80.f;
      sigma_depth = 5.5f;
      bias_int = 0.f;
      bias_depth= 0.f;
  
      if (sigma_estimator_ == SIGMA_MAD)
      {
        timeSigma += computeSigmaMAD(res_intensity, sigma_int);
        timeSigma += computeSigmaMAD(res_depth, sigma_depth);
      }
      else if (sigma_estimator_ == SIGMA_PDF)
      {
        timeSigma +=  computeSigmaPdf(res_intensity, sigma_int, Mestimator_);
        timeSigma +=  computeSigmaPdf(res_depth, sigma_depth, Mestimator_);
      }
      else if (sigma_estimator_ == SIGMA_PDF_BIAS)
      {
        timeSigma +=  computeSigmaPdfWithBias(res_intensity, sigma_int, bias_int, Mestimator_);
        timeSigma +=  computeSigmaPdfWithBias(res_depth, sigma_depth, bias_depth, Mestimator_);
      }
      else if (sigma_estimator_ == SIGMA_PDF_SAMPLING)
      {
        timeSigma +=  computeSigmaPdfWithBiasDirectReduction(res_intensity, sigma_int, bias_int, Mestimator_);
        timeSigma +=  computeSigmaPdfWithBiasDirectReduction(res_depth, sigma_depth, bias_depth, Mestimator_);
      }
      else if  (sigma_estimator_ == SIGMA_CONS)
      {
        sigma_int = exp(log(sigma_int_ref) - 0.f*log(2)) ; //Substitute 0.f by curr lvl index??
        sigma_depth = exp(log(sigma_depth_ref) - 0.f*log(2)) ; //if depth_error_type = DEPTH, sigma_depth is not constant (depends on Z^2). As it is -> assumed const as for depth = 1m, for every depth
      }
          
      //std::cout << "Pyr level " << level_index << ", iter " << iter << std::endl; 
      //std::cout << "   sigma_int: "   << sigma_int << std::endl
                //<< "   sigma_depth: " << sigma_depth << std::endl
                //<< "   bias_int: "    << bias_int << std::endl
                //<< "   bias_depth: " <<  bias_depth << std::endl;

      //Optimise delta
      //bias_depth = 0.f;
      //bias_int = 0.f;
      
      timeBuildSystem += buildSystem (device_current_delta_trans, device_current_delta_rot,
                                      depth_keyframe, intensity_keyframe,
                                      xGradDepth_keyframe, yGradDepth_keyframe,
                                      xGradInt_keyframe, yGradInt_keyframe,
                                      warped_depth_curr, warped_intensity_curr,
                                      Mestimator_, depth_error_type_, weighting_,
                                      sigma_depth, sigma_int,
                                      bias_depth, bias_int,  //there is the option of substracting the bias just in the residuals used for computing IRLS weights.
                                      cam_intrinsics(level_index), B_SIZE,
                                      gbuf_, sumbuf_, A_total.data (), b_total.data ());
                                      
      pcl::StopWatch t_solve;
      
      MatrixXd A_optim(optim_dim_, optim_dim_);
      A_optim = A_total.block(0,0,optim_dim_,optim_dim_);

      MatrixXd b_optim(optim_dim_, 1);
      b_optim = b_total.block(0,0,optim_dim_,1);  

      MatrixXft result(optim_dim_, 1);
      result = A_optim.llt().solve(b_optim).cast<float_trafos>(); 

      //Ilumination correction variables affect the optimisation result when used, but apart from that we dont do anything with them.
      //This part remains equal even if we used ilumination change parameters
      Vector3ft res_trans = result.block(0,0,3,1);  
      Vector3ft res_rot = result.block(3,0,3,1);  

      //If output is B_theta^A and r_B^A
      cam_rot_incremental_inv = expMapRot(res_rot); 
      cam_rot_incremental = cam_rot_incremental_inv.inverse();
      cam_trans_incremental = -cam_rot_incremental*res_trans;
       
      
      //Transform updates are applied by premultiplying. Seems counter-intuitive but it is the correct way,
      //since at each iter we warp curr frame towards prev frame.      
      current_translation = cam_rot_incremental * current_translation + cam_trans_incremental;
      current_rotation = cam_rot_incremental * current_rotation;
            
      timeBuildSystem+=t_solve.getTime();         
    }
    
    if (last_iter_flag)
    {
      //Functions for analysis of experiments, not really need by the algorithm
      //timeError += computeError (warped_intensities_curr_[0], intensities_keyframe_[0], res_intensities_[0]); 
      //timeError += computeError (warped_depths_curr_[0], depths_keyframe_[0], res_depths_[0]);
      
      //timeSigma +=  computeSigmaPdfWithBias(res_intensities_[0], sigma_int, bias_int, Mestimator_);
      //timeSigma +=  computeSigmaPdfWithBias(res_depths_[0], sigma_depth, bias_depth, Mestimator_);
        
      //timeChiSquare += computeChiSquare (res_intensities_[0], res_depths_[0], sigma_int_ref, sigma_depth_ref, Mestimator_, chi_square, chi_test, Ndof); 
      //////////////////////
      
      float visibility_ratio_curr_to_KF = 1.f;
      float visibility_ratio_KF_to_curr = 1.f;
      
      Matrix3f current_rotation_f = current_rotation.cast<float>();
      Vector3f current_translation_f = current_translation.cast<float>();
      
      Matrix3f inv_current_rotation_f = current_rotation.inverse().cast<float>();
      Vector3f inv_current_translation_f = -inv_current_rotation_f*current_translation.cast<float>();
      
      //Visibility test to initialize new KF
      float depth_tol = 0.05; //tol = 5cm (for every dpth if depth_type = depth or at 5cm at 2m if depth_type = inv_depth
      float geom_tol;
      
      if (depth_error_type_ == DEPTH)
        geom_tol = depth_tol;        
      else if (depth_error_type_ == INV_DEPTH)
        geom_tol = depth_tol / (2.f*2.f);
        
      getVisibilityRatio(depths_curr_[0], depths_keyframe_[0],device_cast<Mat33> (current_rotation_f), device_cast<float3> (current_translation_f), 
                         cam_intrinsics(0), visibility_ratio_curr_to_KF, geom_tol, depth_error_type_); 
      std::cout << "vis ratio: " << visibility_ratio_curr_to_KF << std::endl;
      
      getVisibilityRatio(depths_keyframe_[0], depths_curr_[0], device_cast<Mat33> (inv_current_rotation_f), device_cast<float3> (inv_current_translation_f), 
                         cam_intrinsics(0), visibility_ratio_KF_to_curr, geom_tol, depth_error_type_); 
      std::cout << "vis ratio: " << visibility_ratio_KF_to_curr << std::endl;
      
      float visibility_ratio = std::min(visibility_ratio_curr_to_KF, visibility_ratio_KF_to_curr);    
      
      if ((visibility_ratio < visibility_ratio_threshold_) && (keyframe_count_ > 0))
      {
        std::cout << "visibility test failed for keyframe-curr_frame dist of  " << keyframe_count_ <<  std::endl;
        return false;
      }
      
      //if (keyframe_count_ > 8)
        //std::cout << "visibility test success till keyframe-curr_frame dist: " << keyframe_count_ <<  std::endl;
      ////////////////////////
       
      //std::cout << "sigma_int: " << sigma_int << std::endl;
      //std::cout << "sigma_depth: " << sigma_depth << std::endl;
      //std::cout << "bias_int: " << bias_int << std::endl;
      //std::cout << "bias_depth: " << bias_depth << std::endl;
      //std::cout << "RMSE: " << chi_square/Ndof << std::endl;
      
      visibility_ratio_.push_back(visibility_ratio);
      error_sigmas_int_.push_back(sigma_int);
      error_sigmas_depth_.push_back(sigma_depth);
      error_biases_int_.push_back(std::abs(bias_int));
      error_biases_depth_.push_back(std::abs(bias_depth));
      error_RMSE_.push_back(chi_square/Ndof);
      
    }          
  } 
    
  std::cout << "timeTotal: " << t4.getTime() << std::endl
              << "timeTotalSum: " << timeWarping + timeBuildSystem + timeError + timeSigma + timeChiSquare + timeGradients + timeFilters<< std::endl
              << "  time warping: " <<  timeWarping << std::endl
              << "  time build system: " <<  timeBuildSystem << std::endl
              << "  time error: " <<  timeError << std::endl
              << "  time sigma: " <<  timeSigma << std::endl
              << "  time chiSquare: " << timeChiSquare << std::endl
              << "  time gradients: " << timeGradients << std::endl
              << "  time filters: " << timeFilters << std::endl;
                            
  resulting_rotation = current_rotation;
  resulting_translation = current_translation;
            
  //Check condition number of matrix A 
  MatrixXd A_final(6, 6);
  A_final = A_total.block(0,0,6,6);
  //std::cout << "A final: " << std::endl
            //<< A_final     << std::endl;
  
  MatrixXd singval_A(6, 1);
  Eigen::JacobiSVD<MatrixXd> svdA(A_final);
  singval_A = svdA.singularValues();
  
  //std::cout << "singval A: " << singval_A << std::endl; 
  
  double max_eig = singval_A(0,0);
  double min_eig = singval_A(0,0);
  
  for (unsigned int v_indx = 1; v_indx < 6; v_indx++)
  {
      if (singval_A(v_indx,0) >  max_eig)
        max_eig = singval_A(v_indx,0);
      if  (singval_A(v_indx,0) <  min_eig)
        min_eig = singval_A(v_indx,0);
  }
  
  double A_cn = max_eig / min_eig;
  condition_numbers_.push_back(A_cn);
  
  std::cout << "Matrix A condition number is: " << max_eig / min_eig << std::endl;
  std::cout << std::endl
            << A_final.inverse().diagonal().cwiseSqrt() << std::endl;
  ////////////////////   
  
  //Covariance matrix
  Eigen::Matrix<double,6,6> pseudo_adj = Eigen::Matrix<double,6,6>::Zero();
  //pseudo adjoint bcs xi is not a twist ( translational part is yet the translation of the SE3 trafo)
  pseudo_adj.block<3,3>(0,0) = -resulting_rotation.cast<double>();
  pseudo_adj.block<3,3>(3,3) = -resulting_rotation.cast<double>();
  
  Eigen::Matrix<double,6,6> resulting_covariance = pseudo_adj*A_final.inverse()*pseudo_adj.transpose(); 
  
  Eigen::Matrix<double,3,3> cov_trans = resulting_covariance.block<3,3>(0,0);
  Eigen::Matrix<double,3,3> cov_rot = resulting_covariance.block<3,3>(3,3);
  
  MatrixXd singval_trans(3, 1);
  Eigen::JacobiSVD<MatrixXd> svd_trans(cov_trans);
  singval_trans = svd_trans.singularValues();
  
  double max_trans_eig = singval_trans(0,0);
  
  for (unsigned int v_indx = 1; v_indx < 3; v_indx++)
  {
      if (singval_trans(v_indx,0) >  max_trans_eig)
        max_trans_eig = singval_trans(v_indx,0);
  }
  
  MatrixXd singval_rot(3, 1);
  Eigen::JacobiSVD<MatrixXd> svd_rot(cov_rot);
  singval_rot = svd_rot.singularValues();
  
  double max_rot_eig = singval_rot(0,0);
  
  for (unsigned int v_indx = 1; v_indx < 3; v_indx++)
  {
      if (singval_rot(v_indx,0) >  max_rot_eig)
        max_rot_eig = singval_rot(v_indx,0);
  }
  
  std::cout << "max trans eig: " << sqrt(max_trans_eig) << std::endl
            << "max rot eig: " << sqrt(max_rot_eig) << std::endl;
  trans_maxerr_.push_back(sqrt(max_trans_eig));
  rot_maxerr_.push_back(sqrt(max_rot_eig));  
  odo_rmats_.push_back(resulting_rotation);
  odo_tvecs_.push_back(resulting_translation);
  //odo_covmats_.push_back(resulting_covariance);
        
  odo_odoKF_indexes_.push_back(curr_odoKF_index_);      
  odo_curr_indexes_.push_back(global_time_);
  
  
  Matrix3ft delta_rot_curr = previous_rotation.transpose()*current_rotation;
  Vector3ft delta_trans_curr = previous_rotation.transpose()*(current_translation - previous_translation);
    
  Vector6ft twist = logMap(delta_rot_curr, delta_trans_curr);
  velocity_ = twist.block(0,0,3,1) * (1.f / delta_t_);
  omega_ = twist.block(3,0,3,1) * (1.f / delta_t_);
  
  chi_tests_.push_back(chi_test); //chi test is crap. Im not using it
  vis_odo_times_.push_back(t4.getTime());
  

  // Vis Odo has converged
//  std::cout << "chi test: " << chi_test << std::endl;
//  
  //if (chi_test >= 0.5) 
    //std::cout << "chi test for 0.85 failed for keyframe " << keyframe_count_ <<  std::endl;
  //TODO: lost detection
  return (true);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::kinfuRGBD::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time].cast<float>();
  aff.translation () = tvecs_[time].cast<float>();
  return (aff);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float
pcl::gpu::kinfuRGBD::KinfuTracker::getChiTest (int time) const
{
  if (time > (int)chi_tests_.size () || time < 0)
    time = chi_tests_.size () - 1;
  
  return (chi_tests_[time]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float
pcl::gpu::kinfuRGBD::KinfuTracker::getVisOdoTime (int time) const
{
  if (time > (int)vis_odo_times_.size () || time < 0)
    time = vis_odo_times_.size () - 1;
  
  return (vis_odo_times_[time]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned long
pcl::gpu::kinfuRGBD::KinfuTracker::getTimestamp (int time) const
{
  if (time > (int)timestamps_.size () || time < 0)
    time = timestamps_.size () - 1;
  
  return (timestamps_[time]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::kinfuRGBD::KinfuTracker::getLastEstimatedPose () const
{
  Eigen::Affine3f aff;
  aff.linear () = last_estimated_rotation_.cast<float>();
  aff.translation () = last_estimated_translation_.cast<float>();
  return (aff);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
pcl::gpu::kinfuRGBD::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuRGBD::KinfuTracker::getGlobalTime () const
{
  return (global_time_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const pcl::gpu::kinfuRGBD::TsdfVolume& 
pcl::gpu::kinfuRGBD::KinfuTracker::volume() const 
{ 
  return *tsdf_volume_; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::kinfuRGBD::TsdfVolume& 
pcl::gpu::kinfuRGBD::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const pcl::gpu::kinfuRGBD::ColorVolume& 
pcl::gpu::kinfuRGBD::KinfuTracker::colorVolume() const
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::kinfuRGBD::ColorVolume& 
pcl::gpu::kinfuRGBD::KinfuTracker::colorVolume()
{
  return *color_volume_;
}
     
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::getImage (View& view) //const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Vector3f light_source_pose = tvecs_[tvecs_.size () - 1].cast<float>();

  LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);
  
  view.create (rows_ >> PYR_RAYCAST, cols_  >> PYR_RAYCAST );
  generateImage (vmaps_g_prev_[PYR_RAYCAST], nmaps_g_prev_[PYR_RAYCAST], light, view);
  
  //downloadAndSaveRGB(view);
  
  //For Visualizing filtered gradients
  //Intr intr_cam (fx_, fy_, cx_, cy_, baseline_);
  //createVMapFromInvDepthf( intr_cam(0), depths_keyframe_filtered_[0], vmaps_g_prev_[0]);
  //createNMap( vmaps_g_prev_[0], nmaps_g_prev_[0]); 
  //view.create (rows_, cols_ );
  //generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
  
  //view.create (rows_ >> PYR_RAYCAST, cols_  >> PYR_RAYCAST );
  //convertFloat2RGB(intensities_keyframe_filtered_[0], view);
  
  
}

void
pcl::gpu::kinfuRGBD::KinfuTracker::downloadAndSaveRGB(const DeviceArray2D<PixelRGB>& dev_image, int pyrlevel, int iterat){

	std::vector<PixelRGB> data;
	cv::Mat cv_image;
	int cols = dev_image.cols();
	int rows = dev_image.rows();
	char im_file[128];
	data.resize(cols*rows);
	int elem_step = 1;
	dev_image.download(data, cols);	
	
  cv_image.create(rows, cols, CV_8UC3);
	memcpy( cv_image.data, &data[0], data.size()*sizeof(PixelRGB));
	sprintf(im_file, "sceneView/globalTime%07d.png", global_time_);

 	cv::imwrite(im_file, cv_image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  convert (vmaps_g_prev_[PYR_RAYCAST], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuRGBD::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  convert (nmaps_g_prev_[PYR_RAYCAST], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::kinfuRGBD::KinfuTracker::initColorIntegration(int max_weight)
{     
  color_volume_ = pcl::gpu::kinfuRGBD::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );  
}


bool
pcl::gpu::kinfuRGBD::KinfuTracker::operator() (const DepthMap& depth_raw, const View& colors, float delta_t, unsigned long timestamp)
{ 
  // Intrisics of the camera
  Intr intr (fx_, fy_, cx_, cy_, baseline_);
  
  
  if ((timestamp == 0) && (global_time_ > 0)) //not taking timestamps->assume nominal frame rate
  {
    delta_t_ = 0.033333333;
    timestamps_.push_back(timestamps_[global_time_-1]+ ((unsigned long)(1e6*delta_t_)));
  }
  else
  {
    delta_t_ = delta_t;
    timestamps_.push_back(timestamp);
  }
  delta_t_ = 0.033333333;
  //pcl::ScopeTime t1 ("whole loop");

  //std::cout << "operator() : " << endl;
  // Physical volume size (meters)
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());  

  prepareImages (depth_raw, colors, intr); //This function generates the float depth or inv_depth image from raw sensor depth,  
  
  // Initialization at first frame
  
  // sync GPU device
  pcl::device::kinfuRGBD::sync ();

  Matrix3ft last_integration_rotation;     // [Ri|ti] - pos of camera, i.e.
  Vector3ft  last_integration_translation;
  
  if (global_time_ == 0) // this is the frist frame, the tsdf volume needs to be initialized
  {  
    // Initial rotation
    Matrix3ft initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera
    Matrix3ft initial_cam_rot_inv = initial_cam_rot.inverse ();
    // Initial translation
    Vector3ft   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

    // Convert pose to device types
    Mat33 device_initial_cam_rot, device_initial_cam_rot_inv;
    float3 device_initial_cam_trans;
    convertTransforms (initial_cam_rot, initial_cam_rot_inv, initial_cam_trans, device_initial_cam_rot, device_initial_cam_rot_inv, device_initial_cam_trans);

    // integrate current depth map into tsdf volume, from default initial pose.
    {
      //pcl::ScopeTime t2 ("integrate");
      integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, 
      tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), 
      getCyclicalBufferStructure (), depthRawScaled_);
    }

    last_integration_rotation = rmats_[0]; 
    last_integration_translation = tvecs_[0];
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Ray casting   
    // generate synthetic vertex and normal maps from newly-found pose.
    {
      //pcl::ScopeTime t2 ("raycast");
      raycast (intr(PYR_RAYCAST), device_initial_cam_rot, device_initial_cam_trans, 
      tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), 
      getCyclicalBufferStructure (), vmaps_g_prev_[PYR_RAYCAST], nmaps_g_prev_[PYR_RAYCAST]);
    }

    // POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
    Matrix3f rmat_f = rmats_[0].cast<float>();
    Mat33&  rotation_id = device_cast<Mat33> (rmat_f); /// Identity Rotation Matrix. Because we never rotate our volume
    float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
    MapArr& vmap_temp = vmaps_g_prev_[PYR_RAYCAST];
    MapArr& nmap_temp = nmaps_g_prev_[PYR_RAYCAST];

    {
      //pcl::ScopeTime t3 ("transformMaps");
      transformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[PYR_RAYCAST], nmaps_g_prev_[PYR_RAYCAST]);
    }

    if(perform_last_scan_)
      finished_ = true;

    saveCurrentImages(); 
    
    ++global_time_;
    
    keyframe_count_ = 0;
    curr_odoKF_index_ = 0;
    last_known_global_rotation = rmats_[global_time_ - 1];     // [Ri|ti] - pos of camera, i.e.
    last_known_global_translation = tvecs_[global_time_ - 1];   // transform from camera to global coo space for (i-1)th camera pose
    delta_rotation = Matrix3ft::Identity ();
    delta_translation = Vector3ft (0, 0, 0); 
    
    pcl::device::kinfuRGBD::sync ();
    //Matrix3ft inv_mat = rmats_[0].inverse();
    saveCurrentImagesAsKeyframes (); 

    // return and wait for next frame
    return (false);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Matrix3ft current_global_rotation;
  Vector3ft current_global_translation;
  
  //std::cout << "last trans: " << tvecs_[global_time_ - 1] << std::endl;
  //std::cout << "last rot: " << rmats_[global_time_ - 1] << std::endl;
  // Get the last-known pose
  if ( keyframe_count_ >= max_keyframe_count_ )
  {
    keyframe_count_ = 0;
    curr_odoKF_index_ = global_time_ - 1;
    last_known_global_rotation = rmats_[global_time_ - 1];     // [Ri|ti] - pos of camera, i.e.
    last_known_global_translation = tvecs_[global_time_ - 1];   // transform from camera to global coo space for (i-1)th camera pose
    delta_rotation = Matrix3ft::Identity ();
    delta_translation = Vector3ft (0, 0, 0); 
  }
  
  //Daniel: visual odometry by photogametric minimisation in I and D
  pcl::device::kinfuRGBD::sync ();  
  if  (!estimateVisualOdometry(intr, delta_rotation, delta_translation))
  {  
    if (!(keyframe_count_ == 0))
    {  
      keyframe_count_ = 0;
      curr_odoKF_index_ = global_time_ - 1;
      last_known_global_rotation = rmats_[global_time_ - 1];     // [Ri|ti] - pos of camera, i.e.
      last_known_global_translation = tvecs_[global_time_ - 1];   // transform from camera to global coo space for (i-1)th camera pose
      delta_rotation = Matrix3ft::Identity ();
      delta_translation = Vector3ft (0, 0, 0); 
      
      savePreviousImagesAsKeyframes (); 
      
      //TODO: we are lost detection: ¿chiSquare test?, ¿threshold?, ¿det(A)?
      if  (!estimateVisualOdometry(intr, delta_rotation, delta_translation))
      {
        //lost_ = true;
        //std::cout << "I am LOST!!!" << std::endl;
        //return(false);
      }        
    }
    else
    {
      //lost_ = true;
      //std::cout << "I am LOST!!!" << std::endl;
      //return(false);
    }      
  }  
  
  pcl::device::kinfuRGBD::sync ();

  current_global_translation = last_known_global_translation + last_known_global_rotation * delta_translation;
  current_global_rotation = last_known_global_rotation * delta_rotation;
  current_global_rotation = forceOrthogonalisation(current_global_rotation);
  
  //std::cout << "R*R^T no svd " << current_global_rotation*current_global_rotation.transpose() << std::endl;
  // Save newly-computed pose
  rmats_.push_back (current_global_rotation); 
  tvecs_.push_back (current_global_translation);

  // Update last estimated pose to current pairwise ICP result
  last_estimated_translation_ = current_global_translation;
  last_estimated_rotation_ = current_global_rotation;
 
  ///////////////////////////////////////////////////////////////////////////////////////////  
  // check if we need to shift
  bool has_shifted = cyclical_.checkForShift(tsdf_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_); // TODO make target distance from camera a param
  if(has_shifted)
    PCL_WARN ("SHIFTING\n");

  std::cout << "save world" << std::endl;
  //Daniel: Having bad alloc problems in some datasets: Save to HD when world size is too large to free RAM memory
  if (cyclical_.getWorldModel ()->getWorld ()->points.size() > 10000000)
  {
       extractAndSaveWorld ();
       cyclical_.getWorldModel ()->reset ();
  }  

  // get the NEW local pose as device types
  Mat33  device_current_rotation_inv, device_current_rotation;   
  float3 device_current_translation_local;   
  Matrix3ft cam_rot_local_curr_inv = current_global_rotation.inverse (); //rotation (local = global)
  convertTransforms(cam_rot_local_curr_inv, current_global_rotation, current_global_translation, 
  device_current_rotation_inv, device_current_rotation, device_current_translation_local); 
  device_current_translation_local -= getCyclicalBufferStructure()->origin_metric;   // translation (local translation = global translation - origin of cube)

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move far enought.  

  float rnorm = rodrigues2(current_global_rotation.inverse() * last_integration_rotation).norm();
  float tnorm = (current_global_translation - last_integration_translation).norm();    
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;


  // Volume integration  
  if (integrate)
  { 
    //pcl::ScopeTime t5 ("integrate");
    std::cout << "integrate" << std::endl;
    integrateTsdfVolume (depth_raw, intr, device_volume_size, 
    device_current_rotation_inv, device_current_translation_local, 
    tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), 
    getCyclicalBufferStructure (), depthRawScaled_);

    last_integration_rotation = current_global_rotation;     // [Ri|ti] - pos of camera, i.e.
    last_integration_translation = current_global_translation;
  }

  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting


  // generate synthetic vertex and normal maps from newly-found pose.
  {
    std::cout << "raycast" << std::endl;
    //pcl::ScopeTime t7 ("raycast");
    raycast (intr(PYR_RAYCAST), device_current_rotation, device_current_translation_local, 
    tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), 
    getCyclicalBufferStructure (), vmaps_g_prev_[PYR_RAYCAST], nmaps_g_prev_[PYR_RAYCAST]);


    // POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
    Matrix3f rmat_f = rmats_[0].cast<float>();
    Mat33&  rotation_id = device_cast<Mat33> (rmat_f); /// Identity Rotation Matrix. Because we never rotate our volume
    float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
    MapArr& vmap_temp = vmaps_g_prev_[PYR_RAYCAST];
    MapArr& nmap_temp = nmaps_g_prev_[PYR_RAYCAST];
    transformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[PYR_RAYCAST], nmaps_g_prev_[PYR_RAYCAST]);    

    pcl::device::kinfuRGBD::sync ();
    /////////////////////////////////////
  }

  std::cout << "extractworld" << std::endl;
  if(has_shifted && perform_last_scan_)
    extractAndSaveWorld ();

  //Matrix3ft inv_mat = rmats_[global_time_].inverse();
  // save current vertex and normal maps
  saveCurrentImages();
  keyframe_count_ ++;
  
  if ( keyframe_count_ == max_keyframe_count_ )
  {
    saveCurrentImagesAsKeyframes (); 
  }
  
  ++global_time_;

  if (color_volume_)
  {
    //pcl::ScopeTime t7 ("colorvolume");
    const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    Intr intr(fx_, fy_, cx_, cy_);

    Matrix3ft R_inv = rmats_.back().inverse();
    Vector3ft   t     = tvecs_.back();
    
    Matrix3f R_inv_f = R_inv.cast<float>();
    Vector3f   t_f   = t.cast<float>();

    Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv_f);
    float3& device_tcurr = device_cast<float3> (t_f);

    updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], 
    colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
  }


  return (true);
 }


namespace pcl
{
  namespace gpu
  {
    namespace kinfuRGBD
    {
      PCL_EXPORTS void 
      paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
      {
        pcl::device::kinfuRGBD::paint3DView(rgb24, view, colors_weight);
      }

      PCL_EXPORTS void
      mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
      {
        const size_t size = min(cloud.size(), normals.size());
        output.create(size);

        const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
        const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
        const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
        pcl::device::kinfuRGBD::mergePointNormal(c, n, o);           
      }

      Matrix3ft forceOrthogonalisation(const Matrix3ft& matrix)
      {
        Eigen::JacobiSVD<Matrix3ft> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
        Matrix3ft R = svd.matrixU() * svd.matrixV().transpose();
        return R;
      }
      
      
      Vector3ft rodrigues2(const Matrix3ft& matrix)
      {
        Eigen::JacobiSVD<Matrix3ft> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
        Matrix3ft R = svd.matrixU() * svd.matrixV().transpose();

        double rx = R(2, 1) - R(1, 2);
        double ry = R(0, 2) - R(2, 0);
        double rz = R(1, 0) - R(0, 1);

        double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
        double c = (R.trace() - 1) * 0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;

        double theta = acos(c);

        if( s < 1e-5 )
        {
          double t;

          if( c > 0 )
            rx = ry = rz = 0;
          else
          {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
              rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
          }
        }
        else
        {
          double vth = 1/(2*s);
          vth *= theta;
          rx *= vth; ry *= vth; rz *= vth;
        }
        return Eigen::Vector3d(rx, ry, rz).cast<float_trafos>();
      }


      Vector6ft logMap(const Matrix3ft& matrix, const Vector3ft& trans)
      {
        Eigen::JacobiSVD<Matrix3ft> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
        Matrix3ft R = svd.matrixU() * svd.matrixV().transpose();

        double rx = R(2, 1) - R(1, 2);
        double ry = R(0, 2) - R(2, 0);
        double rz = R(1, 0) - R(0, 1);

        double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
        double c = (R.trace() - 1) * 0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;

        double theta = acos(c);
        double theta2 = theta*theta;
        double th_by_sinth;
          
        ////////////////////////////////////////////////////////////
        if( s < 1e-5 )
        {
          double t;

          if( c > 0 )
            rx = ry = rz = 0;
          else
          {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
              rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
          }
        }
        else
        {
          double vth = 1/(2*s);
          vth *= theta;
          rx *= vth; ry *= vth; rz *= vth;
        }
        //////////////////////////////////////////////////////
        
        //Wouldnt this be equivalent and shorter?
        ///////////////////////////////////
        //if ( s < 1e-5 )
        //{
          //th_by_sinth = 1.0 + (1.0/6.0)*theta2 + (7.0/360.0)*theta2*theta2;
        //}
        //else
        //{
          //th_by_sinth = theta/s;
        //}
        
        //double vth = th_by_sinth / 2.0;
        //rx *= vth; ry *= vth; rz *= vth;
        ///////////////////////////////////////////
        
        Eigen::Vector3d omega_t = Eigen::Vector3d(rx, ry, rz);
        Eigen::Matrix3d Omega_t = skew(omega_t);
        Eigen::Matrix3d Omega_t2 = Omega_t*Omega_t;
        double th = omega_t.norm();
        Eigen::Matrix3d Q;
        Eigen::Matrix3d Qinv;
        
        if (th<0.00001)
          Q = (Eigen::Matrix3d::Identity() + (1.0/2.0)*Omega_t + (1.0/6.0)*Omega_t2);
        else
          Q = (Eigen::Matrix3d::Identity()
              + (1-cos(theta))/(theta*theta)*Omega_t
              + (1-(sin(theta)/theta))/(theta*theta)*Omega_t2);
              
        Qinv = Q.inverse();
        Eigen::Vector3d v_t = Qinv*(trans.cast<double>());
        
        Vector6ft twist;
        twist.block(0,0,3,1) = v_t.cast<float_trafos>();
        twist.block(3,0,3,1) = omega_t.cast<float_trafos>();
        
        return twist;
      }
      
      ///////////////////////////////
      Matrix4ft expMap(const Vector3ft& omega_tf, const Vector3ft& v_tf)
      {
        //const Eigen::Vector3d v_t = Eigen::Map<Eigen::Vector3d>(v_tf).cast<double>();
        Eigen::Vector3d omega_t = omega_tf.cast<double>();
        Eigen::Vector3d v_t = v_tf.cast<double>();
        double theta = omega_t.norm();

        Eigen::Matrix3d Omega_t = skew(omega_t);

        Eigen::Matrix3d R;
        Eigen::Matrix3d Q;
        Matrix4ft Trafo = Matrix4ft::Identity();
        Eigen::Matrix3d Omega_t2 = Omega_t*Omega_t;

        if (theta<0.00001)
        {        
          R = (Eigen::Matrix3d::Identity() + Omega_t + (1.0/2.0)*Omega_t2);
          Q = (Eigen::Matrix3d::Identity() + (1.0/2.0)*Omega_t + (1.0/6.0)*Omega_t2);
        }
        else
        {
          R = (Eigen::Matrix3d::Identity()
          + sin(theta)/theta *Omega_t
          + (1-cos(theta))/(theta*theta)*Omega_t2);

          Q = (Eigen::Matrix3d::Identity()
          + (1-cos(theta))/(theta*theta)*Omega_t
          + (1-(sin(theta)/theta))/(theta*theta)*Omega_t2);
        }

        
        
        Matrix3ft Rf = forceOrthogonalisation(R.cast<float_trafos>());
        
        //Eigen::Matrix3ft Rf = (R.cast<float>());
        
        Trafo.block<3,3>(0,0) = Rf;
        Trafo.block<3,1>(0,3) = Q.cast<float_trafos>()*v_t.cast<float_trafos>();

        return Trafo;
      }   
    
    
    
      Matrix3ft expMapRot(const Vector3ft& omega_tf)
      {

        //const Eigen::Vector3d v_t = Eigen::Map<Eigen::Vector3d>(v_tf).cast<double>();
        Eigen::Vector3d omega_t = omega_tf.cast<double>();
        double theta = omega_t.norm();

        Eigen::Matrix3d Omega_t = skew(omega_t);

        Eigen::Matrix3d R;
        Eigen::Matrix3d Omega_t2 = Omega_t*Omega_t;

        if (theta<0.00001)
          R = ( Eigen::Matrix3d::Identity() + Omega_t + (1.0/2.0)*Omega_t2  ); 
        else
          R = ( Eigen::Matrix3d::Identity()
                + sin(theta)/theta *Omega_t
                + (1-cos(theta))/(theta*theta)*Omega_t2 );

        
        
        Matrix3ft Rf = forceOrthogonalisation(R.cast<float_trafos>());
        
        
        //Eigen::Matrix3ft Rf = (R.cast<float>());
        
        return Rf;
      }

      Eigen::Matrix3d skew(const Eigen::Vector3d& w)
      {
        Eigen::Matrix3d res;
        res <<  0.0, -w[2], w[1],
                w[2], 0.0, -w[0],
                -w[1], w[0], 0.0;       

        return res;
      }

   
    }
  }
}

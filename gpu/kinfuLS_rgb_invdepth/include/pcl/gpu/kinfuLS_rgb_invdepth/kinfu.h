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

#ifndef PCL_KINFU_KINFUTRACKER_HPP_RGBD_
#define PCL_KINFU_KINFUTRACKER_HPP_RGBD_

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <vector>
//#include <boost/graph/buffer_concepts.hpp>

#include "internal.h"

#include <pcl/gpu/kinfuLS_rgb_invdepth/float3_operations.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfuLS_rgb_invdepth/pixel_rgb.h>
#include <pcl/gpu/kinfuLS_rgb_invdepth/tsdf_volume.h>
#include <pcl/gpu/kinfuLS_rgb_invdepth/color_volume.h>
#include <pcl/gpu/kinfuLS_rgb_invdepth/raycaster.h>

#include <pcl/gpu/kinfuLS_rgb_invdepth/cyclical_buffer.h>
//#include <pcl/gpu/kinfu_large_scale/standalone_marching_cubes.h>




namespace pcl
{
  namespace gpu
  {
    namespace kinfuRGBD
    {        
      /** \brief KinfuTracker class encapsulates implementation of Microsoft Kinect Fusion algorithm
        * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
        */
      class PCL_EXPORTS KinfuTracker
      {
        public:

          /** \brief Pixel type for rendered image. */
          typedef pcl::gpu::kinfuRGBD::PixelRGB PixelRGB;

          typedef DeviceArray2D<PixelRGB> View;
          typedef DeviceArray2D<unsigned short> DepthMap;
          typedef DeviceArray2D<unsigned char> IntensityMap;
          typedef DeviceArray2D<float> DepthMapf;
          typedef DeviceArray2D<float> IntensityMapf;
          typedef DeviceArray2D<float> GradientMap;
          
          

          /** \brief Number of pyramid levels */
          enum { LEVELS = 5 };          
          
          
          typedef pcl::PointXYZ PointType;
          typedef pcl::Normal NormalType;

          void 
          performLastScan (){perform_last_scan_ = true; PCL_WARN ("Kinfu will exit after next shift\n");}

          bool
          isFinished (){return (finished_);}

          /** \brief Constructor
            * \param[in] volumeSize physical size of the volume represented by the tdsf volume. In meters.
            * \param[in] shiftingDistance when the camera target point is farther than shiftingDistance from the center of the volume, shiting occurs. In meters.
            * \note The target point is located at (0, 0, 0.6*volumeSize) in camera coordinates.
            * \param[in] rows height of depth image
            * \param[in] cols width of depth image
            */
          KinfuTracker (const Eigen::Vector3f &volumeSize, const float shiftingDistance, int optim_dim = pcl::device::kinfuRGBD::NO_ILUM , 
                        int Mestimator = pcl::device::kinfuRGBD::DEFAULT_ESTIMATOR, 
                        int motion_model = pcl::device::kinfuRGBD::DEFAULT_MOTION_MODEL,
                        int depth_error_type = pcl::device::kinfuRGBD::DEFAULT_DEPTH_TYPE, int sigma_estimator = pcl::device::kinfuRGBD::DEFAULT_SIGMA,
                        int weighting = pcl::device::kinfuRGBD::DEFAULT_WEIGHTING, int warping = pcl::device::kinfuRGBD::DEFAULT_WARPING, 
                        int keyframe_count = pcl::device::kinfuRGBD::DEFAULT_KF_COUNT,
                        int finest_level = pcl::device::kinfuRGBD::DEFAULT_FINEST_LEVEL,
                        int termination = pcl::device::kinfuRGBD::DEFAULT_TERMINATION,
                        float visratio = pcl::device::kinfuRGBD::DEFAULT_VISRATIO,
                        int image_filtering = pcl::device::kinfuRGBD::DEFAULT_IMAGE_FILTERING,
                        int rows = 480, int cols = 640);

          /** \brief Sets Depth camera intrinsics
            * \param[in] fx focal length x 
            * \param[in] fy focal length y
            * \param[in] cx principal point x
            * \param[in] cy principal point y
            */
          void
          setDepthIntrinsics (float fx, float fy, float cx = -1, float cy = -1, float baseline = -1);

          /** \brief Sets initial camera pose relative to volume coordiante space
            * \param[in] pose Initial camera pose
            */
          void
          setInitialCameraPose (const Eigen::Affine3f& pose);           
          
          /** \brief Sets integration threshold. TSDF volume is integrated iff a camera movement metric exceedes the threshold value. 
            * The metric represents the following: M = (rodrigues(Rotation).norm() + alpha*translation.norm())/2, where alpha = 1.f (hardcoded constant)
            * \param[in] threshold a value to compare with the metric. Suitable values are ~0.001          
            */
          void
          setCameraMovementThreshold(float threshold = 0.001f);

          /** \brief Performs initialization for color integration. Must be called before calling color integration. 
            * \param[in] max_weight max weighe for color integration. -1 means default weight.
            */
          void
          initColorIntegration(int max_weight = -1);        

          /** \brief Returns cols passed to ctor */
          int
          cols ();

          /** \brief Returns rows passed to ctor */
          int
          rows ();

          /** \brief Processes next frame.
            * \param[in] Depth next frame with values in millimeters
            * \return true if can render 3D view.
            */
          bool operator() (const DepthMap& depth);

          /** \brief Processes next frame (both depth and color integration). Please call initColorIntegration before invpoking this.
            * \param[in] depth next depth frame with values in millimeters
            * \param[in] colors next RGB frame
            * \return true if can render 3D view.
            */
          bool operator() (const DepthMap& depth, const View& colors, float delta_t = 0.033333333f, unsigned long timestamp_rgb_curr = 0);
          
          int
          getGlobalTime () const;

          /** \brief Returns camera pose at given time, default the last pose
            * \param[in] time Index of frame for which camera pose is returned.
            * \return camera pose
            */
          Eigen::Affine3f
          getCameraPose (int time = -1) const;
          
          float
          getChiTest (int time = -1) const;
          
          float
          getVisOdoTime (int time = -1) const;
          
          unsigned long getTimestamp (int time = -1) const;
          
          Eigen::Affine3f
          getLastEstimatedPose () const;
          
          void
          downloadAndSaveIntensity(const DeviceArray2D<float>& dev_image, int pyrlevel, int iterat);

          void
          downloadAndSaveImCoords(const DeviceArray2D<float>& dev_image, int pyrlevel, int iterat);

          template<class T> void
          downloadAndSaveDepth(const DeviceArray2D<T>& dev_image, int pyrlevel, int iterat);
          
          void
          downloadAndSaveRGB(const DeviceArray2D<PixelRGB>& dev_image, int pyrlevel=0, int iterat=0);

          void
          downloadAndSaveGradientsDepth(const DeviceArray2D<float>& dev_gradIx, const DeviceArray2D<float>& dev_gradIy, int pyrlevel, int iterat);

          void
          downloadAndSaveGradientsIntensity(const DeviceArray2D<float>& dev_gradIx, const DeviceArray2D<float>& dev_gradIy, int pyrlevel, int iterat);

          /** \brief Returns number of poses including initial */
          size_t
          getNumberOfPoses () const;

          /** \brief Returns TSDF volume storage */
          const TsdfVolume& volume() const;

          /** \brief Returns TSDF volume storage */
          TsdfVolume& volume();

          /** \brief Returns color volume storage */
          const ColorVolume& colorVolume() const;

          /** \brief Returns color volume storage */
          ColorVolume& colorVolume();
          
          /** \brief Renders 3D scene to display to human
            * \param[out] view output array with image
            */
          void
          getImage (View& view);// const;
          
          /** \brief Returns point cloud abserved from last camera pose
            * \param[out] cloud output array for points
            */
          void
          getLastFrameCloud (DeviceArray2D<PointType>& cloud) const;

          /** \brief Returns point cloud abserved from last camera pose
            * \param[out] normals output array for normals
            */
          void
          getLastFrameNormals (DeviceArray2D<NormalType>& normals) const;
          
          
          /** \brief Returns pointer to the cyclical buffer structure
            */
          tsdf_buffer* 
          getCyclicalBufferStructure () 
          {
            return (cyclical_.getBuffer ());
          }
          
          /** \brief Extract the world and save it.
            */
          void
          extractAndSaveWorld ();
          
          /** \brief Returns true if ICP is currently lost */
          bool
          visOdoIsLost ()
          {
            return (lost_);
          }
          
          /** \brief Performs the tracker reset to initial  state. It's used if camera tracking fails. */
          void
          reset ();
          
          std::vector<double> condition_numbers_;
          std::vector<double> visibility_ratio_;
          std::vector<double> error_sigmas_int_;
          std::vector<double> error_sigmas_depth_;
          std::vector<double> error_biases_int_;
          std::vector<double> error_biases_depth_;
          std::vector<double> error_RMSE_;
          std::vector<double> trans_maxerr_;
          std::vector<double> rot_maxerr_;  
          //std::vector<Matrix6ft> odo_covmats_;
                
          std::vector<unsigned int> odo_odoKF_indexes_;      
          std::vector<unsigned int> odo_curr_indexes_;

        private:
          
          /** \brief Allocates all GPU internal buffers.
            * \param[in] rows_arg
            * \param[in] cols_arg          
            */
          void
          allocateBuffers (int rows_arg, int cols_arg);
                   
          typedef double float_trafos;          

          typedef Eigen::Matrix<float_trafos, 3, 3, Eigen::RowMajor> Matrix3ft;
          typedef Eigen::Matrix<float_trafos,3,1> Vector3ft;
          typedef Eigen::Matrix<float_trafos,6,1> Vector6ft;
          typedef Eigen::Matrix<float_trafos, 4, 4, Eigen::RowMajor> Matrix4ft;
          typedef Eigen::Matrix<float_trafos, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXft;
          
          typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;

          /** \brief Vertex or Normal Map type */
          typedef DeviceArray2D<float> MapArr;

          
          
          /** \brief helper function that converts transforms from host to device types
            * \param[in] transformIn1 first transform to convert
            * \param[in] transformIn2 second transform to convert
            * \param[in] translationIn1 first translation to convert
            * \param[in] translationIn2 second translation to convert
            * \param[out] transformOut1 result of first transform conversion
            * \param[out] transformOut2 result of second transform conversion
            * \param[out] translationOut1 result of first translation conversion
            * \param[out] translationOut2 result of second translation conversion
            */
          inline void 
          convertTransforms (Matrix3ft& transform_in_1, Matrix3ft& transform_in_2, Vector3ft& translation_in_1, Vector3ft& translation_in_2,
                                         pcl::device::kinfuRGBD::Mat33& transform_out_1, pcl::device::kinfuRGBD::Mat33& transform_out_2, float3& translation_out_1, float3& translation_out_2);
          
          /** \brief helper function that converts transforms from host to device types
            * \param[in] transformIn1 first transform to convert
            * \param[in] transformIn2 second transform to convert
            * \param[in] translationIn translation to convert
            * \param[out] transformOut1 result of first transform conversion
            * \param[out] transformOut2 result of second transform conversion
            * \param[out] translationOut result of translation conversion
            */
          inline void 
          convertTransforms (Matrix3ft& transform_in_1, Matrix3ft& transform_in_2, Vector3ft& translation_in,
                                         pcl::device::kinfuRGBD::Mat33& transform_out_1, pcl::device::kinfuRGBD::Mat33& transform_out_2, float3& translation_out);
          
          /** \brief helper function that converts transforms from host to device types
            * \param[in] transformIn transform to convert
            * \param[in] translationIn translation to convert
            * \param[out] transformOut result of transform conversion
            * \param[out] translationOut result of translation conversion
            */
          inline void 
          convertTransforms (Matrix3ft& transform_in, Vector3ft& translation_in,
                                          pcl::device::kinfuRGBD::Mat33& transform_out, float3& translation_out);

          /** \brief helper function that pre-process a raw detph map the kinect fusion algorithm.
            * The raw depth map is first blured, eventually truncated, and downsampled for each pyramid level.
            * Then, vertex and normal maps are computed for each pyramid level.
            * \param[in] depth_raw the raw depth map to process
            * \param[in] cam_intrinsics intrinsics of the camera used to acquire the depth map
            */
          inline void 
          prepareMaps (const DepthMap& depth_raw, const pcl::device::kinfuRGBD::Intr& cam_intrinsics);

          /** \brief helper function that pre-process a raw detph and RGB map the kinect fusion algorithm.
            * RGB to grayscale, 
            * Then, depth and grayscale maps are downsampled for each pyramid level.
            * \param[in] depth_raw the raw depth map to process
            * \param[in] color_raw the RGB image to process
            * \param[in] cam_intrinsics intrinsics of the camera used to acquire the depth map
            */
          inline void 
          prepareImages (const DepthMap& depth_raw, const View& colors_raw, const pcl::device::kinfuRGBD::Intr& cam_intrinsics);

          
          inline bool 
          estimateVisualOdometry(const pcl::device::kinfuRGBD::Intr cam_intrinsics, Matrix3ft& resulting_rotation , Vector3ft& resulting_translation);

          /** \brief Helper function that copies v_maps_curr and n_maps_curr to v_maps_prev_ and n_maps_prev_ */
          inline void 
          saveCurrentMaps();

          /** \brief Helper function that copies depths_curr and intensities_curr to depths_prev_ and intensities_prev_ */
          inline void 
          saveCurrentImages();	
          
          /** \brief Helper function that copies depths_curr and intensities_curr to depths_prev_ and intensities_prev_ */
          inline void 
          saveCurrentImagesAsKeyframes();	
          
          /** \brief Helper function that copies depths_curr and intensities_curr to depths_prev_ and intensities_prev_ */
          inline void 
          savePreviousImagesAsKeyframes();					

          /** \brief Cyclical buffer object */
          CyclicalBuffer cyclical_;

          /** \brief Height of input depth image. */
          int rows_;

          /** \brief Width of input depth image. */
          int cols_;

          /** \brief Frame counter */
          int global_time_;
          
          /** \brief Intrinsic parameters of depth camera. */
          float fx_, fy_, cx_, cy_, baseline_;

          /** \brief Tsdf volume container. */
          TsdfVolume::Ptr tsdf_volume_;

          /** \brief Color volume container. */
          ColorVolume::Ptr color_volume_;

          /** \brief Initial camera rotation in volume coo space. */
          Matrix3ft init_Rcam_;

          /** \brief Initial camera position in volume coo space. */
          Vector3ft   init_tcam_;

          /** \brief array with IPC iteration numbers for each pyramid level */
          int visodo_iterations_[LEVELS];

          /** \brief Depth pyramid. */
          std::vector<DepthMapf> depths_curr_;     
          
          DepthMapf depths_prefilt_;    

          /** \brief intensity pyramid. */
          std::vector<IntensityMapf> intensities_curr_;

          /** \brief Depth pyramid. */
          std::vector<DepthMapf> depths_prev_;

          /** \brief Depth pyramid. */
          std::vector<IntensityMapf> intensities_prev_;
          
          /** \brief Depth pyramid. */
          std::vector<DepthMapf> depths_keyframe_;
          std::vector<DepthMapf> depths_keyframe_filtered_;

          /** \brief Depth pyramid. */
          std::vector<IntensityMapf> intensities_keyframe_;
          std::vector<DepthMapf> intensities_keyframe_filtered_;
          
          /** \brief gradients pyramid. */
          std::vector<GradientMap> xGradsInt_keyframe_; 
          std::vector<GradientMap> yGradsInt_keyframe_;
          std::vector<GradientMap> xGradsDepth_keyframe_;
          std::vector<GradientMap> yGradsDepth_keyframe_;

          /** \brief warped maps. */
          std::vector<DepthMapf> warped_depths_curr_;
          std::vector<IntensityMapf> warped_intensities_curr_;
          std::vector<DepthMapf> warped_level_depths_curr_;
          std::vector<IntensityMapf> warped_level_intensities_curr_;

          /** \brief error vectors to sort. */
          std::vector< DeviceArray<float> > res_intensities_;
          std::vector< DeviceArray<float> > res_depths_;

          /**Auxiliary maps for forward warping**/
          DeviceArray2D<float> projected_transformed_points_;
          DeviceArray2D<float> depth_warped_in_curr_; 

          /** \brief Vertex maps pyramid for previous frame in global coordinate space. */
          std::vector<MapArr> vmaps_g_prev_;

          /** \brief Normal maps pyramid for previous frame in global coordinate space. */
          std::vector<MapArr> nmaps_g_prev_;

          /** \brief Buffer for storing scaled depth image */
          DeviceArray2D<float> depthRawScaled_;

          /** \brief Temporary buffer for ICP */
          DeviceArray2D<double> gbuf_;

          /** \brief Buffer to store MLS matrix. */
          DeviceArray<double> sumbuf_;

          /** \brief Array of camera rotation matrices for each moment of time. */
          std::vector<Matrix3ft> rmats_;

          /** \brief Array of camera translations for each moment of time. */
          std::vector<Vector3ft> tvecs_;
          
          /** \brief Array of chi tests for each moment of time. */
          std::vector<float> chi_tests_;
          
          /** \brief Array of visual odometry time costs for each moment of time. */
          std::vector<float> vis_odo_times_;

          /** \brief Array of timestamps for each moment of time. */
          std::vector<unsigned long> timestamps_;
          
          
          std::vector<Matrix3ft> odo_rmats_;
          std::vector<Vector3ft> odo_tvecs_;
        
          /** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
          float integration_metric_threshold_;          

          /** \brief When set to true, KinFu will extract the whole world and mesh it. */
          bool perform_last_scan_;

          /** \brief When set to true, KinFu notifies that it is finished scanning and can be stopped. */
          bool finished_;

          /** \brief // when the camera target point is farther than DISTANCE_THRESHOLD from the current cube's center, shifting occurs. In meters . */
          float shifting_distance_;

          /** \brief Size of the TSDF volume in meters. */
          float volume_size_;

          /** \brief True if ICP is lost */
          bool lost_;

          /** \brief Last estimated rotation (estimation is done via pairwise alignment when ICP is failing) */
          Matrix3ft last_estimated_rotation_;

          /** \brief Last estimated translation (estimation is done via pairwise alignment when ICP is failing) */
          Vector3ft last_estimated_translation_;

          int keyframe_count_;
          int curr_odoKF_index_;
          
          float delta_t_;
          Vector3ft velocity_;
          Vector3ft omega_;     
		
          int optim_dim_;  //not used
          int Mestimator_;
          int motion_model_;
          int depth_error_type_;
          int sigma_estimator_;
          int weighting_;
          int warping_;
          int max_keyframe_count_;
          int finest_level_;
          int termination_;
          float visibility_ratio_threshold_;
          int image_filtering_;
          
          

          // Update Pose
          Matrix3ft delta_rotation;
          Vector3ft delta_translation;
          Matrix3ft last_known_global_rotation;
          Vector3ft   last_known_global_translation;
          
          public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      };
    }
  }
};

#endif /* PCL_KINFU_KINFUTRACKER_HPP_ */

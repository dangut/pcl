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

#ifndef PCL_KINFU_INTERNAL_HPP_RGBD_
#define PCL_KINFU_INTERNAL_HPP_RGBD_

#include <pcl/gpu/containers/device_array.h>

// should we remove direct include and safe_call.hpp file. Include from gpu/utils ?
// #include <pcl/gpu/utils/safe_call.hpp> 
#include "safe_call.hpp"

#include <iostream> // used by operator << in Struct Intr

#include <pcl/gpu/kinfuLS_rgb_depth/tsdf_buffer.h>

//using namespace pcl::gpu;

namespace pcl
{
  namespace device
  {
    namespace kinfuRGBD
    {
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Types
      typedef unsigned short ushort;
      typedef unsigned char uchar;
      typedef DeviceArray2D<float> MapArr;
      typedef DeviceArray2D<ushort> DepthMap;
      typedef DeviceArray2D<uchar> IntensityMap;
      typedef DeviceArray2D<float> DepthMapf;
      typedef DeviceArray2D<float> IntensityMapf;
      typedef DeviceArray2D<float> GradientMap;
      typedef float4 PointType;
      typedef double float_type;

      //TSDF fixed point divisor (if old format is enabled)
      const int DIVISOR = 32767;     // SHRT_MAX;

      //RGB images resolution
      const float  HEIGHT = 480.0f;
      const float  WIDTH = 640.0f;

      //Should be multiple of 32
      enum { VOLUME_X = 512, VOLUME_Y = 512, VOLUME_Z = 512 };

      enum 
      {
        B_SIZE = 6,
        A_SIZE = (B_SIZE * B_SIZE - B_SIZE) / 2 + B_SIZE,
        TOTAL_SIZE = A_SIZE + B_SIZE
      };

      enum {LSQ, HUBER, TUKEY, STUDENT};
      enum {PYR_RAYCAST = 0};
      enum {NO_ILUM = 6, ILUM = 8};
      enum {NO_MM, CONSTANT_VELOCITY};
      enum {DEPTH, INV_DEPTH};
      enum {SIGMA_MAD, SIGMA_PDF, SIGMA_PDF_BIAS, SIGMA_PDF_SAMPLING, SIGMA_CONS};
      enum {INDEPENDENT, MIN_WEIGHT, DEPTH_ONLY, INT_ONLY};
      enum {EVERY_ITER, EVERY_PYR};
      enum {FORWARD,REVERSE};
      enum {WARP_FIRST, PYR_FIRST};
      enum {CHI_SQUARED, ALL_ITERS};
      enum {NO_FILTERS, FILTER_GRADS};

      const float THRESHOLD_HUBER = 1.345f;
      const float THRESHOLD_TUKEY = 4.685f;
      const float STUDENT_DOF = 5.f;

      //Temporary constant (until we make it automatic) that holds the Kinect's focal length
      //const float FOCAL_LENGTH = 575.816f;

      //xtion Bristol
      //      const float FOCAL_LENGTH = 540.60f;
      //      const float CENTER_X = 317.76f;
      //      const float CENTER_Y = 240.76f;

      //xtion Daniel's
      const float FOCAL_LENGTH = 543.78f;
      const float CENTER_X = 313.45f;
      const float CENTER_Y = 235.00f;

      const float BASELINE = 0.075f; //in m. Not an exact calibration. Just coarse approx by a rule.

      //COnstants for checking vOdo pyramid iterations termination
      const float MIN_DEPTH = 0.5f;  //real one might be a little more, but better to overestimate
      const float PIXEL_ACC = 0.5f; //pixel error, should put less, to be a little bit conservative??

      const float VOLUME_SIZE = 3.0f; // physical size represented by the TSDF volume. In meters
      const float DISTANCE_THRESHOLD = 1.5f; // when the camera target point is farther than DISTANCE_THRESHOLD from the current cube's center, shifting occurs. In meters
      const int SNAPSHOT_RATE = 45; // every 45 frames an RGB snapshot will be saved. -et parameter is needed when calling Kinfu Large Scale in command line.

      //const bool INTENSITY_OPTIM = 1; //estimate change of intensity modeled as variation in contrast (alpha) and brightness (beta),i.e, I(t+dt,p+Dp) = alpha*I(t,p) + beta
      const int DEFAULT_ESTIMATOR = STUDENT;  //LSQ, HUBER, TUKEY, STUDENT
      const int DEFAULT_MOTION_MODEL = CONSTANT_VELOCITY; //NO_MM, CONSTANT_VELOCITY
      const int DEFAULT_DEPTH_TYPE = INV_DEPTH;
      const int DEFAULT_SIGMA = SIGMA_PDF_SAMPLING;
      const int DEFAULT_WEIGHTING = INDEPENDENT;
      const int DEFAULT_WARPING = WARP_FIRST;
      const int DEFAULT_KF_COUNT = 9999999;
      const float DEFAULT_VISRATIO = 0.8;
      const int DEFAULT_FINEST_LEVEL = 0; //0->640x480, 1->320x240, 2-> 160x120, ...
      const int DEFAULT_TERMINATION = ALL_ITERS;
      const int DEFAULT_IMAGE_FILTERING = NO_FILTERS;

      /** \brief Camera intrinsics structure
      */ 
      struct Intr
      {
        float fx, fy, cx, cy, b;
        Intr () {}
        Intr (float fx_, float fy_, float cx_, float cy_, float b_ = 40) : fx (fx_), fy (fy_), cx (cx_), cy (cy_), b(b_) {}

        Intr operator () (int level_index) const
        { 
        int div = 1 << level_index; 
        return (Intr (fx / div, fy / div, cx / div, cy / div, b));
        }

        friend inline std::ostream&
        operator << (std::ostream& os, const Intr& intr)
        {
        os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
        return (os);
      }

      };

      /** \brief 3x3 Matrix for device code
      */ 
      struct Mat33
      {
      float3 data[3];
      };

      /** \brief Light source collection
      */ 
      struct LightSource
      {
      float3 pos[1];
      int number;
      };

      //////////////////
      //Debug utils
      void showGPUMemoryUsage();

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Maps    

      /** \brief Computes vertex map
        * \param[in] intr depth camera intrinsics
        * \param[in] depth depth
        * \param[out] vmap vertex map
        */
      void 
      createVMap (const Intr& intr, const DepthMap& depth, MapArr& vmap);
      
      void
      createVMapFromInvDepthf (const Intr& intr, const DepthMapf& inv_depth, MapArr& vmap);

      /** \brief Computes normal map using cross product
        * \param[in] vmap vertex map
        * \param[out] nmap normal map
        */
      void 
      createNMap (const MapArr& vmap, MapArr& nmap);

      /** \brief Computes normal map using Eigen/PCA approach
        * \param[in] vmap vertex map
        * \param[out] nmap normal map
        */
      void 
      computeNormalsEigen (const MapArr& vmap, MapArr& nmap);

      /** \brief Performs affine tranform of vertex and normal maps
        * \param[in] vmap_src source vertex map
        * \param[in] nmap_src source vertex map
        * \param[in] Rmat Rotation mat
        * \param[in] tvec translation
        * \param[out] vmap_dst destination vertex map
        * \param[out] nmap_dst destination vertex map
        */
      void 
      transformMaps (const MapArr& vmap_src, const MapArr& nmap_src, const Mat33& Rmat, const float3& tvec, MapArr& vmap_dst, MapArr& nmap_dst);

      /** \brief Performs depth truncation
        * \param[out] depth depth map to truncation
        * \param[in] max_distance truncation threshold, values that are higher than the threshold are reset to zero (means not measurement)
        */
      void 
      truncateDepth(DepthMap& depth, float max_distance);      

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //   Images (Daniel)
      float
      bilateralFilter (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst, const float sigma_floatmap);

      /** \brief Computes depth pyramid
        * \param[in] src source
        * \param[out] dst destination
        */      
      float
      pyrDownDepth (const DepthMapf& src, DepthMapf& dst);

      /** \brief Computes intensity pyramid
        * \param[in] src source
        * \param[out] dst destination
        */      
      float
      pyrDownIntensity (const IntensityMapf& src, IntensityMapf& dst);

      void
      convertDepth2Float (const DepthMap& src, DepthMapf& dst);

      void
      convertDepth2InvDepth (const DepthMap& src, DepthMapf& dst);

      /** \brief Computes 8 bits intensity map from 24 bits rgb map
        * \param[in] RGB map source
        * \param[out] intensity map destination
        */              
      void
      computeIntensity (const PtrStepSz<uchar3>& src, IntensityMapf& dst);      
        
      void
      convertFloat2RGB (const IntensityMapf& src, PtrStepSz<uchar3> dst);      

      /** \brief Computes intensity gradient
        * \param[in] intensity map 
        * \param[out] horizontal intensity gradient
        * \param[out] vertical intensity gradient
        */        
      float
      computeGradientIntensity (const IntensityMapf& src, GradientMap& dst_hor, GradientMap& dst_vert);

      /** \brief Computes depth gradient
        * \param[in] depth map 
        * \param[out] horizontal depth gradient
        * \param[out] vertical depth gradient
        */        
      float
      computeGradientDepth (const DepthMapf& src, GradientMap& dst_hor, GradientMap& dst_vert);

      //TODO: depthMapInpainting(...)??

      /** \brief Copies intensity and depth maps
        * \param[in] src depth map 
        * \param[in] src intensity map 
        * \param[out] dst depth map 
        * \param[out] dst intensity map 
        */     
      void 
      copyImages (const DepthMapf& src_depth, const IntensityMapf& src_int,
                  DepthMapf& dst_depth,  IntensityMapf& dst_int);

      void 
      copyImage (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);

      void
      generateDepthf (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMapf& dst);

      ///Median functions//////////////////////////////////////////////
      float 
      computeError (const DeviceArray2D<float>& im1, const DeviceArray2D<float>& im0, DeviceArray<float>& error);  
            
      
      float
      computeChiSquare (DeviceArray<float>& error_int,  DeviceArray<float>& error_depth,  
                        float sigma_int, float sigma_depth, int Mestimator, 
                        float &chi_square, float &chi_test, float &Ndof);   
                        
      float 
      computeSigmaMAD (DeviceArray<float>& error,  float& sigma);

      float 
      computeSigmaPdf (DeviceArray<float>& error,  float& sigma, int Mestimator);    
      
      float  
      computeSigmaPdfWithBias (DeviceArray<float>& error, float &sigma, float &bias, int Mestimator); 
      
      float  
      computeSigmaPdfWithBiasDirectReduction (DeviceArray<float>& error, float &sigma, float &bias, int Mestimator);

      float 
      computeMedian (DeviceArray<float>& error);


      ///Visual Odometry functions ////////////////////////////////

      float buildSystem (const float3 delta_trans, const float3 delta_rot,
                         const DepthMapf& D0, const IntensityMapf& I0,
                         const GradientMap& gradD0_x, const GradientMap& gradD0_y, 
                         const GradientMap& gradI0_x, const GradientMap& gradI0_y,
                         const DepthMapf& D1, const IntensityMapf& I1,
                         int Mestimator, int depth_error, int weighting,
                         float sigma_depth, float sigma_int,
                         float bias_depth, float bias_int,                         
                         const Intr& intr, const int size_A,
                         DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host);     
                         
      /** \brief Warps I1 towards I0
        * \param[in] curr intensity map
        * \param[in] warped intensity map
        * \param[in] prev inv depth map
        * \param[in] current R^T = 1_R^0
        * \param[in] current -R^T * t = t_1->0
        * \param[in] camera intrinsics
        */
      float warpIntensityWithTrafo3DInvDepth  (const IntensityMapf& src, IntensityMapf& dst, const DepthMapf& depth_prev, 
                                               Mat33 inv_rotation, float3 inv_translation, const Intr& intr);  

      /** \brief Warps I1 towards I0
        * \param[in] curr intensity map
        * \param[in] warped intensity map
        * \param[in] prev invDepth map
        * \param[in] current R^T = 1_R^0
        * \param[in] current -R^T * t = t_1->0
        * \param[in] camera intrinsics
        */
      float warpIntensityWithTrafo3D (const IntensityMapf& src, IntensityMapf& dst, const DepthMapf& depth_prev, 
                                      Mat33 inv_rotation, float3 inv_translation, const Intr& intr); 

      /** \brief Warps D1 towards D0
        * \param[in] curr depth map
        * \param[in] warped depth map
        * \param[in] prev depth map
        * \param[in] current R^T = 1_R^0
        * \param[in] current -R^T * t = t_1->0
        * \param[in] camera intrinsics
        * \param[in] current R = 0_R^1
        */       
      float warpDepthWithTrafo3D  (const DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, 
                                   Mat33 inv_rotation, float3 inv_translation, const Intr& intr, Mat33 rotation);    

      /** \brief Warps D1 towards D0
        * \param[in] curr inv depth map
        * \param[in] warped inv depth map
        * \param[in] prev  inv depth map
        * \param[in] current R^T = 1_R^0
        * \param[in] current -R^T * t = t_1->0
        * \param[in] camera intrinsics
        * \param[in] current R = 0_R^1
        */      
      float warpInvDepthWithTrafo3D  (const DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, 
                                      Mat33 inv_rotation, float3 inv_translation, const Intr& intr, Mat33 rotation);      
                                      
        
      /** \brief Lifts and projects point cloud from source to destiny
        * \param[in] source depth map
        * \param[in] points projected on dst after lifting from src and warping
        * \param[in] depth map of the dst, but matched to pixels in src
        * \param[in] dst depth map, passed just for initialise to NaN
        * \param[in] dst intensity map, passed just for initialise to NaN
        * \param[in] current R = 0_R^1
        * \param[in] current t = t_0^1
        * \param[in] camera intrinsics
        */  
      float 
      liftWarpAndProjDepth  (const DepthMapf& depth_src, DeviceArray2D<float>& proj_dst, DepthMapf& depth_dst_in_src,
                             DepthMapf& depth_dst, IntensityMapf& int_dst,
                             Mat33 rotation, float3 translation, const Intr& intr);
                             
      float 
      getVisibilityRatio (DepthMapf& depth_src, DepthMapf& depth_dst,
                          Mat33 rotation, float3 translation, 
                          const Intr& intr, float& visibility_ratio, float geom_tol, int geom_error_type)  ;
                             
      /** \brief Lifts and projects point cloud from source to destiny
        * \param[in] source inverse depth map
        * \param[in] points projected on dst after lifting from src and warping
        * \param[in] inverse depth map of the dst, but matched to pixels in src
        * \param[in] dst inverse depth map, passed just for initialise to NaN
        * \param[in] dst intensity map, passed just for initialise to NaN
        * \param[in] current R = 0_R^1
        * \param[in] current t = t_0^1
        * \param[in] camera intrinsics
        */                              
      float 
      liftWarpAndProjInvDepth (const DepthMapf& depth_src, DeviceArray2D<float>& proj_dst, DepthMapf& depth_dst_in_src,
                               DepthMapf& depth_dst, IntensityMapf& int_dst,
                               Mat33 rotation, float3 translation, const Intr& intr); 
                                  
       /** \brief Computes the dst intensity and depth maps
        * \param[in] depth map of the dst, but matched to pixels in src
        * \param[in] source intensity map
        * \param[in] points projected on dst after lifting from src and warping
        * \param[in] dst depth map
        * \param[in] dst intensity map
        */                             
      float 
      forwardMappingToWarpedDepth (const DepthMapf& depth_dst_in_src, const IntensityMapf& int_src, const DeviceArray2D<float>& proj_dst,
                                   DepthMapf& depth_dst, IntensityMapf& int_dst); 
                                   
      /** \brief Computes the dst intensity and inverse depth maps
        * \param[in] inverse depth map of the dst, but matched to pixels in src
        * \param[in] source intensity map
        * \param[in] points projected on dst after lifting from src and warping
        * \param[in] dst inverse depth map
        * \param[in] dst intensity map
        */   
      float 
      forwardMappingToWarpedInvDepth  (const DepthMapf& depth_dst_in_src, const IntensityMapf& int_src, const DeviceArray2D<float>& proj_dst,
                                       DepthMapf& depth_dst, IntensityMapf& int_dst); 
      

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // TSDF volume functions            

      /** \brief Perform tsdf volume initialization
        *  \param[out] array volume to be initialized
        */
      PCL_EXPORTS void
      initVolume(PtrStep<short2> array);

      //first version
      /** \brief Performs Tsfg volume uptation (extra obsolete now)
        * \param[in] depth_raw Kinect depth image
        * \param[in] intr camera intrinsics
        * \param[in] volume_size size of volume in mm
        * \param[in] Rcurr_inv inverse rotation for current camera pose
        * \param[in] tcurr translation for current camera pose
        * \param[in] tranc_dist tsdf truncation distance
        * \param[in] volume tsdf volume to be updated
        */
      void 
      integrateTsdfVolume (const PtrStepSz<ushort>& depth_raw, const Intr& intr, const float3& volume_size, 
                          const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short2> volume);

      //second version
      /** \brief Function that integrates volume if volume element contains: 2 bytes for round(tsdf*SHORT_MAX) and 2 bytes for integer weight.
        * \param[in] depth Kinect depth image
        * \param[in] intr camera intrinsics
        * \param[in] volume_size size of volume in mm
        * \param[in] Rcurr_inv inverse rotation for current camera pose
        * \param[in] tcurr translation for current camera pose
        * \param[in] tranc_dist tsdf truncation distance
        * \param[in] volume tsdf volume to be updated
        * \param[in] buffer cyclical buffer structure
        * \param[out] depthRawScaled Buffer for scaled depth along ray
        */
      PCL_EXPORTS void 
      integrateTsdfVolume (const PtrStepSz<ushort>& depth, const Intr& intr, const float3& volume_size, 
                          const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, PtrStep<short2> volume, const pcl::gpu::kinfuRGBD::tsdf_buffer* buffer, DeviceArray2D<float>& depthScaled);
      
      /** \brief Function that clears the TSDF values. The clearing takes place from the origin (in indices) to an offset in X,Y,Z values accordingly
        * \param[in] volume Pointer to TSDF volume in GPU
        * \param[in] buffer Pointer to the buffer struct that contains information about memory addresses of the tsdf volume memory block, which are used for the cyclic buffer.
        * \param[in] shiftX Offset in indices that will be cleared from the TSDF volume. The clearing start from buffer.OriginX and stops in OriginX + shiftX
        * \param[in] shiftY Offset in indices that will be cleared from the TSDF volume. The clearing start from buffer.OriginY and stops in OriginY + shiftY
        * \param[in] shiftZ Offset in indices that will be cleared from the TSDF volume. The clearing start from buffer.OriginZ and stops in OriginZ + shiftZ
        */
      PCL_EXPORTS void 
      clearTSDFSlice (PtrStep<short2> volume, pcl::gpu::kinfuRGBD::tsdf_buffer* buffer, int shiftX, int shiftY, int shiftZ);
      
      /** \brief Initialzied color volume
        * \param[out] color_volume color volume for initialization
        */
      void 
      initColorVolume(PtrStep<uchar4> color_volume);    

      /** \brief Performs integration in color volume
        * \param[in] intr Depth camera intrionsics structure
        * \param[in] tranc_dist tsdf truncation distance
        * \param[in] R_inv Inverse camera rotation
        * \param[in] t camera translation      
        * \param[in] vmap Raycasted vertex map
        * \param[in] colors RGB colors for current frame
        * \param[in] volume_size volume size in meters
        * \param[in] color_volume color volume to be integrated
        * \param[in] max_weight max weight for running color average. Zero means not average, one means average with prev value, etc.
        */    
      void 
      updateColorVolume(const Intr& intr, float tranc_dist, const Mat33& R_inv, const float3& t, const MapArr& vmap, 
              const PtrStepSz<uchar3>& colors, const float3& volume_size, PtrStep<uchar4> color_volume, int max_weight = 1);

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Raycast and view generation        
      /** \brief Generation vertex and normal maps from volume for current camera pose
        * \param[in] intr camera intrinsices
        * \param[in] Rcurr current rotation
        * \param[in] tcurr current translation
        * \param[in] tranc_dist volume truncation distance
        * \param[in] volume_size volume size in mm
        * \param[in] volume tsdf volume
        * \param[in] buffer cyclical buffer structure
        * \param[out] vmap output vertex map
        * \param[out] nmap output normals map
        */
      void 
      raycast (const Intr& intr, const Mat33& Rcurr, const float3& tcurr, float tranc_dist, const float3& volume_size, 
              const PtrStep<short2>& volume, const pcl::gpu::kinfuRGBD::tsdf_buffer* buffer, MapArr& vmap, MapArr& nmap);

      /** \brief Renders 3D image of the scene
        * \param[in] vmap vertex map
        * \param[in] nmap normals map
        * \param[in] light pose of light source
        * \param[out] dst buffer where image is generated
        */
      void 
      generateImage (const MapArr& vmap, const MapArr& nmap, const LightSource& light, PtrStepSz<uchar3> dst);


      /** \brief Renders depth image from give pose
        * \param[in] vmap inverse camera rotation
        * \param[in] nmap camera translation
        * \param[in] light vertex map
        * \param[out] dst buffer where depth is generated
        */
      void
      generateDepth (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMap& dst);

      /** \brief Paints 3D view with color map
        * \param[in] colors rgb color frame from OpenNI   
        * \param[out] dst output 3D view
        * \param[in] colors_wight weight for colors   
        */
      void 
      paint3DView(const PtrStep<uchar3>& colors, PtrStepSz<uchar3> dst, float colors_weight = 0.5f);

      /** \brief Performs resize of vertex map to next pyramid level by averaging each four points
        * \param[in] input vertext map
        * \param[out] output resized vertex map
        */
      void 
      resizeVMap (const MapArr& input, MapArr& output);
      
      /** \brief Performs resize of vertex map to next pyramid level by averaging each four normals
        * \param[in] input normal map
        * \param[out] output vertex map
        */
      void 
      resizeNMap (const MapArr& input, MapArr& output);

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Push data to TSDF
      
          /** \brief Loads the values of a tsdf point cloud to the tsdf volume in GPU
        * \param[in] volume tsdf volume 
        * \param[in] cloud_gpu contains the data to be pushed to the tsdf volume
        * \param[in] buffer Pointer to the buffer struct that contains information about memory addresses of the tsdf volume memory block, which are used for the cyclic buffer.
        */     
      /*PCL_EXPORTS*/ void 
      pushCloudAsSliceGPU (const PtrStep<short2>& volume, pcl::gpu::DeviceArray<PointType> cloud_gpu, const pcl::gpu::kinfuRGBD::tsdf_buffer* buffer);
      
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Cloud extraction 

      /** \brief Perform point cloud extraction from tsdf volume
        * \param[in] volume tsdf volume 
        * \param[in] volume_size size of the volume
        * \param[out] output buffer large enought to store point cloud
        * \return number of point stored to passed buffer
        */ 
      PCL_EXPORTS size_t 
      extractCloud (const PtrStep<short2>& volume, const float3& volume_size, PtrSz<PointType> output);

      /** \brief Perform point cloud extraction of a slice from tsdf volume
        * \param[in] volume tsdf volume on GPU
        * \param[in] volume_size size of the volume
        * \param[in] buffer Pointer to the buffer struct that contains information about memory addresses of the tsdf volume memory block, which are used for the cyclic buffer.
        * \param[in] shiftX Offset in indices that will be cleared from the TSDF volume. The clearing start from buffer.OriginX and stops in OriginX + shiftX
        * \param[in] shiftY Offset in indices that will be cleared from the TSDF volume. The clearing start from buffer.OriginY and stops in OriginY + shiftY
        * \param[in] shiftZ Offset in indices that will be cleared from the TSDF volume. The clearing start from buffer.OriginZ and stops in OriginZ + shiftZ
        * \param[out] output_xyz buffer large enought to store point cloud xyz values
        * \param[out] output_intensities buffer large enought to store point cloud intensity values
        * \return number of point stored to passed buffer
        */ 
      PCL_EXPORTS size_t
      extractSliceAsCloud (const PtrStep<short2>& volume, const float3& volume_size, const pcl::gpu::kinfuRGBD::tsdf_buffer* buffer, const int shiftX, const int shiftY, const int shiftZ, PtrSz<PointType> output_xyz, PtrSz<float> output_intensities);

      /** \brief Performs normals computation for given poins using tsdf volume
        * \param[in] volume tsdf volume
        * \param[in] volume_size volume size
        * \param[in] input points where normals are computed
        * \param[out] output normals. Could be float4 or float8. If for a point normal can't be computed, such normal is marked as nan.
        */ 
      template<typename NormalType> 
      void 
      extractNormals (const PtrStep<short2>& volume, const float3& volume_size, const PtrSz<PointType>& input, NormalType* output);

      /** \brief Performs colors exctraction from color volume
        * \param[in] color_volume color volume
        * \param[in] volume_size volume size
        * \param[in] points points for which color are computed
        * \param[out] colors output array with colors.
        */
      void 
      exctractColors(const PtrStep<uchar4>& color_volume, const float3& volume_size, const PtrSz<PointType>& points, uchar4* colors);

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Utility
      struct float8  { float x, y, z, w, c1, c2, c3, c4; };
      struct float12 { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; };

      /** \brief Conversion from SOA to AOS
        * \param[in] vmap SOA map
        * \param[out] output Array of 3D points. Can be float4 or float8.
        */
      template<typename T> 
      void 
      convert (const MapArr& vmap, DeviceArray2D<T>& output);

      /** \brief Merges pcl::PointXYZ and pcl::Normal to PointNormal
        * \param[in] coud points cloud
        * \param[in] normals normals cloud
        * \param[out] output array of PointNomals.
        */
      void 
      mergePointNormal(const DeviceArray<float4>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output);

      /** \brief  Check for qnan (unused now) 
        * \param[in] value
        */
      inline bool 
      valid_host (float value)
      {
        return *reinterpret_cast<int*>(&value) != 0x7fffffff; //QNAN
      }

      /** \brief synchronizes CUDA execution */
      inline 
      void 
      sync () { cudaSafeCall (cudaDeviceSynchronize ()); }


      template<class D, class Matx> D&
      device_cast (Matx& matx)
      {
        return (*reinterpret_cast<D*>(matx.data ()));
      }
   
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Marching cubes implementation

      /** \brief Binds marching cubes tables to texture references */
      void 
      bindTextures(const int *edgeBuf, const int *triBuf, const int *numVertsBuf);            
      
      /** \brief Unbinds */
      void 
      unbindTextures();
      
      /** \brief Scans tsdf volume and retrieves occuped voxes
        * \param[in] volume tsdf volume
        * \param[out] occupied_voxels buffer for occuped voxels. The function fulfills first row with voxel ids and second row with number of vertextes.
        * \return number of voxels in the buffer
        */
      int
      getOccupiedVoxels(const PtrStep<short2>& volume, DeviceArray2D<int>& occupied_voxels);

      /** \brief Computes total number of vertexes for all voxels and offsets of vertexes in final triangle array
        * \param[out] occupied_voxels buffer with occuped voxels. The function fulfills 3nd only with offsets      
        * \return total number of vertexes
        */
      int
      computeOffsetsAndTotalVertexes(DeviceArray2D<int>& occupied_voxels);

      /** \brief Generates final triangle array
        * \param[in] volume tsdf volume
        * \param[in] occupied_voxels occuped voxel ids (first row), number of vertexes(second row), offsets(third row).
        * \param[in] volume_size volume size in meters
        * \param[out] output triangle array            
        */
      void
      generateTriangles(const PtrStep<short2>& volume, const DeviceArray2D<int>& occupied_voxels, const float3& volume_size, DeviceArray<PointType>& output);
    }
  }
}

#endif /* PCL_KINFU_INTERNAL_HPP_ */

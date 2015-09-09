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


      
//*.cu for straightforward misc computations (float depth, float inversedepth, truncateDepth, gradients, RGB2grayscale). 
      
      
#include "device.hpp"

namespace pcl
{
  namespace device
  {
    namespace kinfuRGBD
    {

      namespace grad
      {
        enum
        {
          BLOCK_X = 32,
          BLOCK_Y = 8,
          RADIUS = 1,

          DIAMETER = RADIUS * 2 + 1,
          TILE_X = ( 2 * BLOCK_X + 2 * RADIUS ),
          TILE_Y = ( 2 * BLOCK_Y + 2 * RADIUS ),
          BLOCK_SIZE = BLOCK_X * BLOCK_Y,
          SHARED_SIZE = TILE_X * TILE_Y,
          STRIDE_SHARED = (SHARED_SIZE - 1) / BLOCK_SIZE + 1, //the final (+1) is for ceiling
        };
      }     

      
      /////////////////////////////////////////////////////////////////////
      __global__ void
      depth2floatKernel (const PtrStepSz<ushort> src, PtrStep<float> dst) 
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
        return;

        dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();

        int value =  src.ptr (y)[x];

        if (value > 0)
          dst.ptr (y)[x] = (__int2float_rn( max( 0 , min( value , 10000 ) ))) / 1000.f; 
        
        return;
      }
      
      /////////////////////////////////////////////////////////////////////
      __global__ void
      depth2invDepthKernel  (const PtrStepSz<ushort> src, PtrStep<float> dst) 
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
          return;

        dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();

        int value =  src.ptr (y)[x];

        if ((value > 0) && (value > 300))
          dst.ptr (y)[x] = 1000.f / __int2float_rn( max( 0 , min( value , 10000 ) )); 
        
        return;
      }


      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      intensityKernel(const PtrStepSz<uchar3> src, 
      PtrStep<float> dst, int cols, int rows) 
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= cols || y >= rows)
          return;

        uchar3 value = src.ptr (y)[x];
        int r = value.x;
        int g = value.y;
        int b = value.z;

        dst.ptr (y)[x] =  max ( 0.f, min ( (  0.2126f * __int2float_rn (r) + 
                                              0.7152f * __int2float_rn (g) + 
                                              0.0722f * __int2float_rn (b)  ), 255.f ) );
        return;
      }


      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      gradientKernel  (const PtrStepSz<float> src, 
                       PtrStep<float> dst_hor,
                       PtrStep<float> dst_vert) 
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
          return;

        dst_hor.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
        dst_vert.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
        float value = src.ptr (y)[x];

        if (isnan(value))
          return;
          
        float res_hor = 0;
        float res_vert = 0;

        for (int dx=-1; dx<2; dx++)
        {
          for (int dy=-1; dy<2; dy++)
          {
            int cx = min (max (0, x + dx), src.cols - 1);
            int cy = min (max (0, y + dy), src.rows - 1);

            int weight_hor = dx * (2 - dy * dy); 
            int weight_vert = dy * (2 - dx * dx);
            float temp_h = src.ptr (cy)[cx];
            float temp_v = temp_h;
            
            if (isnan(temp_h))
              return;
            
            //This was some attempt to compute the gradient when some neighboring pixels are NaN
            //but I think its worse than just discarding pixels with NaN neighbors
            /*if (isnan (temp_h)){
              temp_h = src.ptr (cy)[x];
              temp_v = src.ptr (y)[cx];
            }
            if (isnan (temp_h))
              temp_h = value;

            if (isnan (temp_v))
              temp_v = value;
            */
            
            res_hor +=  temp_h*weight_hor;
            res_vert +=  temp_v*weight_vert;
          }
        }
      
        //8 is the sum of abs(weights) of sobel operator-> [grad] = [map_units/px]
        dst_hor.ptr (y)[x] =  (res_hor) / 8.f; 
        dst_vert.ptr (y)[x] = (res_vert) / 8.f;
        
        return;
      }
      
       
      /////////////////////////////////////////////////////////////////////
      __global__ void
      copyImagesKernel (const PtrStepSz<float> src_depth, const PtrStepSz<float> src_int,
                        PtrStep<float> dst_depth, PtrStep<float> dst_int)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src_depth.cols || y >= src_depth.rows)
          return;

        float value_depth = src_depth.ptr (y)[x];
        float value_int = src_int.ptr (y)[x];

        dst_depth.ptr (y)[x] = value_depth;
        dst_int.ptr (y)[x] = value_int;
      }

      ///////////////////////////////////////////////////////////////////////

      __global__ void
      copyImageKernel (const PtrStepSz<float> src, PtrStep<float> dst)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
          return;

        float value = src.ptr (y)[x];

        dst.ptr (y)[x] = value;
      }
      
      
      __global__ void
      float2ucharKernel(const PtrStep<float> src, PtrStepSz<uchar3> dst,int cols, int rows) 
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= cols || y >= rows)
          return;

        float value_src = src.ptr (y)[x];
        uchar3 value_dst;
        float min_val = 0.f;
        float max_val = 255.f;
        
        if (isnan(value_src))
        {
          value_dst.x = 200;
          value_dst.y = 150;
          value_dst.z = 150;
        }
        else if (isinf(value_src))
        {
          value_dst.x = 150;
          value_dst.y = 150;
          value_dst.z = 250;
        }
        else
        {
          unsigned char grey_val = max(0, min( __float2int_rn(255*(value_src - min_val)/(max_val-min_val)), 255));
          value_dst.x = grey_val;
          value_dst.y = grey_val;
          value_dst.z = grey_val;
        }

        dst.ptr (y)[x] =  value_dst;
        return;
      }
        
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void
      convertDepth2Float (const DepthMap& src, DepthMapf& dst)
      {
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

        depth2floatKernel<<<grid, block>>>(src, dst);
        cudaSafeCall ( cudaGetLastError () );
      };
      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void
      convertDepth2InvDepth (const DepthMap& src, DepthMapf& dst)
      {
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

        depth2invDepthKernel<<<grid, block>>>(src, dst);
        cudaSafeCall ( cudaGetLastError () );
      };

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void
      computeIntensity (const PtrStepSz<uchar3>& src, IntensityMapf& dst)
      {
        dim3 block (32, 8);
        dim3 grid (divUp (dst.cols(), block.x), divUp (dst.rows(), block.y));
        
        intensityKernel<<<grid, block>>>(src, dst, dst.cols(), dst.rows());
        cudaSafeCall ( cudaGetLastError () );
      };

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////   
      float
      computeGradientIntensity (const IntensityMapf& src, GradientMap& dst_hor, GradientMap& dst_vert)  
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////
        
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

        gradientKernel<<<grid, block>>>(src, dst_hor, dst_vert);
        cudaSafeCall ( cudaGetLastError () );
        
        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;
      };
      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
      float
      computeGradientDepth (const DepthMapf& src, GradientMap& dst_hor, GradientMap& dst_vert)  
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////
        
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

        gradientKernel<<<grid, block>>>(src, dst_hor, dst_vert);
        cudaSafeCall ( cudaGetLastError () );
        
        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;
      };
      
      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
      void 
      copyImages  (const DepthMapf& src_depth, const IntensityMapf& src_int,
                   DepthMapf& dst_depth,  IntensityMapf& dst_int)
      {
        dim3 block (32, 8);
        dim3 grid (divUp (src_depth.cols (), block.x), divUp (src_depth.rows (), block.y));

        copyImagesKernel<<<grid, block>>>(src_depth, src_int, dst_depth, dst_int);
        cudaSafeCall ( cudaGetLastError () );
      };

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
      void 
      copyImage (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst)
      {
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

        copyImageKernel<<<grid, block>>>(src, dst);
        cudaSafeCall ( cudaGetLastError () );
      };
      
      void
      convertFloat2RGB (const IntensityMapf& src, PtrStepSz<uchar3> dst)
      {
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols(), block.x), divUp (src.rows(), block.y));
        
        float2ucharKernel<<<grid, block>>>(src, dst, src.cols(), src.rows());
        cudaSafeCall ( cudaGetLastError () );
      };  
      
      void //for debugging
      showGPUMemoryUsage()
      {
        // show memory usage of GPU
        size_t free_byte ;
        size_t total_byte ;

        cudaSafeCall (cudaMemGetInfo( &free_byte, &total_byte )) ;

        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;

        std::cout << "GPU memory usage: used =  " << used_db/1024.0/1024.0 << " MB, free = " << free_db/1024.0/1024.0 << " MB, total = " << total_db/1024.0/1024.0 << std::endl;
      };
    }
  }
 }

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


#include "device.hpp"

#include <stdio.h>

using namespace pcl::device;

namespace pcl
{
  namespace device
  {
    namespace kinfuRGBD
    {
      //typedef double float_type;

      texture<float, 2, cudaReadModeElementType> texRefDepth;
      texture<float, 2, cudaReadModeElementType> texRefIntensity;

      __constant__ float MAX_DEPTH = 15.f;  
      
      template<int CTA_SIZE_, typename T>
      static __device__ __forceinline__ void reduce(volatile T* buffer)
      {
        int tid = Block::flattenedThreadId();
        T val =  buffer[tid];

        if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; __syncthreads(); }
        if (CTA_SIZE_ >=  512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; __syncthreads(); }
        if (CTA_SIZE_ >=  256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; __syncthreads(); }
        if (CTA_SIZE_ >=  128) { if (tid <  64) buffer[tid] = val = val + buffer[tid +  64]; __syncthreads(); }

        if (tid < 32)
        {
          if (CTA_SIZE_ >=   64) { buffer[tid] = val = val + buffer[tid +  32]; }
          if (CTA_SIZE_ >=   32) { buffer[tid] = val = val + buffer[tid +  16]; }
          if (CTA_SIZE_ >=   16) { buffer[tid] = val = val + buffer[tid +   8]; }
          if (CTA_SIZE_ >=    8) { buffer[tid] = val = val + buffer[tid +   4]; }
          if (CTA_SIZE_ >=    4) { buffer[tid] = val = val + buffer[tid +   2]; }
          if (CTA_SIZE_ >=    2) { buffer[tid] = val = val + buffer[tid +   1]; }
        }
      } 

      //Kernels for reverse warping
      ///////////////////////////////////////////////////////////////////////////////////////////////////////      
      ///////////////////////////////////////////////////////////////////////////////////////////////////////     
      __global__ void
      trafo3DKernelIntensity  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depth_prev, 
                               const Mat33 inv_rotation, const float3 inv_translation, const Intr intr, int colOff)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= dst.cols || y >= dst.rows)
          return;

        float3 X_dst, X_src;
        float x_src, y_src;
        float z = depth_prev.ptr (y)[x];

        float res = numeric_limits<float>::quiet_NaN (); 
        dst.ptr (y)[x] = res;

        if (isnan(z))
          return;	

        X_dst.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
        X_dst.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
        X_dst.z = 1.f * z;

        X_src = inv_rotation*X_dst + inv_translation;

        //Texture ref frame is in (-0.5, -0.5) wrt pixel (0,0). 
        //That means that pixel (0,0)->(0.5,0.5) in texture. So: p_tex = p + 0.5
        x_src = ( X_src.x / X_src.z ) * intr.fx + intr.cx + 0.5f ;
        y_src = ( X_src.y / X_src.z ) * intr.fy + intr.cy + 0.5f;


        if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
              || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows 
              || isnan(z))  )
        {
          res = tex2D(texRefIntensity, x_src , y_src);
          dst.ptr (y)[x] = max (0.f, min (res, 255.f));
        }

      }


      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      trafo3DKernelDepth  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depth_prev, 
                           const Mat33 inv_rotation, const float3 inv_translation, const Intr intr, const Mat33 rotation, int colOff)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= dst.cols || y >= dst.rows)
          return;

        float3 X_dst, X_src;
        float x_src, y_src;
        float z = depth_prev.ptr (y)[x] ;

        float res = numeric_limits<float>::quiet_NaN (); 
        dst.ptr (y)[x] = res;

        if (isnan(z))
          return;	

        X_dst.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
        X_dst.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
        X_dst.z = 1.f * z;

        X_src = inv_rotation*X_dst + inv_translation;

        //Texture ref frame is in (-0.5, -0.5) wrt pixel (0,0). 
        //That means that pixel (0,0)->(0.5,0.5) in texture. So: p_tex = p + 0.5
        x_src = ( X_src.x / X_src.z ) * intr.fx + intr.cx + 0.5f ;
        y_src = ( X_src.y / X_src.z ) * intr.fy + intr.cy + 0.5f;

        float z_src = tex2D(texRefDepth, x_src, y_src) ;

        if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
              || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows 
              || isnan(z) || isnan(z_src))  ) 
        {
          float3 v1_aux = (X_src - inv_translation) * (1.f / z);
          res = (z_src - inv_translation.z) / v1_aux.z;

          dst.ptr (y)[x] = res;
        }

      }

      ///////////////////////////////////////////////////////////////////////////////////////////////////////     
      __global__ void
      trafo3DKernelIntensityWithInvDepth  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depth_prev, 
                                           const Mat33 inv_rotation, const float3 inv_translation, const Intr intr, int colOff)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= dst.cols || y >= dst.rows)
          return;

        float3 X_dst, X_src;
        float x_src, y_src;
        float z = 1.f / depth_prev.ptr (y)[x];

        float res = numeric_limits<float>::quiet_NaN (); 
        dst.ptr (y)[x] = res;

        if (isnan(z))
          return;	

        X_dst.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
        X_dst.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
        X_dst.z = 1.f * z;

        X_src = inv_rotation*X_dst + inv_translation;

        //Texture ref frame is in (-0.5, -0.5) wrt pixel (0,0). 
        //That means that pixel (0,0)->(0.5,0.5) in texture. So: p_tex = p + 0.5
        x_src = ( X_src.x / X_src.z ) * intr.fx + intr.cx + 0.5f ;
        y_src = ( X_src.y / X_src.z ) * intr.fy + intr.cy + 0.5f;

        if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
              || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows 
              || isnan(z))  )
        {
          res = tex2D(texRefIntensity, x_src , y_src);
          dst.ptr (y)[x] = max (0.f, min (res, 255.f));
        }

      }   
      
      ///////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      trafo3DKernelInvDepth  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depth_prev, 
                              const Mat33 inv_rotation, const float3 inv_translation, const Intr intr, const Mat33 rotation, int colOff)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= dst.cols || y >= dst.rows)
          return;

        float3 X_dst, X_src;
        float x_src, y_src;
        float z = (1.0 / depth_prev.ptr (y)[x]);

        float res = numeric_limits<float>::quiet_NaN (); 
        dst.ptr (y)[x] = res;

        if (isnan(z))
          return;	

        X_dst.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
        X_dst.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
        X_dst.z = 1.f * z;

        X_src = inv_rotation*X_dst + inv_translation;

        //Texture ref frame is in (-0.5, -0.5) wrt pixel (0,0). 
        //That means that pixel (0,0)->(0.5,0.5) in texture. So: p_tex = p + 0.5
        x_src = ( X_src.x / X_src.z ) * intr.fx + intr.cx + 0.5f ;
        y_src = ( X_src.y / X_src.z ) * intr.fy + intr.cy + 0.5f;

        float z_src = (1.0 / tex2D(texRefDepth, x_src, y_src));

        if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
              || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows 
              || isnan(z) || isnan(z_src)))
        {  
         
          float3 v1_aux = (X_src - inv_translation) * (1.f / z);
          res = v1_aux.z / (z_src - inv_translation.z);

          dst.ptr (y)[x] = res;
        }
      }
      ///////////////////////////////////////////////////////////////////////////////////////////////////////      
      ///////////////////////////////////////////////////////////////////////////////////////////////////////  
      
      __global__ void  
      interpolateImagesAndGradientsKernel ( int cols, int rows, PtrStepSz<float> proj_dst, PtrStepSz<float> depth_interp, PtrStepSz<float> intensity_interp, 
                                                             PtrStepSz<float> xGradDepth_interp,  PtrStepSz<float> yGradDepth_interp,
                                                             PtrStepSz<float> xGradInt_interp, PtrStepSz<float> yGradInt_interp, int colOff)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= cols || y >= rows)
          return;
        
        depth_interp.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
        intensity_interp.ptr (y)[x] = numeric_limits<float>::quiet_NaN (); 
        xGradDepth_interp.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
        yGradDepth_interp.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
        xGradInt_interp.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
        yGradInt_interp.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
                                                             
        float x_proj_f = numeric_limits<float>::quiet_NaN ();
        float y_proj_f = numeric_limits<float>::quiet_NaN ();
        
        x_proj_f = proj_dst.ptr(y)[x] + 0.5f;
        y_proj_f = proj_dst.ptr(y + rows)[x]  + 0.5f;

        if (isnan(x_proj_f) || isnan(y_proj_f))
          return;
        	
        float depth_value = tex2D(texRefDepth, x_proj_f , y_proj_f);
        float intensity_value = tex2D(texRefIntensity, x_proj_f, y_proj_f);
        
        float res_hor;
        float res_vert;
        
        if (!isnan(depth_value))
        {
          depth_interp.ptr (y)[x] = depth_value;
          
          res_hor = 0;
          res_vert = 0;          

          for (int dx=-1; dx<2; dx++)
          {
            float temp;
            
            for (int dy=-1; dy<2; dy++)
            {
              float cx = min (max (0.f, x_proj_f + __int2float_rn(dx)), __int2float_rn(cols));
              float cy = min (max (0.f, y_proj_f + __int2float_rn(dy)), __int2float_rn(rows));

              float weight_hor = dx * (2 - dy * dy); 
              float weight_vert = dy * (2 - dx * dx);
              temp = tex2D(texRefDepth, cx, cy);
              
              res_hor +=  temp*weight_hor;
              res_vert +=  temp*weight_vert;
              
              if (isnan(temp)) 
                break;
            }
            
            if (isnan(temp)) 
                break;
          
          }
        }
        
        xGradDepth_interp.ptr (y)[x] =  (res_hor) / 8.f; 
        yGradDepth_interp.ptr (y)[x] = (res_vert) / 8.f;
        
        if (!isnan(intensity_value))
        {
          intensity_interp.ptr (y)[x] = intensity_value;
          
          res_hor = 0;
          res_vert = 0;

          for (int dx=-1; dx<2; dx++)
          {
            float temp;
            
            for (int dy=-1; dy<2; dy++)
            {
              float cx = min (max (0.f, x_proj_f + __int2float_rn(dx)), __int2float_rn(cols));
              float cy = min (max (0.f, y_proj_f + __int2float_rn(dy)), __int2float_rn(rows));

              float weight_hor = dx * (2 - dy * dy); 
              float weight_vert = dy * (2 - dx * dx);
              temp = tex2D(texRefIntensity, cx, cy);
              
              res_hor +=  temp*weight_hor;
              res_vert +=  temp*weight_vert;
              
              if (isnan(temp)) 
                break;
            }
            
            if (isnan(temp)) 
                break;
          
          }
        }
        
        xGradInt_interp.ptr (y)[x] =  (res_hor) / 8.f; 
        yGradInt_interp.ptr (y)[x] = (res_vert) / 8.f;
        
      }
      
      __global__ void 
      partialVisibilityKernel (int cols, int rows,  const PtrStepSz<float> depth_src, const PtrStep<float> depth_dst, PtrStep<float> gbuf,
                                     const Mat33 rotation, const float3 translation, const Intr intr, float geom_tol, int geom_error_type)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        
        float is_visible = 0.f;
        float is_valid = 0.f;
        
        if ((x<cols) && (y<rows))
        {
          float3 X_dst, X_src;
          float x_dst, y_dst;
          float z = depth_src.ptr (y)[x];
          
          if (geom_error_type == INV_DEPTH)
            z = 1.f / z;

          if (!isnan(z))
          {     
            is_valid = 1.f;
               
            X_src.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
            X_src.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
            X_src.z = 1.f * z;

            X_dst = rotation*X_src + translation;
            //X_dst = X_src;

            x_dst = ( X_dst.x / X_dst.z ) * intr.fx + intr.cx; 
            y_dst = ( X_dst.y / X_dst.z ) * intr.fy + intr.cy; 
            

            if ((x_dst > 0) && (x_dst < (cols-1)) && (y_dst > 0) && (y_dst < (rows-1)))
            {
                int x_int = __float2int_rn(x_dst);
                int y_int = __float2int_rn(y_dst);
                
                if (geom_error_type == INV_DEPTH)
                {
                  float inv_depth = 1.f / X_dst.z;
                  
                  if (abs(inv_depth -  depth_dst.ptr (y_int)[x_int]) < geom_tol)
                    is_visible = 1.f;
                }
                else //geom_error_type == DEPTH
                {
                  float depth = X_dst.z;
                  
                  if (abs(depth -  depth_dst.ptr (y_int)[x_int]) < geom_tol)
                    is_visible = 1.f;
                }
                  
                
                
                //Also consider a point not visible if it is projected in front of corresp. point in dst
                //that would be a point behind the camera or a sign of poor odometry estimation
                
                    
            }
          }
        }

        __syncthreads ();
        __shared__ float smem_visible[256];
        __shared__ float smem_valid[256];
        int tid = Block::flattenedThreadId ();
        
        __syncthreads ();
        smem_visible[tid] = is_visible;
        smem_valid[tid] = is_valid;
        
        __syncthreads ();
        reduce<256>(smem_visible);
        __syncthreads ();
        reduce<256>(smem_valid);
        __syncthreads ();
        
        if (tid == 0)
        {
          gbuf.ptr(0)[blockIdx.x + gridDim.x * blockIdx.y] = smem_visible[0];
          gbuf.ptr(1)[blockIdx.x + gridDim.x * blockIdx.y] = smem_valid[0];
        }

        return;
      }
      
      __global__ void
      finalVisibilityReductionKernel(int length, const PtrStep<float> gbuf, float* output)
      {
          const float *beg = gbuf.ptr (blockIdx.x);  //1 block per element in A and b
          const float *end = beg + length;   //length = num_constraints

          int tid = threadIdx.x;
          
          float sum = 0.f;
          for (const float *t = beg + tid; t < end; t += 512)  //Each thread sums #(num_contraints/CTA_SIZE) elements
            sum += *t;

          __syncthreads ();
          __shared__ float smem[512];

          smem[tid] = sum;
          __syncthreads ();

          reduce<512>(smem);

          if (tid == 0)
            output[blockIdx.x] = smem[0];
        }
        
        __global__ void 
      liftWarpAndProjKernelInvDepth (int cols, int rows,  const PtrStepSz<float> depth_src, 
                                     PtrStep<float> proj_dst, PtrStep<float> depth_dst, 
                                     const Mat33 rotation, const float3 translation, const Intr intr)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= cols || y >= rows)
          return;
          
        proj_dst.ptr(y)[x] = numeric_limits<float>::quiet_NaN (); 
        proj_dst.ptr(y + rows)[x] = numeric_limits<float>::quiet_NaN (); 	
        depth_dst.ptr(y)[x] = numeric_limits<float>::quiet_NaN ();  
        //depth_dst.ptr(y)[x] =  depth_src.ptr (y)[x]; 

        float3 X_dst, X_src;
        float x_dst, y_dst;
        float z = 1.f / depth_src.ptr (y)[x];

        if (isnan(z))
          return;
                
        X_src.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
        X_src.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
        X_src.z = 1.f * z;

        X_dst = rotation*X_src + translation;
        //X_dst = X_src;

        x_dst = ( X_dst.x / X_dst.z ) * intr.fx + intr.cx; 
        y_dst = ( X_dst.y / X_dst.z ) * intr.fy + intr.cy; 
        

        if ((x_dst > 0) && (x_dst < (cols-1)) && (y_dst > 0) && (y_dst < (rows-1)))
        {
          proj_dst.ptr(y)[x] = x_dst;
          proj_dst.ptr(y + rows)[x] = y_dst;
          depth_dst.ptr(y)[x] = 1.f / X_dst.z;	    	
        }

        

        return;
      }
      
      
      __global__ void 
      liftWarpAndProjKernelDepth (int cols, int rows,  const PtrStepSz<float> depth_src, 
                                     PtrStep<float> proj_dst, PtrStep<float> depth_dst, 
                                     const Mat33 rotation, const float3 translation, const Intr intr)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= cols || y >= rows)
          return;
          
        proj_dst.ptr(y)[x] = numeric_limits<float>::quiet_NaN (); 
        proj_dst.ptr(y + rows)[x] = numeric_limits<float>::quiet_NaN (); 	
        depth_dst.ptr(y)[x] = numeric_limits<float>::quiet_NaN ();  
        //depth_dst.ptr(y)[x] =  depth_src.ptr (y)[x];

        float3 X_dst, X_src;
        float x_dst, y_dst;
        float z = depth_src.ptr (y)[x];

        if (isnan(z))
          return;

                
        X_src.x = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx) * z;
        X_src.y = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy) * z;
        X_src.z = 1.f * z;

        X_dst = rotation*X_src + translation;
        //X_dst = X_src;

        x_dst = ( X_dst.x / X_dst.z ) * intr.fx + intr.cx; 
        y_dst = ( X_dst.y / X_dst.z ) * intr.fy + intr.cy; 

        if ((x_dst > 0) && (x_dst < (cols-1)) && (y_dst > 0) && (y_dst < (rows-1)))
        {
          proj_dst.ptr(y)[x] = x_dst;
          proj_dst.ptr(y + rows)[x] = y_dst;
          depth_dst.ptr(y)[x] = X_dst.z;	    	
        }
        
        return;
      }
 
      ///////////////////////////////////////////////////////////////
      float
      warpIntensityWithTrafo3D  (const IntensityMapf& src, IntensityMapf& dst, const DepthMapf& depth_prev, 
                                 Mat33 inv_rotation, float3 inv_translation, const Intr& intr)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        //src here is I1 and dst is I1 warped towards I0. depth_prev is D0.
        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);

        // Set texture reference parameters
        texRefIntensity.addressMode[0] = cudaAddressModeClamp;
        texRefIntensity.addressMode[1] = cudaAddressModeClamp;
        texRefIntensity.filterMode = cudaFilterModeLinear; //initial intensity maps are NaN free. It is always the unwarped intensity_curr
        texRefIntensity.normalized = false;

        int colOff = 0;

        cudaSafeCall( cudaBindTexture2D(0, texRefIntensity, src.ptr(), channelDesc, src.cols(), src.rows(),  src.step()) );

        dim3 block (32, 8);
        dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

        trafo3DKernelIntensity<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, inv_rotation, inv_translation, intr, colOff);

        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall (cudaUnbindTexture(texRefIntensity) );

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;
      };
      
      ///////////////////////////////////////////////////////////////
      float 
      warpDepthWithTrafo3D  (const DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, 
                             Mat33 inv_rotation, float3 inv_translation, const Intr& intr, Mat33 rotation)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        //src here is D1 and dst is D1 warped towards D0. depth_prev is D0.
        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);

        // Set texture reference parameters
        texRefDepth.addressMode[0] = cudaAddressModeClamp;
        texRefDepth.addressMode[1] = cudaAddressModeClamp;        
        texRefDepth.filterMode = cudaFilterModeLinear;  
        texRefDepth.normalized = false;

        int colOff = 0;

        cudaSafeCall (cudaBindTexture2D(0, texRefDepth, src.ptr(), channelDesc, src.cols(), src.rows(),  src.step()));

        dim3 block (32, 8);
        dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

        trafo3DKernelDepth<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, inv_rotation, inv_translation, intr, rotation, colOff);

        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall (cudaUnbindTexture(texRefDepth) );

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;

      };
      
      
      ///////////////////////////////////////////////////////////////
      float 
      warpIntensityWithTrafo3DInvDepth  (const IntensityMapf& src, IntensityMapf& dst, const DepthMapf& depth_prev, 
                                         Mat33 inv_rotation, float3 inv_translation, const Intr& intr)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        //src here is I1 and dst is I1 warped towards I0. depth_prev is D0.
        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);

        // Set texture reference parameters
        texRefIntensity.addressMode[0] = cudaAddressModeClamp;
        texRefIntensity.addressMode[1] = cudaAddressModeClamp;
        texRefIntensity.filterMode = cudaFilterModeLinear; //initial intensity maps are NaN free
        texRefIntensity.normalized = false;

        int colOff = 0;

        cudaSafeCall( cudaBindTexture2D(0, texRefIntensity, src.ptr(), channelDesc, src.cols(), src.rows(),  src.step()) );

        dim3 block (32, 8);
        dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

        trafo3DKernelIntensityWithInvDepth<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, inv_rotation, inv_translation, intr, colOff);

        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall (cudaUnbindTexture(texRefIntensity) );

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;

      };
      
      
      ///////////////////////////////////////////////////////////////
      float 
      warpInvDepthWithTrafo3D (const DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, 
                               Mat33 inv_rotation, float3 inv_translation, const Intr& intr, Mat33 rotation)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        //src here is D1 and dst is D1 warped towards D0. depth_prev is D0.
        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);

        // Set texture reference parameters
        texRefDepth.addressMode[0] = cudaAddressModeClamp;
        texRefDepth.addressMode[1] = cudaAddressModeClamp;        
        texRefDepth.filterMode = cudaFilterModeLinear; 
        texRefDepth.normalized = false;

        int colOff = 0;

        cudaSafeCall (cudaBindTexture2D(0, texRefDepth, src.ptr(), channelDesc, src.cols(), src.rows(),  src.step()));
        dim3 block (32, 8);
        dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

        trafo3DKernelInvDepth<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, inv_rotation, inv_translation, intr, rotation, colOff);

        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall (cudaUnbindTexture(texRefDepth) );

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;
      };
 
 
      float 
      getVisibilityRatio (DepthMapf& depth_src, DepthMapf& depth_dst,
                          Mat33 rotation, float3 translation, const Intr& intr, float& visibility_ratio, float geom_tol, int geom_error_type)  
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        dim3 block (32, 8);
        dim3 grid (divUp (depth_src.cols (), block.x), divUp (depth_src.rows (), block.y));
        
        DeviceArray2D<float> gbuf;
        float gbuf_length = grid.x*grid.y;
        gbuf.create(2,gbuf_length);
        
        DeviceArray<float> output_dev;
        output_dev.create(2);
        
        float output_host[2];

        partialVisibilityKernel<<<grid, block>>>(depth_src.cols(), depth_src.rows(),  depth_src, depth_dst, gbuf, rotation, translation, intr, geom_tol, geom_error_type);

        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        
        finalVisibilityReductionKernel<<<2,512>>>(gbuf_length, gbuf, output_dev);
        
        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        
        output_dev.download(output_host);
        visibility_ratio = output_host[0] / output_host[1];// / ((float) (depth_src.cols ()*depth_src.rows ()));
        
        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;
      };
      
      float 
      liftWarpAndProj (const DepthMapf& depth_src, DeviceArray2D<float>& proj_dst, DepthMapf& depth_dst,
                                   Mat33 rotation, float3 translation, const Intr& intr, int depth_error_type)  
       {
          /////////////////////////////////////////
          cudaEvent_t start, stop;
          float elapsedTime;
          cudaEventCreate(&start);
          cudaEventRecord(start,0);
          /////////////////////////////////////////////

          dim3 block (32, 8);
          dim3 grid (divUp (depth_src.cols (), block.x), divUp (depth_src.rows (), block.y));
          
          if (depth_error_type == DEPTH)
          {
            liftWarpAndProjKernelDepth<<<grid, block>>>(depth_src.cols(), depth_src.rows(),  depth_src, proj_dst, depth_dst, rotation, translation, intr);	          

            cudaSafeCall (cudaDeviceSynchronize());
            cudaSafeCall ( cudaGetLastError () );		
          }
          else
          {            
            liftWarpAndProjKernelInvDepth<<<grid, block>>>(depth_src.cols(), depth_src.rows(),  depth_src, proj_dst, depth_dst, rotation, translation, intr);			
            cudaSafeCall (cudaDeviceSynchronize());
            cudaSafeCall ( cudaGetLastError () );		
          }	

          //////////////////////////////////////////////////////////////
          cudaEventCreate(&stop);
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&elapsedTime, start,stop);
          //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
          ////////////////////////////////////////////////////////

          return elapsedTime;
       };
       
       ///////////////////////////////////////////////////////////////
      float 
      interpolateImagesAndGradients  (DeviceArray2D<float>& proj_dst, const DepthMapf& depth, const IntensityMapf& intensity,
                                      DepthMapf& depth_interp, IntensityMapf& intensity_interp, 
                                      DeviceArray2D<float>& xGradDepth_interp, DeviceArray2D<float>& yGradDepth_interp,
                                      DeviceArray2D<float>& xGradInt_interp, DeviceArray2D<float>& yGradInt_interp)
                              
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        //src here is D1 and dst is D1 warped towards D0. depth_prev is D0.
        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8*sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);

        // Set texture reference parameters
        texRefDepth.addressMode[0] = cudaAddressModeClamp;
        texRefDepth.addressMode[1] = cudaAddressModeClamp;        
        texRefDepth.filterMode = cudaFilterModeLinear;  
        texRefDepth.normalized = false;

        int colOff = 0;

        cudaSafeCall (cudaBindTexture2D(0, texRefDepth, depth.ptr(), channelDesc, depth.cols(), depth.rows(),  depth.step()));
        cudaSafeCall (cudaBindTexture2D(0, texRefIntensity, intensity.ptr(), channelDesc, intensity.cols(), intensity.rows(),  intensity.step()));

        dim3 block (32, 8);
        dim3 grid (divUp (depth_interp.cols (), block.x), divUp (depth_interp.rows (), block.y));

        interpolateImagesAndGradientsKernel<<<grid, block>>>(depth.cols(), depth.rows(), proj_dst, depth_interp, intensity_interp, 
                                                             xGradDepth_interp,  yGradDepth_interp,
                                                             xGradInt_interp, yGradInt_interp, colOff);

        cudaSafeCall (cudaDeviceSynchronize());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall (cudaUnbindTexture(texRefDepth) );
        cudaSafeCall (cudaUnbindTexture(texRefIntensity) );

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("reduce to final depth System took : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////

        return elapsedTime;

      };	
    }
  }
}

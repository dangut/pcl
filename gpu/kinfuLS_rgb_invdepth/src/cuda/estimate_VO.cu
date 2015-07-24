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


using namespace pcl::device;

namespace pcl
{
  namespace device
  {
    namespace kinfuRGBD
    {
      typedef double float_type;
      
      __constant__ float THRESHOLD_HUBER_DEV = 1.345f;
      __constant__ float THRESHOLD_TUKEY_DEV = 4.685f;
      __constant__ float STUDENT_DOF_DEV = 5.f;

      
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

      struct constraintsHandler
      {
        enum
        {
          CTA_SIZE_X = 32,
          CTA_SIZE_Y = 8,
          CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
        };


        float3 delta_trans;
        float3 delta_rot;

        PtrStep<float> D0;
        PtrStep<float> I0;
        PtrStep<float> D1;
        PtrStep<float> I1;
        
        PtrStep<float> proj_in0;

        Intr intr;

        PtrStep<float> gradI0_x;
        PtrStep<float> gradI0_y;

        PtrStep<float> gradD0_x;
        PtrStep<float> gradD0_y;

        int cols;
        int rows;
        float nel;

        int Mestimator;
        int depth_error_type;
        int weighting;
        float sigma_int;
        float sigma_depth;
        float bias_int;
        float bias_depth;

        mutable PtrStep<float_type> gbuf;


        __device__ __forceinline__ float
        computeWeight (float *error) const
        {
          float weight = 1.f;

          if (Mestimator == HUBER)
          {
            if (fabs(*error) > THRESHOLD_HUBER_DEV)
              weight = (THRESHOLD_HUBER_DEV / fabs(*error));
          }
          else if (Mestimator == TUKEY)
          {
            if (fabs(*error) < THRESHOLD_TUKEY_DEV)
            {
              float aux1 = ( (*error) / THRESHOLD_TUKEY_DEV ) * ( (*error) / THRESHOLD_TUKEY_DEV );
              weight = (1.f - aux1) * (1.f - aux1);
            }
            else
              weight = 0.f;
          }
          else if (Mestimator == STUDENT)
          {   
            weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + (*error)*(*error));
          }

          return weight;
        }


        __device__ __forceinline__ void
        intensityConstraint (int x, int y, float *row, float *error) const
        {
          float w0;

          if (depth_error_type == INV_DEPTH)
            w0 = D0.ptr (y)[x];
          else
            w0 = (1.f / D0.ptr (y)[x]);

          float i0 = I0.ptr(y)[x];
          float i1 = I1.ptr(y)[x];
          float gradx = gradI0_x.ptr (y)[x];
          float grady = gradI0_y.ptr (y)[x];
          //float gradMagnitude = sqrt(gradx*gradx + grady*grady);

          if (isnan (w0) || isnan (i0) || isnan (i1) || isnan (gradx) || isnan(grady))
            return;	

          float3 p;
          p.x = (__int2float_rn(x) - intr.cx) / intr.fx ;
          p.y = (__int2float_rn(y) - intr.cy) / intr.fy ;
          p.z = 1.f;

          float3 gradI0_times_KP;
          gradI0_times_KP.x = gradx * intr.fx * w0;
          gradI0_times_KP.y = grady * intr.fy * w0;
          gradI0_times_KP.z = - (gradI0_times_KP.x * p.x + gradI0_times_KP.y * p.y);

          float weight = 1.f / sigma_int;

          float3 row_rot = - cross(gradI0_times_KP,p) * (1.f / w0) ;

          float3 row_trans = (gradI0_times_KP);	

          float row_alpha = -i0 ;
          float row_beta = -1.f;

          *(float3*)&row[0] = row_trans * weight;
          *(float3*)&row[3] = row_rot * weight;
          
          //B_SIZE = 6 ->these two terms are not reduced.
          row[6] = row_alpha * weight;
          row[7] = row_beta * weight;

          *error = - (i1-i0)*weight ; 
          
          //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot ) + row_alpha*1.f + row_beta*0.f + i1)*weight ;
          
          return;
        }



        __device__ __forceinline__ void
        depthConstraint (int x, int y, float *row, float *error) const
        {
          float d0 = D0.ptr (y)[x];
          float d1 = D1.ptr (y)[x];
          float gradx = gradD0_x.ptr (y)[x];
          float grady = gradD0_y.ptr (y)[x];

          if (isnan (d0) || isnan (d1) || isnan (gradx) || isnan(grady))
            return;	

          float3 p;
          p.x = (__int2float_rn(x) - intr.cx) / intr.fx ;
          p.y = (__int2float_rn(y) - intr.cy) / intr.fy ;
          p.z = 1.f;

          float3 gradD0_times_KP_minus_ez;
          gradD0_times_KP_minus_ez.x = (gradx)*intr.fx * (1.f / d0);  //[m]/[px]*([px]/[1])* (1/[m]) = [1]
          gradD0_times_KP_minus_ez.y = (grady)*intr.fy * (1.f / d0);
          gradD0_times_KP_minus_ez.z = - (gradD0_times_KP_minus_ez.x * p.x + gradD0_times_KP_minus_ez.y * p.y) - 1.f;

          float weight = 1.f / sigma_depth;

          float3 row_rot;
          row_rot = - cross(gradD0_times_KP_minus_ez, p) * d0; //a^T->gradD0, b->p. a^T*mcross(b) = mcross(a) * b = cross(a,b)

          float3 row_trans;
          row_trans = (gradD0_times_KP_minus_ez); 
          
          float b = (d1 - d0);

          *(float3*)&row[0] = row_trans * weight;
          *(float3*)&row[3] = row_rot * weight;
          row[6] = 0.f;
          row[7] = 0.f;

          *error = - b*weight;       
          //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot )  + b)*weight;   
          return;
        }

        __device__ __forceinline__ void
        invDepthConstraint (int x, int y, float *row, float *error) const
        {
          float w0 = D0.ptr (y)[x];
          float w1 = D1.ptr (y)[x];
          float gradx = gradD0_x.ptr (y)[x] ;
          float grady = gradD0_y.ptr (y)[x] ;
          //float gradMagnitude = sqrt(gradx*gradx + grady*grady);

          if (isnan (w0) || isnan (w1) || isnan (gradx) || isnan(grady))
          return;	


          float3 p;
          p.x = (__int2float_rn(x) - intr.cx) / intr.fx ;
          p.y = (__int2float_rn(y) - intr.cy) / intr.fy ;
          p.z = 1.f;

          float3 gradD0_times_KP_plus_invdepthsqez;
          gradD0_times_KP_plus_invdepthsqez.x = (gradx)*intr.fx * w0;  //[mm]/[px]*([px]/[1])* (1/[mm]) = [1]
          gradD0_times_KP_plus_invdepthsqez.y = (grady)*intr.fy * w0;
          gradD0_times_KP_plus_invdepthsqez.z = - (gradD0_times_KP_plus_invdepthsqez.x * p.x + gradD0_times_KP_plus_invdepthsqez.y * p.y) + w0*w0;

          float weight = 1.f / sigma_depth;

          float3 row_rot;
          row_rot = - cross(gradD0_times_KP_plus_invdepthsqez, p) *(1.f / w0); //a^T->gradD0, b->p. a^T*mcross(b) = mcross(a) * b = cross(a,b)

          float3 row_trans;
          row_trans = (gradD0_times_KP_plus_invdepthsqez); 

          //float row_alpha = 0.f;
          //float row_beta = 0.f;
          float b = (w1 - w0);

          *(float3*)&row[0] = row_trans * weight;
          *(float3*)&row[3] = row_rot * weight;
          row[6] = 0.f;
          row[7] = 0.f;

          *error = - b*weight; 
          //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot )  + b)*weight;             
          
          return;
        }
        
        __device__ __forceinline__ void
        computeSystem () const
        {
          int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

          float row_int[8];
          float row_depth[8];	
          float error_int;
          float error_depth;
          float weight_int = 0.f;
          float weight_depth = 0.f;


          for (int i=0; i < 8; i++)
          { 
            row_int[i] = 0.f;
            row_depth[i] = 0.f;
          }
          
          error_int = 0.f;
          error_depth = 0.f;

          if (x < cols && y < rows)
          {
            if (depth_error_type == DEPTH)
              depthConstraint (x, y, row_depth, &error_depth); 
            else if (depth_error_type == INV_DEPTH)
              invDepthConstraint(x, y, row_depth, &error_depth); 

            intensityConstraint (x, y, row_int, &error_int); 

            float error_depth_unbiased = error_depth - (bias_depth / sigma_depth);
            float error_int_unbiased = error_int - (bias_int / sigma_int);
            weight_depth = computeWeight(&error_depth_unbiased)*__int2float_rn(1 - (weighting == INT_ONLY));
            weight_int = computeWeight(&error_int_unbiased)*__int2float_rn(1 - (weighting == DEPTH_ONLY));

            if (weighting == MIN_WEIGHT)
            {
              //float weight_min = min(weight_depth, weight_int);
              //weight_depth = weight_min;
              //weight_int = weight_min;
              weight_int = min(weight_depth, weight_int);
            }
            //float weight_min = min(weight_depth, weight_int);
          }

          __shared__ float smem[CTA_SIZE];
          int tid = Block::flattenedThreadId ();

          int shift = 0;
          for (int i = 0; i < B_SIZE; ++i)        //rows
          {
            #pragma unroll
            for (int j = i; j < B_SIZE; ++j)          // cols + b
            {
              __syncthreads ();
              smem[tid] = weight_int*(row_int[i] * row_int[j]) + weight_depth*(row_depth[i] * row_depth[j]);
              __syncthreads ();

              reduce<CTA_SIZE>(smem);

              if (tid == 0)
              gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
            }

            __syncthreads ();
            smem[tid] = weight_int*(row_int[i] * (error_int)) + weight_depth*(row_depth[i] * (error_depth));
            __syncthreads ();

            reduce<CTA_SIZE>(smem);

            if (tid == 0)
              gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
          }
        }
        
        __device__ __forceinline__ void
        intensityConstraintInterp (int x, int y, float *row, float *error) const
        {
          float w0;

          if (depth_error_type == INV_DEPTH)
            w0 = D0.ptr (y)[x];
          else
            w0 = (1.f / D0.ptr (y)[x]);

          float i0 = I0.ptr(y)[x];
          float i1 = I1.ptr(y)[x];
          float gradx = gradI0_x.ptr (y)[x];
          float grady = gradI0_y.ptr (y)[x];
          float x_f = proj_in0.ptr (y)[x];
          float y_f = proj_in0.ptr (y+rows)[x];
          
          
          //float gradMagnitude = sqrt(gradx*gradx + grady*grady);

          if (isnan (w0) || isnan (i0) || isnan (i1) || isnan (gradx) || isnan(grady) || isnan(x_f) || isnan(y_f))
            return;	

          float3 p;
          p.x = ( x_f - intr.cx) / intr.fx ;
          p.y = ( y_f - intr.cy) / intr.fy ;
          p.z = 1.f;

          float3 gradI0_times_KP;
          gradI0_times_KP.x = gradx * intr.fx * w0;
          gradI0_times_KP.y = grady * intr.fy * w0;
          gradI0_times_KP.z = - (gradI0_times_KP.x * p.x + gradI0_times_KP.y * p.y);

          float weight = 1.f / sigma_int;

          float3 row_rot = - cross(gradI0_times_KP,p) * (1.f / w0) ;

          float3 row_trans = (gradI0_times_KP);	

          float row_alpha = -i0 ;
          float row_beta = -1.f;

          *(float3*)&row[0] = row_trans * weight;
          *(float3*)&row[3] = row_rot * weight;
          
          //B_SIZE = 6 ->these two terms are not reduced.
          row[6] = row_alpha * weight;
          row[7] = row_beta * weight;

          *error = - (i1-i0)*weight ; 
          
          //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot ) + row_alpha*1.f + row_beta*0.f + i1)*weight ;
          
          return;
        }



        __device__ __forceinline__ void
        depthConstraintInterp (int x, int y, float *row, float *error) const
        {
          float d0 = D0.ptr (y)[x];
          float d1 = D1.ptr (y)[x];
          float gradx = gradD0_x.ptr (y)[x];
          float grady = gradD0_y.ptr (y)[x];
          float x_f = proj_in0.ptr (y)[x];
          float y_f = proj_in0.ptr (y+rows)[x];

          if (isnan (d0) || isnan (d1) || isnan (gradx) || isnan(grady) || isnan(x_f) || isnan(y_f))
            return;	

          float3 p;
          p.x = ( x_f - intr.cx) / intr.fx ;
          p.y = ( y_f - intr.cy) / intr.fy ;
          p.z = 1.f;

          float3 gradD0_times_KP_minus_ez;
          gradD0_times_KP_minus_ez.x = (gradx)*intr.fx * (1.f / d0);  //[m]/[px]*([px]/[1])* (1/[m]) = [1]
          gradD0_times_KP_minus_ez.y = (grady)*intr.fy * (1.f / d0);
          gradD0_times_KP_minus_ez.z = - (gradD0_times_KP_minus_ez.x * p.x + gradD0_times_KP_minus_ez.y * p.y) - 1.f;

          float weight = 1.f / sigma_depth;

          float3 row_rot;
          row_rot = - cross(gradD0_times_KP_minus_ez, p) * d0; //a^T->gradD0, b->p. a^T*mcross(b) = mcross(a) * b = cross(a,b)

          float3 row_trans;
          row_trans = (gradD0_times_KP_minus_ez); 
          
          float b = (d1 - d0);

          *(float3*)&row[0] = row_trans * weight;
          *(float3*)&row[3] = row_rot * weight;
          row[6] = 0.f;
          row[7] = 0.f;

          *error = - b*weight;       
          //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot )  + b)*weight;   
          return;
        }

        __device__ __forceinline__ void
        invDepthConstraintInterp (int x, int y, float *row, float *error) const
        {
          float w0 = D0.ptr (y)[x];
          float w1 = D1.ptr (y)[x];
          float gradx = gradD0_x.ptr (y)[x] ;
          float grady = gradD0_y.ptr (y)[x] ;
          //float gradMagnitude = sqrt(gradx*gradx + grady*grady);
          float x_f = proj_in0.ptr (y)[x];
          float y_f = proj_in0.ptr (y+rows)[x];

          if (isnan (w0) || isnan (w1) || isnan (gradx) || isnan(grady) || isnan(x_f) || isnan(y_f))
            return;	

          float3 p;
          p.x = ( x_f - intr.cx) / intr.fx ;
          p.y = ( y_f - intr.cy) / intr.fy ;
          p.z = 1.f;

          float3 gradD0_times_KP_plus_invdepthsqez;
          gradD0_times_KP_plus_invdepthsqez.x = (gradx)*intr.fx * w0;  //[mm]/[px]*([px]/[1])* (1/[mm]) = [1]
          gradD0_times_KP_plus_invdepthsqez.y = (grady)*intr.fy * w0;
          gradD0_times_KP_plus_invdepthsqez.z = - (gradD0_times_KP_plus_invdepthsqez.x * p.x + gradD0_times_KP_plus_invdepthsqez.y * p.y) + w0*w0;

          float weight = 1.f / sigma_depth;

          float3 row_rot;
          row_rot = - cross(gradD0_times_KP_plus_invdepthsqez, p) *(1.f / w0); //a^T->gradD0, b->p. a^T*mcross(b) = mcross(a) * b = cross(a,b)

          float3 row_trans;
          row_trans = (gradD0_times_KP_plus_invdepthsqez); 

          //float row_alpha = 0.f;
          //float row_beta = 0.f;
          float b = (w1 - w0);

          *(float3*)&row[0] = row_trans * weight;
          *(float3*)&row[3] = row_rot * weight;
          row[6] = 0.f;
          row[7] = 0.f;

          *error = - b*weight; 
          //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot )  + b)*weight;             
          
          return;
        }
        
        __device__ __forceinline__ void
        computeSystemInterp () const
        {
          int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
          int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

          float row_int[8];
          float row_depth[8];	
          float error_int;
          float error_depth;
          float weight_int = 0.f;
          float weight_depth = 0.f;


          for (int i=0; i < 8; i++)
          { 
            row_int[i] = 0.f;
            row_depth[i] = 0.f;
          }
          
          error_int = 0.f;
          error_depth = 0.f;

          if (x < cols && y < rows)
          {
            if (depth_error_type == DEPTH)
              depthConstraintInterp (x, y, row_depth, &error_depth); 
            else if (depth_error_type == INV_DEPTH)
              invDepthConstraintInterp (x, y, row_depth, &error_depth); 

            intensityConstraintInterp (x, y, row_int, &error_int); 

            float error_depth_unbiased = error_depth - (bias_depth / sigma_depth);
            float error_int_unbiased = error_int - (bias_int / sigma_int);
            weight_depth = computeWeight(&error_depth_unbiased)*__int2float_rn(1 - (weighting == INT_ONLY));
            weight_int = computeWeight(&error_int_unbiased)*__int2float_rn(1 - (weighting == DEPTH_ONLY));

            if (weighting == MIN_WEIGHT)
            {
              //float weight_min = min(weight_depth, weight_int);
              //weight_depth = weight_min;
              //weight_int = weight_min;
              weight_int = min(weight_depth, weight_int);
            }
            //float weight_min = min(weight_depth, weight_int);
          }

          __shared__ float smem[CTA_SIZE];
          int tid = Block::flattenedThreadId ();

          int shift = 0;
          for (int i = 0; i < B_SIZE; ++i)        //rows
          {
            #pragma unroll
            for (int j = i; j < B_SIZE; ++j)          // cols + b
            {
              __syncthreads ();
              smem[tid] = weight_int*(row_int[i] * row_int[j]) + weight_depth*(row_depth[i] * row_depth[j]);
              __syncthreads ();

              reduce<CTA_SIZE>(smem);

              if (tid == 0)
              gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
            }

            __syncthreads ();
            smem[tid] = weight_int*(row_int[i] * (error_int)) + weight_depth*(row_depth[i] * (error_depth));
            __syncthreads ();

            reduce<CTA_SIZE>(smem);

            if (tid == 0)
              gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
          }
        }
        
        
      };
 

      __global__ void
      partialSumKernel (const constraintsHandler ch) 
      {
        ch.computeSystem ();
      }
      
      __global__ void
      partialSumKernelInterp (const constraintsHandler ch) 
      {
        ch.computeSystemInterp ();
      }

      struct FinalReductionHandlerVO
      {
        enum
        {
          CTA_SIZE = 512,
          STRIDE = CTA_SIZE,
        };

        PtrStep<float_type> gbuf;
        int length;
        mutable float_type* output;

        __device__ __forceinline__ void
        operator () () const
        {
          const float_type *beg = gbuf.ptr (blockIdx.x);  //1 block per element in A and b
          const float_type *end = beg + length;   //length = num_constraints

          int tid = threadIdx.x;
          
          float_type sum = 0.f;
          for (const float_type *t = beg + tid; t < end; t += STRIDE)  //Each thread sums #(num_contraints/CTA_SIZE) elements
            sum += *t;

          __shared__ float_type smem[CTA_SIZE];

          smem[tid] = sum;
          __syncthreads ();

          reduce<CTA_SIZE>(smem);

          if (tid == 0)
            output[blockIdx.x] = smem[0];
        }
      };

      __global__ void
      FinalReductionKernel (const FinalReductionHandlerVO frh) 
      {
        frh ();
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      float 
      buildSystem (const float3 delta_trans, const float3 delta_rot,
                   const DepthMapf& D0, const IntensityMapf& I0,
                   const GradientMap& gradD0_x, const GradientMap& gradD0_y, 
                   const GradientMap& gradI0_x, const GradientMap& gradI0_y,
                   const DepthMapf& D1, const IntensityMapf& I1,
                   int Mestimator, int depth_error_type, int weighting,
                   float sigma_depth, float sigma_int,
                   float bias_depth, float bias_int,
                   const Intr& intr, const int size_A,
                   DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host)
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        int cols = D0.cols();
        int rows = D0.rows();

        constraintsHandler ch;

        ch.delta_trans = delta_trans;
        ch.delta_rot = delta_rot;

        ch.D0 = D0;
        ch.gradD0_x = gradD0_x;
        ch.gradD0_y = gradD0_y;

        ch.I0 = I0;
        ch.gradI0_x = gradI0_x;
        ch.gradI0_y = gradI0_y;

        ch.D1 = D1;
        ch.I1 = I1;

        ch.intr.fx = intr.fx; // / 1000.f;
        ch.intr.fy = intr.fy; // / 1000.f;
        ch.intr.cx = intr.cx; // / 1000.f;
        ch.intr.cy = intr.cy; // / 1000.f;

        ch.cols = cols;
        ch.rows = rows;
        ch.nel = (float) (cols*rows);
        ch.Mestimator = Mestimator;
        ch.depth_error_type = depth_error_type;
        ch.weighting = weighting;
        ch.sigma_depth = sigma_depth; //sigma_{u/fb} in [m^-1] ->[m^-1]
        ch.sigma_int = sigma_int;
        ch.bias_depth = bias_depth;
        ch.bias_int = bias_int;

        dim3 block (constraintsHandler::CTA_SIZE_X, constraintsHandler::CTA_SIZE_Y);
        dim3 grid (1, 1, 1);
        grid.x = divUp (cols, block.x);
        grid.y = divUp (rows, block.y);

        mbuf.create (TOTAL_SIZE);

        if (gbuf.rows () != TOTAL_SIZE || gbuf.cols () != (int)(grid.x * grid.y))
          gbuf.create (TOTAL_SIZE, grid.x * grid.y);

        ch.gbuf = gbuf; 
            
        cudaFuncSetCacheConfig (partialSumKernel, cudaFuncCachePreferL1);
        partialSumKernel<<<grid, block>>>(ch);

        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());

        //printFuncAttrib(partialSumKernel_depth);

        FinalReductionHandlerVO frh;
        frh.gbuf = gbuf;
        frh.length = grid.x * grid.y;
        frh.output = mbuf;

        cudaFuncSetCacheConfig (FinalReductionKernel, cudaFuncCachePreferL1);
        FinalReductionKernel<<<TOTAL_SIZE, FinalReductionHandlerVO::CTA_SIZE>>>(frh);
        cudaSafeCall (cudaGetLastError ());
        
        //printFuncAttrib(FinalReductionKernel);
        cudaSafeCall (cudaDeviceSynchronize ());

        float_type host_data[TOTAL_SIZE];
        mbuf.download (host_data);

        int shift = 0;
        for (int i = 0; i < B_SIZE; ++i)  //rows
        {
          for (int j = i; j < B_SIZE + 1; ++j)    // cols + b
          {
            float_type value = host_data[shift++];
            
            if (j == B_SIZE)       // vector b
              vectorB_host[i] = value;
            else
              matrixA_host[j * B_SIZE + i] = matrixA_host[i * B_SIZE + j] = value;
          }
        }

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("time_visOdo : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////
        
        return elapsedTime;
      };        
      
      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      float 
      buildSystem2 (const float3 delta_trans, const float3 delta_rot,
                   const DeviceArray2D<float> proj_in0,
                   const DepthMapf& D0, const IntensityMapf& I0,
                   const GradientMap& gradD0_x, const GradientMap& gradD0_y, 
                   const GradientMap& gradI0_x, const GradientMap& gradI0_y,
                   const DepthMapf& D1, const IntensityMapf& I1,
                   int Mestimator, int depth_error_type, int weighting,
                   float sigma_depth, float sigma_int,
                   float bias_depth, float bias_int,
                   const Intr& intr, const int size_A,
                   DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host)
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////

        int cols = D0.cols();
        int rows = D0.rows();

        constraintsHandler ch;

        ch.delta_trans = delta_trans;
        ch.delta_rot = delta_rot;

        ch.D0 = D0;
        ch.gradD0_x = gradD0_x;
        ch.gradD0_y = gradD0_y;

        ch.I0 = I0;
        ch.gradI0_x = gradI0_x;
        ch.gradI0_y = gradI0_y;

        ch.D1 = D1;
        ch.I1 = I1;
        
        ch.proj_in0 = proj_in0;

        ch.intr.fx = intr.fx; // / 1000.f;
        ch.intr.fy = intr.fy; // / 1000.f;
        ch.intr.cx = intr.cx; // / 1000.f;
        ch.intr.cy = intr.cy; // / 1000.f;

        ch.cols = cols;
        ch.rows = rows;
        ch.nel = (float) (cols*rows);
        ch.Mestimator = Mestimator;
        ch.depth_error_type = depth_error_type;
        ch.weighting = weighting;
        ch.sigma_depth = sigma_depth; //sigma_{u/fb} in [m^-1] ->[m^-1]
        ch.sigma_int = sigma_int;
        ch.bias_depth = bias_depth;
        ch.bias_int = bias_int;

        dim3 block (constraintsHandler::CTA_SIZE_X, constraintsHandler::CTA_SIZE_Y);
        dim3 grid (1, 1, 1);
        grid.x = divUp (cols, block.x);
        grid.y = divUp (rows, block.y);

        mbuf.create (TOTAL_SIZE);

        if (gbuf.rows () != TOTAL_SIZE || gbuf.cols () != (int)(grid.x * grid.y))
          gbuf.create (TOTAL_SIZE, grid.x * grid.y);

        ch.gbuf = gbuf; 
            
        cudaFuncSetCacheConfig (partialSumKernel, cudaFuncCachePreferL1);
        partialSumKernelInterp<<<grid, block>>>(ch);

        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());

        //printFuncAttrib(partialSumKernel_depth);

        FinalReductionHandlerVO frh;
        frh.gbuf = gbuf;
        frh.length = grid.x * grid.y;
        frh.output = mbuf;

        cudaFuncSetCacheConfig (FinalReductionKernel, cudaFuncCachePreferL1);
        FinalReductionKernel<<<TOTAL_SIZE, FinalReductionHandlerVO::CTA_SIZE>>>(frh);
        cudaSafeCall (cudaGetLastError ());
        
        //printFuncAttrib(FinalReductionKernel);
        cudaSafeCall (cudaDeviceSynchronize ());

        float_type host_data[TOTAL_SIZE];
        mbuf.download (host_data);

        int shift = 0;
        for (int i = 0; i < B_SIZE; ++i)  //rows
        {
          for (int j = i; j < B_SIZE + 1; ++j)    // cols + b
          {
            float_type value = host_data[shift++];
            
            if (j == B_SIZE)       // vector b
              vectorB_host[i] = value;
            else
              matrixA_host[j * B_SIZE + i] = matrixA_host[i * B_SIZE + j] = value;
          }
        }

        //////////////////////////////////////////////////////////////
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("time_visOdo : %f ms\n" ,elapsedTime);
        ////////////////////////////////////////////////////////
        
        return elapsedTime;
      };        
 
    }
  }
}

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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cmath>

namespace pcl
{
  namespace device
  {
    namespace kinfuRGBD
    {

      __constant__ float THRESHOLD_HUBER_DEV = 1.345;
      __constant__ float THRESHOLD_TUKEY_DEV = 4.685;
      __constant__ float STUDENT_DOF_DEV = 5.f;

      template<int CTA_SIZE_, typename T>
      static __device__ __forceinline__ void reduce(volatile T* buffer)
      {
        int tid = threadIdx.x;
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

      
      struct errorHandler
      {
        enum
        {
          CTA_SIZE_X = 32,
          CTA_SIZE_Y = 8
        };

        PtrStep<float> D0;
        PtrStep<float> im0;
        PtrStep<float> im1;

        int cols;
        int rows;	
        
        mutable float* error;
        

        __device__ __forceinline__ void
        computeError () const
        {
          int x = threadIdx.x + blockIdx.x * blockDim.x;
          int y = threadIdx.y + blockIdx.y * blockDim.y;

          if (x >= cols || y >= rows)
            return;

          int error_idx = y*cols + x;
          float value = im1.ptr (y)[x] - im0.ptr (y)[x];
          //asume that nan pixels are randomly spread in pixels in odd and even positions 
          //distribute them in both tails of the sorted vector to contamine the median as less as possible
          if (isnan(value))
          {
            if (error_idx & 1)
              value = numeric_limits<float>::infinity ();
            else
              value = - numeric_limits<float>::infinity ();
          }
          
          error[error_idx] = value;	
                    
          return;					    
        }		
      };
  	
      struct sigmaHandler
      {
        enum
        {
          CTA_SIZE = 256,
          CTA_SIZE_DR = 512,
          STRIDE = CTA_SIZE_DR
        };

        float* error;

        int nel;

        int Mestimator;
        float sigma;
        float bias;
        float* moments_dev;

        mutable PtrStep<float> gbuf;

        __device__ __forceinline__ void
        partialSigma () const
        {

          int x = threadIdx.x + blockIdx.x * CTA_SIZE;

          __shared__ float smem_N[CTA_SIZE];
          __shared__ float smem_sum[CTA_SIZE];
          
          float is_valid = 0.f;
          float weighted_sq_error = 0.f;          

          if (x < nel)
          {
            if ( !(isinf(error[x])) && !(isnan(error[x])))
            {
              float error_unbias = error[x] - bias;
              float error_normalised = error_unbias/sigma;
              float weight = 1.f;
              is_valid = 1.f;

              if ( (Mestimator == HUBER) && (fabs(error_normalised) > THRESHOLD_HUBER_DEV))
              {
                weight = (THRESHOLD_HUBER_DEV / fabs(error_normalised));
              }
              else if (Mestimator == TUKEY)
              {
                if (fabs(error_normalised) < THRESHOLD_TUKEY_DEV)
                {
                  float aux1 = ( error_normalised / THRESHOLD_TUKEY_DEV ) * ( error_normalised / THRESHOLD_TUKEY_DEV );
                  weight = (1.f - aux1) * (1.f - aux1);		
                }
                else
                {
                  weight = 0.f;	
                  is_valid = 0.f;
                }
              }
              else if (Mestimator == STUDENT)
              {
                weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + error_normalised*error_normalised);             
              } 
              
              weighted_sq_error = error_unbias*error_unbias*weight;   
            }
          }

          __syncthreads ();
          smem_N[threadIdx.x] = is_valid;
          smem_sum[threadIdx.x] = weighted_sq_error;     
          __syncthreads ();

          reduce<CTA_SIZE>(smem_N);	__syncthreads ();
          reduce<CTA_SIZE>(smem_sum);	__syncthreads ();					

          if (threadIdx.x == 0)
          {
            gbuf.ptr(0)[blockIdx.x] = smem_N[0];
            gbuf.ptr(1)[blockIdx.x] = smem_sum[0];	
          }
          
          return;			
        }
        
        __device__ __forceinline__ void
        partialBias () const
        {
          int x = threadIdx.x + blockIdx.x * CTA_SIZE;

          __shared__ float smem_weight[CTA_SIZE];
          __shared__ float smem_weighted_res[CTA_SIZE];
          
          float weight = 0.f;
          float weighted_error = 0.f;

          if (x < nel)
          {
            if ( !(isinf(error[x])) && !(isnan(error[x])))
            {
              float error_normalised = (error[x]-bias)/sigma;
              weight = 1.f;

              if ( (Mestimator == HUBER) && (fabs(error_normalised) > THRESHOLD_HUBER_DEV))
              {
                weight = (THRESHOLD_HUBER_DEV / fabs(error_normalised));
              }
              else if (Mestimator == TUKEY)
              {
                if (fabs(error_normalised) < THRESHOLD_TUKEY_DEV)
                {
                  float aux1 = ( error_normalised / THRESHOLD_TUKEY_DEV ) * ( error_normalised / THRESHOLD_TUKEY_DEV );
                  weight = (1.f - aux1) * (1.f - aux1);		
                }
                else
                {
                  weight = 0.f;	
                }
              }
              else if (Mestimator == STUDENT)
              {
                weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + error_normalised*error_normalised);             
              } 
              
              weighted_error = error[x]*weight;   
            }
          }

          __syncthreads ();
          smem_weight[threadIdx.x] = weight;
          smem_weighted_res[threadIdx.x] = weighted_error;     
          __syncthreads ();

          reduce<CTA_SIZE>(smem_weight);	__syncthreads ();
          reduce<CTA_SIZE>(smem_weighted_res);	__syncthreads ();					

          if (threadIdx.x == 0)
          {
            gbuf.ptr(0)[blockIdx.x] = smem_weight[0];
            gbuf.ptr(1)[blockIdx.x] = smem_weighted_res[0];	
          }
          
          return;			
        }
        
        __device__ __forceinline__ void
        directReduction() const
        {
          const float *beg = &error[0];  
          const float *end = beg + nel;  

          int tid = threadIdx.x;
          __shared__ float smem_weighted_sq_res[CTA_SIZE_DR];
          __shared__ float smem_weighted_res[CTA_SIZE_DR];
          __shared__ float smem_weights[CTA_SIZE_DR];
          __shared__ float smem_nel[CTA_SIZE_DR];
          __shared__ float smem_sigma[2];
          __shared__ float smem_bias[2];
          __syncthreads ();
          
          if (tid==0)
          {
            smem_bias[0] = 0;
            smem_sigma[0] = 99999999.f;
          }
          __syncthreads ();
          
          int max_iters = 10;
          
          for (int i=0; i < max_iters; i++)
          {
            float sum_weighted_sq_res = 0.f;
            float sum_weighted_res = 0.f;
            float sum_weights = 0.f;
            float sum_nel = 0.f;

            for (const float *t = beg + tid; t < end; t += STRIDE) 
            { 
              float is_valid = 0.f;
              float weighted_sq_res = 0.f; 
              float weighted_res = 0.f; 
              float weight = 0.f; 
              float error_raw = *t;

              if ( !(isinf(error_raw)) && !(isnan(error_raw)))
              {		          
                weight = 1.f;
                is_valid = 1.f;

                if (!(i==0))
                {
                  float error_unbias = error_raw - smem_bias[0];
                  float error_normalised = error_unbias/smem_sigma[0];

                  if ( (Mestimator == HUBER) && (fabs(error_normalised) > THRESHOLD_HUBER_DEV))
                  {
                    weight = (THRESHOLD_HUBER_DEV / fabs(error_normalised));
                  }
                  else if (Mestimator == TUKEY)
                  {
                    if (fabs(error_normalised) < THRESHOLD_TUKEY_DEV)
                    {
                      float aux1 = ( error_normalised / THRESHOLD_TUKEY_DEV ) * ( error_normalised / THRESHOLD_TUKEY_DEV );
                      weight = (1.f - aux1) * (1.f - aux1);		
                    }
                    else
                    {
                    weight = 0.f;	

                    //not what theory says, but empirically the performance is better. 
                    /*Maybe it is related with the fact that if one allows Tukey estimator to be defined in (-infty, infty) 
                    pdf's normalizing const = infty so that integral(pdf) = 1, and loglikelihood = infty for every sigma. 
                    Hence the need of cutting tails  of Tukey's associated "pdf" to have a real pdf*/
                    is_valid = 0.f; 

                    }
                  }
                  else if (Mestimator == STUDENT)
                  {
                    weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + error_normalised*error_normalised);             
                  }
                } //else (i==0) weight = 1->first iter is like asuming LeastSquares estimator

                weighted_res = error_raw*weight;
                weighted_sq_res = weighted_res*error_raw;
              }

              sum_weighted_sq_res += weighted_sq_res;
              sum_weighted_res += weighted_res;
              sum_weights += weight;
              sum_nel += is_valid; 
            }

            __syncthreads ();
            smem_weighted_sq_res[tid] = sum_weighted_sq_res;
            smem_weighted_res[tid] = sum_weighted_res;
            smem_weights[tid] = sum_weights;
            smem_nel[tid] = sum_nel;
            __syncthreads ();

            reduce<CTA_SIZE>(smem_weighted_sq_res);
            __syncthreads ();
            reduce<CTA_SIZE>(smem_weighted_res);
            __syncthreads ();
            reduce<CTA_SIZE>(smem_weights);
            __syncthreads ();
            reduce<CTA_SIZE>(smem_nel);
            __syncthreads ();

            if (tid == 0)
            {
              smem_bias[1] = smem_weighted_res[0] / smem_weights[0];
              smem_sigma[1] = sqrt((smem_weighted_sq_res[0] - 2.f*smem_bias[1]*smem_weighted_res[0] + smem_bias[1]*smem_bias[1]*smem_weights[0]) / smem_nel[0]);
            }
            __syncthreads ();

            if ((i>0) && (abs(smem_sigma[1]-smem_sigma[0])/smem_sigma[0]) < 0.1)
            {		 
              break;
            }

            if (tid == 0)
            {
              smem_sigma[0] = smem_sigma[1];
              smem_bias[0] = smem_bias[1];
            }		
            __syncthreads ();       	
          }
		      
		      if (tid == 0)
  				{
	          moments_dev[0] = smem_bias[1];
	          moments_dev[1] = smem_sigma[1]; 
	        }
				        
		      return;
		      
        }        
      };
      
      
      struct chiSquaredHandler
      {
        enum
        {
          CTA_SIZE = 256
        };

        float* error;

        int Mestimator;
        float sigma;
        int nel;

        mutable PtrStep<float> gbuf;

        __device__ __forceinline__ void
        partialChiSquared () const
        {

          int x = threadIdx.x + blockIdx.x * CTA_SIZE;

          __shared__ float smem_N[CTA_SIZE];
          __shared__ float smem_sum[CTA_SIZE];
          
          float is_valid = 0.f;
          float rho = 0.f;

          if (x < nel)
          {
            if ( !(isinf(error[x])) && !(isnan(error[x])))
            {
              float error_normalised = error[x]/sigma;
              is_valid = 1.f;
              rho = (error_normalised*error_normalised)/2.f;

              if ( (Mestimator == HUBER) && (fabs(error_normalised) > THRESHOLD_HUBER_DEV))
              {                
                rho = THRESHOLD_HUBER_DEV*(fabs(error_normalised) - THRESHOLD_HUBER_DEV/2.f);
              }
              else if (Mestimator == TUKEY)
              {
                if (fabs(error_normalised) < THRESHOLD_TUKEY_DEV)
                {
                  float aux1 = ( error_normalised / THRESHOLD_TUKEY_DEV ) * ( error_normalised / THRESHOLD_TUKEY_DEV );
                  float aux2 = (1.f-aux1)*(1.f-aux1)*(1.f-aux1);
                  rho = ((THRESHOLD_TUKEY_DEV*THRESHOLD_TUKEY_DEV)/6.f)*(1.f-aux2);
                }
                else
                {
                  rho = ((THRESHOLD_TUKEY_DEV*THRESHOLD_TUKEY_DEV)/6.f);
                }
              }
              else if (Mestimator == STUDENT)
              {
                rho = ((STUDENT_DOF_DEV + 1.f) /  2.f)*log(1.f + (error_normalised*error_normalised)/STUDENT_DOF_DEV);             
              }  
            }
          }

          __syncthreads ();
          smem_N[threadIdx.x] = is_valid;
          smem_sum[threadIdx.x] = rho;     
          __syncthreads ();

          reduce<CTA_SIZE>(smem_N);	__syncthreads ();
          reduce<CTA_SIZE>(smem_sum);	__syncthreads ();					

          if (threadIdx.x == 0)
          {
            gbuf.ptr(0)[blockIdx.x] = smem_N[0];
            gbuf.ptr(1)[blockIdx.x] = smem_sum[0];	
          }
          
          return;			
        }
      };

		
  	
      struct finalReductionHandlerSigma
      {
        enum
        {
          CTA_SIZE = 512,
          STRIDE = CTA_SIZE,
        };

        PtrStep<float> gbuf;
        int length;
        mutable float* output;

        __device__ __forceinline__ void
        operator () () const
        {
          const float *beg = gbuf.ptr (blockIdx.x);  
          const float *end = beg + length;  

          int tid = threadIdx.x;

          float sum = 0.f;
          
          for (const float *t = beg + tid; t < end; t += STRIDE)  
            sum += *t;

          __shared__ float smem[CTA_SIZE];

          __syncthreads ();
          smem[tid] = sum;
          __syncthreads ();

          reduce<CTA_SIZE>(smem);

          __syncthreads ();
          
          if (tid == 0)
            output[blockIdx.x] = smem[0];
           
          return;
        }        
      };
  		
      __global__ void
      errorKernel (const errorHandler eh) 
      {
        eh.computeError ();
      }
      

      __global__ void
      finalReductionKernel (const finalReductionHandlerSigma frh) 
      {
        frh ();
      }

      __global__ void
      computeSigmaKernelPartial (const sigmaHandler sh) 
      {
        sh.partialSigma ();
      }
      
      __global__ void
      computeBiasKernelPartial (const sigmaHandler sh) 
      {
        sh.partialBias ();
      }
      
      __global__ void
      computeDirectReductionKernel (const sigmaHandler sh) 
      {
        sh.directReduction ();
      }
      
      __global__ void
      chiSquaredKernelPartial (const chiSquaredHandler csh) 
      {
        csh.partialChiSquared ();
      }
  
  
      __global__ void
      centerAndAbsKernel (float* error, float medianAux, int size)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x >= size)
          return;

        float value = error[x] - medianAux;
        //we change -inf to 0.f due to new [0,inf) domain after fabs()
        if (isinf(value) && value < 0.f) 
          value = 0.f;

        float value_abs = fabs(value);
        error[x] = value_abs;
        return;
      }
      
      __global__ void
      errorSamplingKernel (float* error_src, float* error_dst, int stride, int src_size, int dst_size)
      {
        int dst_id = threadIdx.x + blockIdx.x * blockDim.x;
        int src_id = stride*(threadIdx.x + blockIdx.x * blockDim.x);
        
        if ((dst_id >= dst_size) || (src_id  >= src_size))
          return;

        src_id = max(0, min(src_id, src_size-1));					
        float value = error_src[src_id];	
        
        error_dst[dst_id] =  value;
        
        return;					    
      }		
        
      float
      computeMedian (DeviceArray<float>& error)
      {
        int size_error = error.size();
        float median_error[2];

        thrust::device_ptr<float> dev_error_ptr(error.ptr());
        thrust::sort(dev_error_ptr, dev_error_ptr + size_error);		

        cudaMemcpy(median_error, error.ptr() + (size_error / 2), 2*sizeof(float), cudaMemcpyDeviceToHost);  

        return (size_error & 1) ? median_error[0] : 0.5f*(median_error[0] + median_error[1]);
      };

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

      float 
      computeError (const DeviceArray2D<float>& im1,  const DeviceArray2D<float>& im0,  DeviceArray<float>& error)
      {
        ////////////////////////////////////////////	
        cudaEvent_t start_total, stop_total;	////
        float elapsedTime_total;		////
        cudaEventCreate(&start_total);		////
        cudaEventRecord(start_total,0);		////
        ////////////////////////////////////////////
        
        errorHandler eh;

        eh.im0 = im0;
        eh.im1 = im1;

        eh.cols = im0.cols();
        eh.rows = im0.rows();

        if (error.size() != eh.rows*eh.cols)
          error.create(eh.rows*eh.cols);

        eh.error = error;

        dim3 block (errorHandler::CTA_SIZE_X, errorHandler::CTA_SIZE_Y);
        dim3 grid (divUp (eh.cols, block.x), divUp (eh.rows, block.y));

        errorKernel<<<grid,block>>>(eh);
        cudaSafeCall ( cudaGetLastError () );

        //////////////////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&stop_total);							//////
        cudaEventRecord(stop_total,0);							//////
        cudaEventSynchronize(stop_total);						//////
        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////	
        //////////////////////////////////////////////////////////////////////////////////////
      
        return elapsedTime_total;
      };
		
				
      float  computeSigmaMAD (DeviceArray<float>& error, float &sigma)
      {
        ////////////////////////////////////////////	
        cudaEvent_t start_total, stop_total;	////
        float elapsedTime_total;		////
        cudaEventCreate(&start_total);		////
        cudaEventRecord(start_total,0);		////
        ////////////////////////////////////////////
        
        int Nsamples = 1000000000;
        int stride;
        
        if (error.size() < Nsamples)
        {
          Nsamples = error.size();
          stride = 1;
        }
        else
        {
          stride = error.size() / Nsamples;
          Nsamples  = error.size() / stride;
        }
        
        DeviceArray<float> error_sampled;
        error_sampled.create(Nsamples);
        
        int block1D = 256;
        int grid1D (divUp (Nsamples,block1D));
        
        cudaFuncSetCacheConfig (errorSamplingKernel, cudaFuncCachePreferShared);
        errorSamplingKernel<<<grid1D,block1D>>> (error, error_sampled, stride, error.size(), error_sampled.size());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());        

        float medianAux_error, median_error;
        medianAux_error = computeMedian(error_sampled);

        centerAndAbsKernel<<<grid1D,block1D>>>(error_sampled, medianAux_error, error_sampled.size());
        cudaSafeCall ( cudaGetLastError () );

        median_error = computeMedian(error_sampled);
        sigma = 1.4286f*median_error;

        //////////////////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&stop_total);							//////
        cudaEventRecord(stop_total,0);							//////
        cudaEventSynchronize(stop_total);						//////
        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////
        //////////////////////////////////////////////////////////////////////////////////////
        
        return elapsedTime_total;
      };

      
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
      float  
      computeSigmaPdf (DeviceArray<float>& error, float &sigma, int Mestimator)
      {        
        ////////////////////////////////////////////	
        cudaEvent_t start_total, stop_total;	////
        float elapsedTime_total;		////
        cudaEventCreate(&start_total);		////
        cudaEventRecord(start_total,0);		////
        ////////////////////////////////////////////

        sigmaHandler sh;

        sh.error = error;

        sh.nel = error.size();	
        sh.Mestimator = LSQ; //first estimate assuming gaussian error...
        //sh.Mestimator = Mestimator; //...or start directly with our chosen estimator (conv. guaranteed for high init guess??)
        sh.sigma = sigma;
        sh.bias = 0.f;

        int block1D = sigmaHandler::CTA_SIZE;
        int grid1D (divUp (error.size(),block1D));
        
        DeviceArray2D<float> gbuf;
        gbuf.create (2,grid1D);
        sh.gbuf = gbuf; 

        DeviceArray<float> mbuf;
        mbuf.create(2);
        
        finalReductionHandlerSigma frh;
        frh.gbuf = gbuf;
        frh.length = grid1D;
        frh.output = mbuf;

        int max_iters = 10;
        int iters_count = 0;
        //std::cout << std::endl;

        for (int i=0; i < max_iters; i++)
        {
          iters_count++;
          float rel_tol = 0.01f;
          
          cudaFuncSetCacheConfig (computeSigmaKernelPartial, cudaFuncCachePreferShared);
          computeSigmaKernelPartial<<<grid1D,block1D>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaDeviceSynchronize());

          frh.gbuf = gbuf;
          finalReductionKernel<<<2,finalReductionHandlerSigma::CTA_SIZE>>>(frh);
          cudaSafeCall (cudaGetLastError ());			
          cudaSafeCall (cudaDeviceSynchronize ());
          
          float host_data[2];
          mbuf.download (host_data);
          float Nel = host_data[0];
          float sum_squares = host_data[1];

          sigma = sqrt(sum_squares/Nel);	

          if ((i > 0) && ((abs(sigma - sh.sigma)/sh.sigma) < rel_tol))
            break;

          sh.sigma = sigma;
          sh.Mestimator = Mestimator;
        }

        gbuf.release();
        mbuf.release();

        //////////////////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&stop_total);							//////
        cudaEventRecord(stop_total,0);							//////
        cudaEventSynchronize(stop_total);						//////
        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////	
        //////////////////////////////////////////////////////////////////////////////////////

        return elapsedTime_total;

      };
      
      float  
      computeSigmaPdfWithBias (DeviceArray<float>& error, float &sigma, float &bias, int Mestimator)
      {        
        ////////////////////////////////////////////	
        cudaEvent_t start_total, stop_total;	////
        float elapsedTime_total;		////
        cudaEventCreate(&start_total);		////
        cudaEventRecord(start_total,0);		////
        ////////////////////////////////////////////
        
        int Nsamples = 100000000;
        int stride;
        
        if (error.size() < Nsamples)
        {
          Nsamples = error.size();
          stride = 1;
        }
        else
        {
          stride = error.size() / Nsamples;
        }
        
        DeviceArray<float> error_sampled;
        error_sampled.create(Nsamples);
				
        int block1D = sigmaHandler::CTA_SIZE;
        int grid1D (divUp (Nsamples,block1D));
        
        cudaFuncSetCacheConfig (errorSamplingKernel, cudaFuncCachePreferShared);
        errorSamplingKernel<<<grid1D,block1D>>> (error, error_sampled, stride, error.size(), error_sampled.size());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());
        
        sigmaHandler sh;

        sh.error = error_sampled;

        sh.nel = error_sampled.size();	
        
        //first estimate assuming gaussian error...
        sh.Mestimator = LSQ; 
        
        //...or start directly with our chosen estimator
        //sh.Mestimator = Mestimator;  
        //setting init bias and sigma (conv. guaranteed for high init guess??)
        sh.sigma = sigma;
        sh.bias = bias;
        
        
        DeviceArray2D<float> gbuf;
        gbuf.create (2,grid1D);
        sh.gbuf = gbuf; 

        DeviceArray<float> mbuf;
        mbuf.create(2);
        
        finalReductionHandlerSigma frh;
        frh.gbuf = gbuf;
        frh.length = grid1D;
        frh.output = mbuf;

        int max_iters = 10;
        int iters_count = 0;
        //std::cout << std::endl;

        for (int i=0; i < max_iters; i++)
        {
          iters_count++;
          float rel_tol = 0.01f;
          float host_data[2];
          
          cudaFuncSetCacheConfig (computeBiasKernelPartial, cudaFuncCachePreferShared);
          computeBiasKernelPartial<<<grid1D,block1D>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaDeviceSynchronize());
          
          frh.gbuf = gbuf;
          finalReductionKernel<<<2,finalReductionHandlerSigma::CTA_SIZE>>>(frh);
          cudaSafeCall (cudaGetLastError ());			
          cudaSafeCall (cudaDeviceSynchronize ());
            
          mbuf.download (host_data);
          float sum_weighted_res = host_data[1];
          float sum_weights = host_data[0];
          sh.bias =     sum_weighted_res / sum_weights;          
               
          cudaFuncSetCacheConfig (computeSigmaKernelPartial, cudaFuncCachePreferShared);
          computeSigmaKernelPartial<<<grid1D,block1D>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaDeviceSynchronize());

          frh.gbuf = gbuf;
          finalReductionKernel<<<2,finalReductionHandlerSigma::CTA_SIZE>>>(frh);
          cudaSafeCall (cudaGetLastError ());			
          cudaSafeCall (cudaDeviceSynchronize ());
          
          mbuf.download (host_data);
          float Nel = host_data[0];
          float sum_squares = host_data[1];

          sigma = sqrt(sum_squares/Nel);	
          bias = sh.bias;
					
          if ((i > 0) && ((abs(sigma - sh.sigma)/sh.sigma) < rel_tol))
            break;

          sh.sigma = sigma;
          sh.Mestimator = Mestimator;
        }

        gbuf.release();
        mbuf.release();

        //////////////////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&stop_total);							//////
        cudaEventRecord(stop_total,0);							//////
        cudaEventSynchronize(stop_total);						//////
        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////	
        //////////////////////////////////////////////////////////////////////////////////////

        return elapsedTime_total;
      };
      
      
      float  
      computeSigmaPdfWithBiasDirectReduction (DeviceArray<float>& error, float &sigma, float &bias, int Mestimator)
      {        
        ////////////////////////////////////////////	
        cudaEvent_t start_total, stop_total;	////
        float elapsedTime_total;		////
        cudaEventCreate(&start_total);		////
        cudaEventRecord(start_total,0);		////
        ////////////////////////////////////////////
        
        int Nsamples = 10000;
        int stride;
        
        if (error.size() < Nsamples)
        {
          Nsamples = error.size();
          stride = 1;
        }
        else
        {
          stride = error.size() / Nsamples;
        }
        
        DeviceArray<float> error_sampled;
        error_sampled.create(Nsamples);
				
        int block1D = sigmaHandler::CTA_SIZE;
        int grid1D (divUp (Nsamples,block1D));
        
        cudaFuncSetCacheConfig (errorSamplingKernel, cudaFuncCachePreferShared);
        errorSamplingKernel<<<grid1D,block1D>>> (error, error_sampled, stride, error.size(), error_sampled.size());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());
        
        sigmaHandler sh;

        sh.error = error_sampled;

        sh.nel = error_sampled.size();	
        
        //first estimate assuming gaussian error...
        sh.Mestimator = Mestimator; 
        
        //...or start directly with our chosen estimator
        //sh.Mestimator = Mestimator;  
        //setting init bias and sigma (conv. guaranteed for high sigma init guess??)
        sh.sigma = sigma;
        sh.bias = bias;
        
        DeviceArray<float> moments_dev;
        moments_dev.create(2);  
        sh.moments_dev = moments_dev;    
        
        cudaFuncSetCacheConfig (computeDirectReductionKernel, cudaFuncCachePreferShared);
        computeDirectReductionKernel<<<grid1D,block1D>>>(sh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());
        
        float moments_host[2];
        
        moments_dev.download(moments_host);
        
        bias = moments_host[0];
        sigma = moments_host[1];
          
        
        moments_dev.release();
        error_sampled.release();

        //////////////////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&stop_total);							//////
        cudaEventRecord(stop_total,0);							//////
        cudaEventSynchronize(stop_total);						//////
        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////	
        //////////////////////////////////////////////////////////////////////////////////////

        return elapsedTime_total;
      };
      
      
      float 
      computeChiSquare (DeviceArray<float>& error_int,  DeviceArray<float>& error_depth,  float sigma_int, float sigma_depth, int Mestimator, float &chi_square, float &chi_test, float &Ndof)
      {
        ////////////////////////////////////////////	
        cudaEvent_t start_total, stop_total;	////
        float elapsedTime_total;		////
        cudaEventCreate(&start_total);		////
        cudaEventRecord(start_total,0);		////
        ////////////////////////////////////////////
      
        chiSquaredHandler csh;
        
        int block1D = chiSquaredHandler::CTA_SIZE;
        int grid1D (divUp (error_depth.size(),block1D));
        
        DeviceArray2D<float> gbuf;
        gbuf.create (2,grid1D);
        csh.gbuf = gbuf; 

        DeviceArray<float> mbuf;
        mbuf.create(2);
        
        float host_data[2];
        float Nint, Ndepth;
        float sum_chi_int, sum_chi_depth;
        
        finalReductionHandlerSigma frh;
        frh.gbuf = gbuf;
        frh.length = grid1D;
        frh.output = mbuf;        
        
        csh.error = error_int;

        csh.nel = error_int.size();	
        csh.Mestimator = Mestimator; 
        csh.sigma = sigma_int;
          
        cudaFuncSetCacheConfig (chiSquaredKernelPartial, cudaFuncCachePreferShared);
        chiSquaredKernelPartial<<<grid1D,block1D>>>(csh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());

        frh.gbuf = gbuf;
        finalReductionKernel<<<2,finalReductionHandlerSigma::CTA_SIZE>>>(frh);
        cudaSafeCall (cudaGetLastError ());			
        cudaSafeCall (cudaDeviceSynchronize ());
        
        
        mbuf.download (host_data);
        Nint = host_data[0];
        sum_chi_int = host_data[1];        
        

        csh.error = error_depth;

        csh.nel = error_depth.size();	
        csh.Mestimator = Mestimator; 
        csh.sigma = sigma_depth;
          
        cudaFuncSetCacheConfig (chiSquaredKernelPartial, cudaFuncCachePreferShared);
        chiSquaredKernelPartial<<<grid1D,block1D>>>(csh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaDeviceSynchronize());

        frh.gbuf = gbuf;
        finalReductionKernel<<<2,finalReductionHandlerSigma::CTA_SIZE>>>(frh);
        cudaSafeCall (cudaGetLastError ());			
        cudaSafeCall (cudaDeviceSynchronize ());        
        
        mbuf.download (host_data);
        Ndepth = host_data[0];
        sum_chi_depth = host_data[1];
        
        chi_square = sum_chi_int + sum_chi_depth;
        Ndof = Nint + Ndepth;
        
        float z_gauss = (chi_square - Ndof ) / (sqrt(2.f*Ndof));
        chi_test =  0.5f*(1.f + erf(z_gauss / sqrt(2.f)));

        gbuf.release();
        mbuf.release();

        //////////////////////////////////////////////////////////////////////////////////////
        cudaEventCreate(&stop_total);							//////
        cudaEventRecord(stop_total,0);							//////
        cudaEventSynchronize(stop_total);						//////
        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////	
        //////////////////////////////////////////////////////////////////////////////////////

        return elapsedTime_total;        
      }
//      float 
//      computeChiSquare (const DeviceArray<float>& error_int,  const DeviceArray<float>& error_depth,  float sigma_int, float sigma_depth, float &xiScore)
//      {
//        ////////////////////////////////////////////	
//        cudaEvent_t start_total, stop_total;	////
//        float elapsedTime_total;		////
//        cudaEventCreate(&start_total);		////
//        cudaEventRecord(start_total,0);		////
//        ////////////////////////////////////////////
//        
//        error_size = error_int.size();
//        DeviceArray2D<float> gbuf;
//        gbuf.create (2,grid1D);

//        int block1D  = 256;
//        dim3 grid1D = divUp (error_size, block1D);

//        chiSquareKernel<<<grid1D,block1D>>>(error_size, error_int, gbuf, error_depth, sigma_int, sigma_depth);
//        cudaSafeCall ( cudaGetLastError () );
//        
//        DeviceArray<float> mbuf;
//        mbuf.create(2);
//        
//        finalReductionHandlerSigma frh;
//        frh.gbuf = gbuf;
//        frh.length = grid1D;
//        frh.output = mbuf;        
//        frh.gbuf = gbuf;
//        
//        finalReductionKernel<<<2,finalReductionHandlerSigma::CTA_SIZE>>>(frh);
//        cudaSafeCall (cudaGetLastError ());			
//        cudaSafeCall (cudaDeviceSynchronize ());

//        //////////////////////////////////////////////////////////////////////////////////////
//        cudaEventCreate(&stop_total);							//////
//        cudaEventRecord(stop_total,0);							//////
//        cudaEventSynchronize(stop_total);						//////
//        cudaEventElapsedTime(&elapsedTime_total, start_total,stop_total);		//////	
//        //////////////////////////////////////////////////////////////////////////////////////
//      
//        return elapsedTime_total;
//      };
      
    }
  }
}

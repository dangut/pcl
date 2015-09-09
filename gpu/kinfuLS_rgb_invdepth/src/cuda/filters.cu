/*
 * Software License Agreement (BSD License)
 *
 *  Shared Memory version of the bilateral filter gpu implementation of the PCL libraries.
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

namespace pcl
{
  namespace device
  {
    namespace kinfuRGBD
    {
                  
      namespace filter
      {
        enum
        {
          BLOCK_X = 32,
          BLOCK_Y = 8,
          RADIUS = 6,

          DIAMETER = RADIUS * 2 + 1,
          TILE_X = ( 2 * BLOCK_X + 2 * RADIUS ),
          TILE_Y = ( 2 * BLOCK_Y + 2 * RADIUS ),
          BLOCK_SIZE = BLOCK_X * BLOCK_Y,	
          SHARED_SIZE = TILE_X * TILE_Y,
          STRIDE_SHARED = (SHARED_SIZE - 1) / BLOCK_SIZE + 1, //the final (+1) is for ceiling
          AREA = DIAMETER*DIAMETER
        };
      }

      __constant__ float sigma_space = 5.f;     

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      bilateralKernel (const PtrStepSz<float> src, 
                      PtrStep<float> dst, 
                      const float sigma_floatmap)
      {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= src.cols || y >= src.rows)
          return;

//        const int R = 6;       //static_cast<int>(sigma_space * 1.5);
//        const int D = R * 2 + 1;

        float value = src.ptr (y)[x];
        
        if (isnan(value))
        {
          dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
          return;
        }

        int tx = min (x + filter::RADIUS + 1, src.cols);
        int ty = min (y + filter::RADIUS + 1, src.rows);

        float sum1 = 0;
        float sum2 = 0;

        for (int cy = max (y - filter::RADIUS, 0); cy < ty; ++cy)
        {
          for (int cx = max (x - filter::RADIUS, 0); cx < tx; ++cx)
          {
            float tmp = src.ptr (cy)[cx];
            
            if (!isnan(tmp))
            {
              float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
              float floatval_normalised = (value - tmp) / sigma_floatmap;              
              float sigma2_space_inv_half = 0.5 / (sigma_space*sigma_space);
              float weight = __expf (-(sigma2_space_inv_half*space2 + 0.5*floatval_normalised*floatval_normalised  ));
              sum1 += tmp * weight;
              sum2 += weight;
            }            
          }
        }

        float res =  (sum1 / sum2);
        dst.ptr (y)[x] = res;
      }

      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      float
      bilateralFilter (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst, const float sigma_floatmap)
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////        
        
        dim3 block (32, 8);
        dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));
        
        //float inv_sigma_floatmap_half = 0.5f / (sigma_floatmap * sigma_floatmap);

        cudaFuncSetCacheConfig (bilateralKernel, cudaFuncCachePreferL1);
        bilateralKernel<<<grid, block>>>(src, dst, sigma_floatmap);

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

    }
  }
}
      

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
                  
      namespace pyr
      {
        enum
        {
          BLOCK_X = 32,
          BLOCK_Y = 8,
          RADIUS = 2,

          DIAMETER = RADIUS * 2 + 1,
          TILE_X = ( 2 * BLOCK_X + 2 * RADIUS ),
          TILE_Y = ( 2 * BLOCK_Y + 2 * RADIUS ),
          BLOCK_SIZE = BLOCK_X * BLOCK_Y,
          SHARED_SIZE = TILE_X * TILE_Y,
          STRIDE_SHARED = (SHARED_SIZE - 1) / BLOCK_SIZE + 1, //the final (+1) is for ceiling
          AREA = DIAMETER*DIAMETER
        };
      }

      __constant__ float sigma = 1.f;      

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      pyrDownKernelf (const PtrStepSz<float> src, PtrStepSz<float> dst)
      {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst.cols || y >= dst.rows)
          return;        

        float res = numeric_limits<float>::quiet_NaN ();

        int tx = min (2 * x + pyr::RADIUS + 1, src.cols);
        int ty = min (2 * y + pyr::RADIUS + 1, src.rows);

        float sum1 = 0.f;
        float sum2 = 0.f;
        int count = 0;

        float sigma_space2_inv_half = 0.5f / (sigma *sigma);

        for (int cy = max (0, 2 * y - pyr::RADIUS); cy < ty; ++cy)
        {
          for (int cx = max (0, 2 * x - pyr::RADIUS); cx < tx; ++cx)
          {
            float val = src.ptr (cy)[cx];
            if (!isnan(val))
            {
              float space2 = (2 * x - cx) * (2 * x - cx) + (2 * y - cy) * (2 * y - cy);
              float weight = __expf (-(space2 * sigma_space2_inv_half));
              //float weight = __expf (-(space2 * 0.5f));
              sum1 += val*weight;
              sum2 += weight;
              ++count;
            }
          }
        }

        if (count > (pyr::AREA / 2))  //if more than half of windowed pixels on lower pyr are OK, we downsample.
          res = sum1 / sum2; 

        dst.ptr (y)[x] = res;
      }
      
      //Both int and depth are float maps. Same function could be used, but prefer to separate them for clarity and maybe use different kernels in future.
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      float
      pyrDownDepth (const DepthMapf& src, DepthMapf& dst)
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////
        
        dst.create (src.rows () / 2, src.cols () / 2);

        dim3 block (pyr::BLOCK_X, pyr::BLOCK_Y);
        dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));
        
        pyrDownKernelf<<<grid, block>>>(src, dst);
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
      pyrDownIntensity (const IntensityMapf& src, IntensityMapf& dst)
      {
        /////////////////////////////////////////
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        /////////////////////////////////////////////
        
        dst.create (src.rows () / 2, src.cols () / 2);

        dim3 block (pyr::BLOCK_X, pyr::BLOCK_Y);
        dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

        pyrDownKernelf<<<grid, block>>>(src, dst);
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
      

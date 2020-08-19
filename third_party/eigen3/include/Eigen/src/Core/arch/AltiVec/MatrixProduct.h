// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_ALTIVEC_H

#ifdef __MMA__

namespace Eigen {

namespace internal {

const int accRows = 4;
const int accCols = 4;
const int accCount = 4;
const int floatVectorSize = 4;

typedef struct
{
  __vector float v0;
  __vector float v1;
  __vector float v2;
  __vector float v3;
} Packet4fx4;

union PacketQuad
{
  __struct_quad sc;
  Packet4fx4    sf;
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
    int ri = 0, j;
    for(j = 0; j + floatVectorSize < rows; j+=floatVectorSize)
    {
        int i;
        for(i = 0; i + floatVectorSize < depth; i+=floatVectorSize)
        {
            PacketBlock<Packet4f, 4> block;
            block.packet[0] = lhs.template loadPacket<Packet4f>(j, i + 0);
            block.packet[1] = lhs.template loadPacket<Packet4f>(j, i + 1);
            block.packet[2] = lhs.template loadPacket<Packet4f>(j, i + 2);
            block.packet[3] = lhs.template loadPacket<Packet4f>(j, i + 3);

            pstore<float>((float *)(blockA + ri     ), block.packet[0]);
            pstore<float>((float *)(blockA + ri +  4), block.packet[1]);
            pstore<float>((float *)(blockA + ri +  8), block.packet[2]);
            pstore<float>((float *)(blockA + ri + 12), block.packet[3]);
            ri += 4*floatVectorSize;
        }
        for(; i < depth; i++)
        {
            Packet4f lhsV = lhs.template loadPacket<Packet4f>(j, i);
            pstore<float>((float *)(blockA + ri), lhsV);
            ri += floatVectorSize;
        }
    }
    for(int i = 0; i < depth; i++)
    {
        int k = j;
        for(; k < rows; k++)
        {
            blockA[ri] = lhs(k, i);
            ri += 1;
        }
    }
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
    int ri = 0, j;
    for(j = 0; j + floatVectorSize < cols; j+=floatVectorSize)
    {
        int i;
        for(i = 0; i + floatVectorSize < depth; i+=floatVectorSize)
        {
            PacketBlock<Packet4f, 4> block;
            block.packet[0] = rhs.template loadPacket<Packet4f>(i, j + 0);
            block.packet[1] = rhs.template loadPacket<Packet4f>(i, j + 1);
            block.packet[2] = rhs.template loadPacket<Packet4f>(i, j + 2);
            block.packet[3] = rhs.template loadPacket<Packet4f>(i, j + 3);

            ptranspose(block);

            pstore<float>((float *)(blockB + ri     ), block.packet[0]);
            pstore<float>((float *)(blockB + ri +  4), block.packet[1]);
            pstore<float>((float *)(blockB + ri +  8), block.packet[2]);
            pstore<float>((float *)(blockB + ri + 12), block.packet[3]);

            ri += 4*floatVectorSize;
        }
        for(; i < depth; i++)
        {
            blockB[ri+0] = rhs(i, j+0);
            blockB[ri+1] = rhs(i, j+1);
            blockB[ri+2] = rhs(i, j+2);
            blockB[ri+3] = rhs(i, j+3);
            ri += floatVectorSize;
        }
    }
    for(int i = 0; i < depth; i++)
    {
        int k = j;
        for(; k < cols; k++)
        {
            blockB[ri] = rhs(i, k);
            ri += 1;
        }
    }
}

template<typename DataMapper, typename Index, typename Scalar>
EIGEN_STRONG_INLINE void storeAccumulator(Index i, Index j, const DataMapper& data, Scalar alpha, __vector_quad *acc)
{
  //[TODO]
  //
  //Packet4fx4 r;
  //
  //__builtin_mma_disassemble_acc((void *)&r, *acc);
  //
  PacketQuad result;
  result.sc  =  __builtin_mma_disassemble_acc(*acc);

  Packet4f pAlpha = pset1<Packet4f>(alpha);

  PacketBlock<Packet4f, 4> block;
  block.packet[0] = pAlpha*result.sf.v3;
  block.packet[1] = pAlpha*result.sf.v2;
  block.packet[2] = pAlpha*result.sf.v1;
  block.packet[3] = pAlpha*result.sf.v0;

  data.template storePacketBlock<Packet4f, 4>(i, j, block);
}

template<typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, RhsScalar, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  void operator()(const DataMapper& res, const float* blockA, const RhsScalar* blockB,
                  Index rows, Index depth, Index cols, float alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, RhsScalar, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const float* blockA, const RhsScalar* blockB,
               Index rows, Index depth, Index cols, float alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
      const int remaining_rows = rows % accRows;
      const int remaining_cols = cols % accCols;
      const int remaining_depth = depth % floatVectorSize;
      const int timesRows = (rows / accRows);
      const int timesCols = (cols / accCols);

      int row;
      for(row = 0; row + accRows <= rows; row += accRows)
      {
          const float *rhs_base = blockB;
          const float *lhs_base = blockA + (row/accRows)*depth*floatVectorSize;

          int col;
          for(col = 0; col + accCount*accCols <= cols; col += accCount*accCols){
              const float *rhs_ptr  = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *rhs_ptr2 = rhs_base + ((col/accCols) + 1)*depth*floatVectorSize;
              const float *rhs_ptr3 = rhs_base + ((col/accCols) + 2)*depth*floatVectorSize;
              const float *rhs_ptr4 = rhs_base + ((col/accCols) + 3)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
             
              __vector_quad acc, acc2, acc3, acc4;
              __builtin_mma_xxsetaccz(&acc);
              __builtin_mma_xxsetaccz(&acc2);
              __builtin_mma_xxsetaccz(&acc3);
              __builtin_mma_xxsetaccz(&acc4);

              for(int k = 0; k < depth; k++)
              {
                  __vector float lhsV  = *((__vector float *)lhs_ptr );
                  __vector float rhsV  = *((__vector float *)rhs_ptr );
                  __vector float rhs2V = *((__vector float *)rhs_ptr2);
                  __vector float rhs3V = *((__vector float *)rhs_ptr3);
                  __vector float rhs4V = *((__vector float *)rhs_ptr4);
                  
                  __builtin_mma_xvf32gerpp(&acc, (__vector unsigned char) rhsV, (__vector unsigned char) lhsV);
                  __builtin_mma_xvf32gerpp(&acc2, (__vector unsigned char) rhs2V, (__vector unsigned char) lhsV);
                  __builtin_mma_xvf32gerpp(&acc3, (__vector unsigned char) rhs3V, (__vector unsigned char) lhsV);
                  __builtin_mma_xvf32gerpp(&acc4, (__vector unsigned char) rhs4V, (__vector unsigned char) lhsV);

                  lhs_ptr += floatVectorSize;
                  rhs_ptr += floatVectorSize;
                  rhs_ptr2 += floatVectorSize;
                  rhs_ptr3 += floatVectorSize;
                  rhs_ptr4 += floatVectorSize;
              }

              storeAccumulator<DataMapper, Index, float>(row, col            , res, alpha, &acc );
              storeAccumulator<DataMapper, Index, float>(row, col + 1*accCols, res, alpha, &acc2);
              storeAccumulator<DataMapper, Index, float>(row, col + 2*accCols, res, alpha, &acc3);
              storeAccumulator<DataMapper, Index, float>(row, col + 3*accCols, res, alpha, &acc4);
          }
          for(; col + accCols <= cols; col += accCols){
              const float *rhs_ptr  = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
             
              __vector_quad acc;
              __builtin_mma_xxsetaccz(&acc);
              for(int k = 0; k < depth; k++)
              {
                  __vector float lhsV = *((__vector float *)lhs_ptr);
                  __vector float rhsV = *((__vector float *)rhs_ptr);
                  
                  __builtin_mma_xvf32gerpp(&acc, (__vector unsigned char) rhsV, (__vector unsigned char) lhsV);
                 
                  lhs_ptr += floatVectorSize;
                  rhs_ptr += floatVectorSize;
              }

              storeAccumulator<DataMapper, Index, float>(row, col, res, alpha, &acc);
          }
        
          if(remaining_cols > 0)
          {
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
              for(int k = 0; k < depth; k++)
              {
                 for(int arow = 0; arow < accRows; arow++)
                 {
                     for(int acol = 0; acol < remaining_cols; acol++ )
                     {
                        res(row + arow, col + acol) += lhs_ptr[arow]*rhs_ptr[acol];
                     }
                 }
                 rhs_ptr += remaining_cols;
                 lhs_ptr += floatVectorSize;
              }
          }
      }

      if(remaining_rows > 0)
      {
          const float *rhs_base = blockB;
          const float *lhs_base = blockA + (row/accRows)*depth*floatVectorSize;

          int col;
          for(col = 0; col + accCols <= cols; col += accCols)
          {
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
              for(int k = 0; k < depth; k++)
              {
                 for(int arow = 0; arow < remaining_rows; arow++)
                 {
                     for(int acol = 0; acol < accCols; acol++ )
                     {
                        res(row + arow, col + acol) += lhs_ptr[arow]*rhs_ptr[acol];
                     }
                 }
                 rhs_ptr += floatVectorSize;
                 lhs_ptr += remaining_rows;
              }
          }
         
          if(remaining_cols > 0)
          {
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
              for(int k = 0; k < depth; k++)
              {
                 for(int arow = 0; arow < remaining_rows; arow++)
                 {
                     for(int acol = 0; acol < remaining_cols; acol++ )
                     {
                        res(row + arow, col + acol) += lhs_ptr[arow]*rhs_ptr[acol];
                     }
                 }
                 rhs_ptr += remaining_cols;
                 lhs_ptr += remaining_rows;
              }
          }
      }
  }

} // end namespace internal

} // end namespace Eigen

#endif // __MMA__
#endif // EIGEN_MATRIX_PRODUCT_ALTIVEC_H

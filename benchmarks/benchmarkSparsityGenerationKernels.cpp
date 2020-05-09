/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

// Source includes
#include "benchmarkSparsityGenerationKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{

LVARRAY_HOST_DEVICE
INDEX_TYPE getNeighborNodes( INDEX_TYPE (& neighborNodes)[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ],
                             ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                             ArraySlice< INDEX_TYPE const, RAJA::PERM_I > const nodeElems )
{
  INDEX_TYPE numNeighbors = 0;
  for( INDEX_TYPE localElem = 0; localElem < nodeElems.size(); ++localElem )
  {
    INDEX_TYPE const elemID = nodeElems[ localElem ];
    for( INDEX_TYPE localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
    {
      neighborNodes[ numNeighbors++ ] = elemToNodeMap( elemID, localNode );
    }
  }

  return sortedArrayManipulation::makeSortedUnique( neighborNodes, neighborNodes + numNeighbors );
}


DISABLE_HD_WARNING
template< typename SPARSITY_TYPE >
LVARRAY_HOST_DEVICE
void insertEntriesForNode( SPARSITY_TYPE & sparsity,
                           INDEX_TYPE const nodeID,
                           ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                           ArraySlice< INDEX_TYPE const, RAJA::PERM_I > const nodeElems )
{
  INDEX_TYPE neighborNodes[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ];
  INDEX_TYPE const numNeighbors = getNeighborNodes( neighborNodes, elemToNodeMap, nodeElems );

  INDEX_TYPE dofNumbers[ MAX_COLUMNS_PER_ROW ];
  for( INDEX_TYPE i = 0; i < numNeighbors; ++i )
  {
    for( INDEX_TYPE dim = 0; dim < NDIM; ++dim )
    {
      dofNumbers[ NDIM * i + dim ] = NDIM * neighborNodes[ i ] + dim;
    }
  }

  for( int dim = 0; dim < NDIM; ++dim )
  {
    sparsity.insertNonZeros( NDIM * nodeID + dim, dofNumbers, dofNumbers + NDIM * numNeighbors );
  }
}


void SparsityGenerationNative::resize( INDEX_TYPE const initialCapacity )
{
  LVARRAY_MARK_FUNCTION_TAG( "resize" );
  m_sparsity = SparsityPattern< COLUMN_TYPE, INDEX_TYPE >( NDIM * m_numNodes, NDIM * m_numNodes, initialCapacity );
}

void SparsityGenerationNative::resizeExact()
{
  std::vector< INDEX_TYPE > nnzPerRow( 3 * m_numNodes );

  INDEX_TYPE neighborNodes[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ];
  for( INDEX_TYPE nodeID = 0; nodeID < m_numNodes; ++nodeID )
  {
    INDEX_TYPE const numNeighbors = getNeighborNodes( neighborNodes, m_elemToNodeMap, m_nodeToElemMap[ nodeID ] );
    for( int dim = 0; dim < NDIM; ++dim )
    {
      nnzPerRow[ NDIM * nodeID + dim ] = NDIM * numNeighbors;
    }
  }

  resizeFromNNZPerRow< serialPolicy >( nnzPerRow );
}

template< typename POLICY >
void SparsityGenerationNative::resizeFromNNZPerRow( std::vector< INDEX_TYPE > const & nnzPerRow )
{
  SparsityPattern< COLUMN_TYPE, INDEX_TYPE > newSparsity;
  newSparsity.resizeFromRowCapacities< POLICY >( NDIM * m_numNodes, NDIM * m_numNodes, nnzPerRow.data() );
  m_sparsity = std::move( newSparsity );
}

template< typename SPARSITY_TYPE >
void SparsityGenerationNative::generateElemLoop( SPARSITY_TYPE & sparsity,
                                                 ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap )
{
  COLUMN_TYPE dofNumbers[ NODES_PER_ELEM * NDIM ];
  for( INDEX_TYPE elemID = 0; elemID < elemToNodeMap.size( 0 ); ++elemID )
  {
    for( INDEX_TYPE localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
    {
      for( int dim = 0; dim < NDIM; ++dim )
      {
        dofNumbers[ NDIM * localNode + dim ] = NDIM * elemToNodeMap( elemID, localNode ) + dim;
      }
    }

    sortedArrayManipulation::makeSorted( &dofNumbers[ 0 ], &dofNumbers[ NODES_PER_ELEM * NDIM ] );

    for( int localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
    {
      for( int dim = 0; dim < NDIM; ++dim )
      {
        sparsity.insertNonZeros( dofNumbers[ NDIM * localNode + dim ], &dofNumbers[ 0 ], &dofNumbers[ NODES_PER_ELEM * NDIM ] );
      }
    }
  }
}

template< typename SPARSITY_TYPE >
void SparsityGenerationNative::generateNodeLoop( SPARSITY_TYPE & sparsity,
                                                 ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                                                 ArrayOfArraysView< INDEX_TYPE const, INDEX_TYPE const, true > const & nodeToElemMap )
{
  /// Iterate over all the nodes.
  for( INDEX_TYPE nodeID = 0; nodeID < nodeToElemMap.size(); ++nodeID )
  {
    insertEntriesForNode( sparsity, nodeID, elemToNodeMap, nodeToElemMap[ nodeID ] );
  }
}

template< typename POLICY >
void SparsityGenerationRAJA< POLICY >::
resizeExact()
{
  LVARRAY_MARK_FUNCTION_TAG( "resizeExact" );
  std::vector< INDEX_TYPE > nnzPerRow( 3 * m_numNodes );

  #if defined(USE_OPENMP)
  using RESIZE_POLICY = std::conditional_t< std::is_same_v< serialPolicy, POLICY >, serialPolicy, parallelHostPolicy >;
  #else
  using RESIZE_POLICY = serialPolicy;
  #endif

  ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap = m_elemToNodeMap.toViewConst();
  ArrayOfArraysView< INDEX_TYPE const, INDEX_TYPE const, true > const & nodeToElemMap = m_nodeToElemMap.toViewConst();
  forall< RESIZE_POLICY >( m_numNodes, [&nnzPerRow, elemToNodeMap, nodeToElemMap] ( INDEX_TYPE const nodeID )
  {
    INDEX_TYPE neighborNodes[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ];
    INDEX_TYPE const numNeighbors = getNeighborNodes( neighborNodes, elemToNodeMap, nodeToElemMap[ nodeID ] );
    for( int dim = 0; dim < NDIM; ++dim )
    {
      nnzPerRow[ NDIM * nodeID + dim ] = NDIM * numNeighbors;
    }
  } );

  resizeFromNNZPerRow< RESIZE_POLICY >( nnzPerRow );
}

// Note this shoule be protected but cuda won't let you put an extended lambda in a protected or private method.
template< typename POLICY >
void SparsityGenerationRAJA< POLICY >::
generateNodeLoop( SparsityPatternView< COLUMN_TYPE, INDEX_TYPE const > const & sparsity,
                  ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                  ArrayOfArraysView< INDEX_TYPE const, INDEX_TYPE const, true > const & nodeToElemMap,
                  ::benchmark::State & state )
{
  LVARRAY_MARK_FUNCTION_TAG( "generateNodeLoop" );

  // This isn't measured for the other benchmarks so it's not here either.
  if( state.iterations() )
  {
    state.PauseTiming();
  }

  sparsity.move( RAJAHelper< POLICY >::space );

  if( state.iterations() )
  {
    state.ResumeTiming();
  }

  forall< POLICY >( nodeToElemMap.size(), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const nodeID )
      {
        insertEntriesForNode( sparsity, nodeID, elemToNodeMap, nodeToElemMap[ nodeID ] );
      } );
}

template< typename POLICY >
void CRSMatrixAddToRow< POLICY >::
addKernel( CRSMatrixView< ENTRY_TYPE, COLUMN_TYPE const, INDEX_TYPE const > const & matrix,
           ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
           ArrayOfArraysView< INDEX_TYPE const, INDEX_TYPE const, true > const & nodeToElemMap )
{
  LVARRAY_MARK_FUNCTION_TAG( "addKernel" );

  forall< POLICY >( elemToNodeMap.size( 0 ), [matrix, elemToNodeMap, nodeToElemMap] LVARRAY_HOST_DEVICE ( INDEX_TYPE const elemID )
      {
        COLUMN_TYPE dofNumbers[ NODES_PER_ELEM * NDIM ];
        ENTRY_TYPE additions[ NODES_PER_ELEM * NDIM ][ NODES_PER_ELEM * NDIM ];
        for( INDEX_TYPE localNode0 = 0; localNode0 < NODES_PER_ELEM; ++localNode0 )
        {
          for( int dim0 = 0; dim0 < NDIM; ++dim0 )
          {
            INDEX_TYPE const dof0 = NDIM * elemToNodeMap( elemID, localNode0 ) + dim0;
            dofNumbers[ NDIM * localNode0 + dim0 ] = dof0;

            for( INDEX_TYPE localNode1 = 0; localNode1 < NODES_PER_ELEM; ++localNode1 )
            {
              for( int dim1 = 0; dim1 < NDIM; ++dim1 )
              {
                INDEX_TYPE const dof1 = NDIM * elemToNodeMap( elemID, localNode1 ) + dim1;
                additions[ NDIM * localNode0 + dim0][ NDIM * localNode1 + dim1 ] = dof0 - dof1;
              }
            }

          }
        }

        for( int localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
        {
          for( int dim = 0; dim < NDIM; ++dim )
          {
            matrix.addToRowBinarySearchUnsorted< typename RAJAHelper< POLICY >::AtomicPolicy >( dofNumbers[ NDIM * localNode + dim ], dofNumbers,
                                                                                                additions[ NDIM * localNode + dim ], NODES_PER_ELEM * NDIM );
          }
        }
      } );
}

// Explicit instantiation of the templated SparsityGenerationNative static methods.
template void SparsityGenerationNative::generateElemLoop< SparsityPattern< COLUMN_TYPE, INDEX_TYPE > >(
  SparsityPattern< COLUMN_TYPE, INDEX_TYPE > &,
  ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & );

template void SparsityGenerationNative::generateElemLoop< SparsityPatternView< COLUMN_TYPE, INDEX_TYPE const > const >(
  SparsityPatternView< COLUMN_TYPE, INDEX_TYPE const > const &,
  ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & );

template void SparsityGenerationNative::generateNodeLoop< SparsityPattern< COLUMN_TYPE, INDEX_TYPE > >(
  SparsityPattern< COLUMN_TYPE, INDEX_TYPE > &,
  ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const &,
  ArrayOfArraysView< INDEX_TYPE const, INDEX_TYPE const, true > const & nodeToElemMap );

template void SparsityGenerationNative::generateNodeLoop< SparsityPatternView< COLUMN_TYPE, INDEX_TYPE const > const >(
  SparsityPatternView< COLUMN_TYPE, INDEX_TYPE const > const &,
  ArrayView< INDEX_TYPE const, ELEM_TO_NODE_PERM > const &,
  ArrayOfArraysView< INDEX_TYPE const, INDEX_TYPE const, true > const & nodeToElemMap );

// Explicit instantiation of SparsityGenerationRAJA.
template class SparsityGenerationRAJA< serialPolicy >;
template class CRSMatrixAddToRow< serialPolicy >;

#if defined(USE_OPENMP)
template class SparsityGenerationRAJA< parallelHostPolicy >;
template class CRSMatrixAddToRow< parallelHostPolicy >;
#endif

#if defined(USE_CUDA)
template class SparsityGenerationRAJA< parallelDevicePolicy< THREADS_PER_BLOCK > >;
template class CRSMatrixAddToRow< parallelDevicePolicy< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray

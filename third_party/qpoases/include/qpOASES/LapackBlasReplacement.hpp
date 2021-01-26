/*
 *	This file is part of qpOASES.
 *
 *	qpOASES -- An Implementation of the Online Active Set Strategy.
 *	Copyright (C) 2007-2017 by Hans Joachim Ferreau, Andreas Potschka,
 *	Christian Kirches et al. All rights reserved.
 *
 *	qpOASES is free software; you can redistribute it and/or
 *	modify it under the terms of the GNU Lesser General Public
 *	License as published by the Free Software Foundation; either
 *	version 2.1 of the License, or (at your option) any later version.
 *
 *	qpOASES is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *	See the GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public
 *	License along with qpOASES; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


/**
 *	\file include/qpOASES/LapackBlasReplacement.hpp
 *	\author Andreas Potschka, Hans Joachim Ferreau, Christian Kirches
 *	\version 3.2
 *	\date 2009-2017
 *
 *  Declarations for external LAPACK/BLAS functions.
 */



#ifndef QPOASES_LAPACKBLASREPLACEMENT_HPP
#define QPOASES_LAPACKBLASREPLACEMENT_HPP


#ifdef __AVOID_LA_NAMING_CONFLICTS__

	#define SGEMM  qpOASES_sgemm
	#define DGEMM  qpOASES_gemm
	#define SPOTRF qpOASES_spotrf
	#define DPOTRF qpOASES_dpotrf
	#define STRTRS qpOASES_strtrs
	#define DTRTRS qpOASES_dtrtrs
	#define STRCON qpOASES_strcon
	#define DTRCON qpOASES_dtrcon

#else

	#define SGEMM  sgemm_
	#define DGEMM  dgemm_
	#define SPOTRF spotrf_
	#define DPOTRF dpotrf_
	#define STRTRS strtrs_
	#define DTRTRS dtrtrs_
	#define STRCON strcon_
	#define DTRCON dtrcon_

#endif


#ifdef __USE_SINGLE_PRECISION__

	/** Macro for calling level 3 BLAS operation in single precision. */
	#define GEMM  SGEMM
	/** Macro for calling level 3 BLAS operation in single precision. */
	#define POTRF SPOTRF

	/** Macro for calling level 3 BLAS operation in single precision. */
	#define TRTRS STRTRS
	/** Macro for calling level 3 BLAS operation in single precision. */
	#define TRCON strcon_

#else

	/** Macro for calling level 3 BLAS operation in double precision. */
	#define GEMM  DGEMM
	/** Macro for calling level 3 BLAS operation in double precision. */
	#define POTRF DPOTRF

	/** Macro for calling level 3 BLAS operation in double precision. */
	#define TRTRS DTRTRS
	/** Macro for calling level 3 BLAS operation in double precision. */
	#define TRCON DTRCON

#endif /* __USE_SINGLE_PRECISION__ */


extern "C"
{
	/** Performs one of the matrix-matrix operation in double precision. */
	void DGEMM(		const char*, const char*, const la_uint_t*, const la_uint_t*, const la_uint_t*,
					const double*, const double*, const la_uint_t*, const double*, const la_uint_t*,
					const double*, double*, const la_uint_t* );
	/** Performs one of the matrix-matrix operation in single precision. */
	void SGEMM(		const char*, const char*, const la_uint_t*, const la_uint_t*, const la_uint_t*,
					const float*, const float*, const la_uint_t*, const float*, const la_uint_t*,
					const float*, float*, const la_uint_t* );

	/** Calculates the Cholesky factorization of a real symmetric positive definite matrix in double precision. */
	void DPOTRF(	const char*, const la_uint_t*, double*, const la_uint_t*, la_int_t* );
	/** Calculates the Cholesky factorization of a real symmetric positive definite matrix in single precision. */
	void SPOTRF(	const char*, const la_uint_t*, float*, const la_uint_t*, la_int_t* );

	/** Solves a triangular system (double precision) */
	void DTRTRS(	const char* UPLO, const char* TRANS, const char* DIAG, const la_uint_t* N, const la_uint_t* NRHS,
					double* A, const la_uint_t* LDA, double* B, const la_uint_t* LDB, la_int_t* INFO );
	/** Solves a triangular system (single precision) */
	void STRTRS(	const char* UPLO, const char* TRANS, const char* DIAG, const la_uint_t* N, const la_uint_t* NRHS,
					float* A, const la_uint_t* LDA, float* B, const la_uint_t* LDB, la_int_t* INFO );

	/** Estimate the reciprocal of the condition number of a triangular matrix in double precision */
	void DTRCON(	const char* NORM, const char* UPLO, const char* DIAG, const la_uint_t* N, double* A, const la_uint_t* LDA,
					double* RCOND, double* WORK, const la_uint_t* IWORK, la_int_t* INFO );
	/** Estimate the reciprocal of the condition number of a triangular matrix in single precision */
	void STRCON(	const char* NORM, const char* UPLO, const char* DIAG, const la_uint_t* N, float* A, const la_uint_t* LDA,
					float* RCOND, float* WORK, const la_uint_t* IWORK, la_int_t* INFO );
}

#endif	/* QPOASES_LAPACKBLASREPLACEMENT_HPP */


/*
 *	end of file
 */

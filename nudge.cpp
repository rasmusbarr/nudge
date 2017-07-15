//
// Copyright (c) 2017 Rasmus Barringer
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "nudge.h"
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <intrin.h>
#define NUDGE_ALIGNED(n) __declspec(align(n))
#define NUDGE_FORCEINLINE __forceinline
#else
#define NUDGE_ALIGNED(n) __attribute__((aligned(n)))
#define NUDGE_FORCEINLINE inline __attribute__((always_inline))
#endif

#ifdef __AVX2__
#define NUDGE_SIMDV_WIDTH 256
#else
#define NUDGE_SIMDV_WIDTH 128
#endif

#define NUDGE_ARENA_SCOPE(A) Arena& scope_arena_##A = A; Arena A = scope_arena_##A

namespace nudge {

static const float allowed_penetration = 1e-3f;
static const float bias_factor = 2.0f;

#if NUDGE_SIMDV_WIDTH == 128
#define NUDGE_SIMDV_ALIGNED NUDGE_ALIGNED(16)
static const unsigned simdv_width32 = 4;
static const unsigned simdv_width32_log2 = 2;
#elif NUDGE_SIMDV_WIDTH == 256
#define NUDGE_SIMDV_ALIGNED NUDGE_ALIGNED(32)
static const unsigned simdv_width32 = 8;
static const unsigned simdv_width32_log2 = 3;
#endif

#ifdef _WIN32
NUDGE_FORCEINLINE __m128 operator - (__m128 a) {
	return _mm_xor_ps(a, _mm_set1_ps(-0.0f));
}

NUDGE_FORCEINLINE __m128 operator + (__m128 a, __m128 b) {
	return _mm_add_ps(a, b);
}

NUDGE_FORCEINLINE __m128 operator - (__m128 a, __m128 b) {
	return _mm_sub_ps(a, b);
}

NUDGE_FORCEINLINE __m128 operator * (__m128 a, __m128 b) {
	return _mm_mul_ps(a, b);
}

NUDGE_FORCEINLINE __m128 operator / (__m128 a, __m128 b) {
	return _mm_div_ps(a, b);
}

NUDGE_FORCEINLINE __m128& operator += (__m128& a, __m128 b) {
	return a = _mm_add_ps(a, b);
}

NUDGE_FORCEINLINE __m128& operator -= (__m128& a, __m128 b) {
	return a = _mm_sub_ps(a, b);
}

NUDGE_FORCEINLINE __m128& operator *= (__m128& a, __m128 b) {
	return a = _mm_mul_ps(a, b);
}

NUDGE_FORCEINLINE __m128& operator /= (__m128& a, __m128 b) {
	return a = _mm_div_ps(a, b);
}
#ifdef __AVX2__
NUDGE_FORCEINLINE __m256 operator - (__m256 a) {
	return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
}

NUDGE_FORCEINLINE __m256 operator + (__m256 a, __m256 b) {
	return _mm256_add_ps(a, b);
}

NUDGE_FORCEINLINE __m256 operator - (__m256 a, __m256 b) {
	return _mm256_sub_ps(a, b);
}

NUDGE_FORCEINLINE __m256 operator * (__m256 a, __m256 b) {
	return _mm256_mul_ps(a, b);
}

NUDGE_FORCEINLINE __m256 operator / (__m256 a, __m256 b) {
	return _mm256_div_ps(a, b);
}

NUDGE_FORCEINLINE __m256& operator += (__m256& a, __m256 b) {
	return a = _mm256_add_ps(a, b);
}

NUDGE_FORCEINLINE __m256& operator -= (__m256& a, __m256 b) {
	return a = _mm256_sub_ps(a, b);
}

NUDGE_FORCEINLINE __m256& operator *= (__m256& a, __m256 b) {
	return a = _mm256_mul_ps(a, b);
}

NUDGE_FORCEINLINE __m256& operator /= (__m256& a, __m256 b) {
	return a = _mm256_div_ps(a, b);
}
#endif
#endif

typedef __m128 simd4_float;
typedef __m128i simd4_int32;

namespace simd128 {
	NUDGE_FORCEINLINE __m128 unpacklo32(__m128 x, __m128 y) {
		return _mm_unpacklo_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m128 unpackhi32(__m128 x, __m128 y) {
		return _mm_unpackhi_ps(x, y);
	}

	NUDGE_FORCEINLINE __m128i unpacklo32(__m128i x, __m128i y) {
		return _mm_unpacklo_epi32(x, y);
	}

	NUDGE_FORCEINLINE __m128i unpackhi32(__m128i x, __m128i y) {
		return _mm_unpackhi_epi32(x, y);
	}
	
	template<unsigned x0, unsigned x1, unsigned y0, unsigned y1>
	NUDGE_FORCEINLINE __m128 concat2x32(__m128 x, __m128 y) {
		return _mm_shuffle_ps(x, y, _MM_SHUFFLE(y1, y0, x1, x0));
	}
	
	template<unsigned i0, unsigned i1, unsigned i2, unsigned i3>
	NUDGE_FORCEINLINE __m128 shuffle32(__m128 x) {
		return _mm_shuffle_ps(x, x, _MM_SHUFFLE(i3, i2, i1, i0));
	}
	
	template<unsigned i0, unsigned i1, unsigned i2, unsigned i3>
	NUDGE_FORCEINLINE __m128i shuffle32(__m128i x) {
		return _mm_shuffle_epi32(x, _MM_SHUFFLE(i3, i2, i1, i0));
	}
	
	NUDGE_FORCEINLINE void transpose32(simd4_float& x, simd4_float& y, simd4_float& z, simd4_float& w) {
		_MM_TRANSPOSE4_PS(x, y, z, w);
	}
}

namespace simd {
	NUDGE_FORCEINLINE unsigned signmask32(__m128 x) {
		return _mm_movemask_ps(x);
	}
	
	NUDGE_FORCEINLINE unsigned signmask32(__m128i x) {
		return _mm_movemask_ps(_mm_castsi128_ps(x));
	}
	
	NUDGE_FORCEINLINE __m128 bitwise_xor(__m128 x, __m128 y) {
		return _mm_xor_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m128 bitwise_or(__m128 x, __m128 y) {
		return _mm_or_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m128 bitwise_and(__m128 x, __m128 y) {
		return _mm_and_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m128 bitwise_notand(__m128 x, __m128 y) {
		return _mm_andnot_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m128i bitwise_xor(__m128i x, __m128i y) {
		return _mm_xor_si128(x, y);
	}
	
	NUDGE_FORCEINLINE __m128i bitwise_or(__m128i x, __m128i y) {
		return _mm_or_si128(x, y);
	}
	
	NUDGE_FORCEINLINE __m128i bitwise_and(__m128i x, __m128i y) {
		return _mm_and_si128(x, y);
	}
	
	NUDGE_FORCEINLINE __m128i bitwise_notand(__m128i x, __m128i y) {
		return _mm_andnot_si128(x, y);
	}
	
	NUDGE_FORCEINLINE __m128 blendv32(__m128 x, __m128 y, __m128 s) {
#if defined(__SSE4_1__) || defined(__AVX__)
#define NUDGE_NATIVE_BLENDV32
		return _mm_blendv_ps(x, y, s);
#else
		s = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(s), 31));
		return _mm_or_ps(_mm_andnot_ps(s, x), _mm_and_ps(s, y));
#endif
	}

	NUDGE_FORCEINLINE __m128i blendv32(__m128i x, __m128i y, __m128i s) {
		return _mm_castps_si128(blendv32(_mm_castsi128_ps(x), _mm_castsi128_ps(y), _mm_castsi128_ps(s)));
	}
}

namespace simd_float {
	NUDGE_FORCEINLINE float extract_first_float(simd4_float x) {
		return _mm_cvtss_f32(x);
	}
	
	NUDGE_FORCEINLINE simd4_float zero4() {
		return _mm_setzero_ps();
	}
	
	NUDGE_FORCEINLINE simd4_float make4(float x) {
		return _mm_set1_ps(x);
	}
	
	NUDGE_FORCEINLINE simd4_float make4(float x, float y, float z, float w) {
		return _mm_setr_ps(x, y, z, w);
	}
	
	NUDGE_FORCEINLINE simd4_float broadcast_load4(const float* p) {
		return _mm_set1_ps(*p);
	}
	
	NUDGE_FORCEINLINE simd4_float load4(const float* p) {
		return _mm_load_ps(p);
	}
	
	NUDGE_FORCEINLINE simd4_float loadu4(const float* p) {
		return _mm_loadu_ps(p);
	}
	
	NUDGE_FORCEINLINE void store4(float* p, simd4_float x) {
		_mm_store_ps(p, x);
	}
	
	NUDGE_FORCEINLINE void storeu4(float* p, simd4_float x) {
		_mm_storeu_ps(p, x);
	}
	
	NUDGE_FORCEINLINE simd4_float madd(simd4_float x, simd4_float y, simd4_float z) {
#ifdef __FMA__
		return _mm_fmadd_ps(x, y, z);
#else
		return _mm_add_ps(_mm_mul_ps(x, y), z);
#endif
	}
	
	NUDGE_FORCEINLINE simd4_float msub(simd4_float x, simd4_float y, simd4_float z) {
#ifdef __FMA__
		return _mm_fmsub_ps(x, y, z);
#else
		return _mm_sub_ps(_mm_mul_ps(x, y), z);
#endif
	}
	
	// Note: First operand is returned on NaN.
	NUDGE_FORCEINLINE simd4_float min(simd4_float x, simd4_float y) {
		return _mm_min_ps(y, x); // Note: For SSE, second operand is returned on NaN.
	}
	
	// Note: First operand is returned on NaN.
	NUDGE_FORCEINLINE simd4_float max(simd4_float x, simd4_float y) {
		return _mm_max_ps(y, x); // Note: For SSE, second operand is returned on NaN.
	}
	
	NUDGE_FORCEINLINE simd4_float rsqrt(simd4_float x) {
		return _mm_rsqrt_ps(x);
	}
	
	NUDGE_FORCEINLINE simd4_float recip(simd4_float x) {
		return _mm_rcp_ps(x);
	}
	
	NUDGE_FORCEINLINE simd4_float sqrt(simd4_float x) {
		return _mm_sqrt_ps(x);
	}
	
	NUDGE_FORCEINLINE simd4_float abs(simd4_float x) {
		return _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
	}
	
	NUDGE_FORCEINLINE simd4_float cmp_gt(simd4_float x, simd4_float y) {
		return _mm_cmpgt_ps(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_float cmp_ge(simd4_float x, simd4_float y) {
		return _mm_cmpge_ps(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_float cmp_le(simd4_float x, simd4_float y) {
		return _mm_cmple_ps(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_float cmp_eq(simd4_float x, simd4_float y) {
		return _mm_cmpeq_ps(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_float cmp_neq(simd4_float x, simd4_float y) {
		return _mm_cmpneq_ps(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_int32 asint(simd4_float x) {
		return _mm_castps_si128(x);
	}
	
	NUDGE_FORCEINLINE simd4_int32 toint(simd4_float x) {
		return _mm_cvttps_epi32(x);
	}
}

namespace simd_int32 {
	NUDGE_FORCEINLINE simd4_int32 zero4() {
		return _mm_setzero_si128();
	}
	
	NUDGE_FORCEINLINE simd4_int32 make4(int32_t x) {
		return _mm_set1_epi32(x);
	}
	
	NUDGE_FORCEINLINE simd4_int32 make4(int32_t x, int32_t y, int32_t z, int32_t w) {
		return _mm_setr_epi32(x, y, z, w);
	}
	
	NUDGE_FORCEINLINE simd4_int32 load4(const int32_t* p) {
		return _mm_load_si128((const __m128i*)p);
	}
	
	NUDGE_FORCEINLINE simd4_int32 loadu4(const int32_t* p) {
		return _mm_loadu_si128((const __m128i*)p);
	}
	
	NUDGE_FORCEINLINE void store4(int32_t* p, simd4_int32 x) {
		_mm_store_si128((__m128i*)p, x);
	}
	
	NUDGE_FORCEINLINE void storeu4(int32_t* p, simd4_int32 x) {
		_mm_storeu_si128((__m128i*)p, x);
	}
	
	template<unsigned bits>
	NUDGE_FORCEINLINE simd4_int32 shift_left(simd4_int32 x) {
		return _mm_slli_epi32(x, bits);
	}
	
	template<unsigned bits>
	NUDGE_FORCEINLINE simd4_int32 shift_right(simd4_int32 x) {
		return _mm_srli_epi32(x, bits);
	}
	
	NUDGE_FORCEINLINE simd4_int32 add(simd4_int32 x, simd4_int32 y) {
		return _mm_add_epi32(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_int32 cmp_eq(simd4_int32 x, simd4_int32 y) {
		return _mm_cmpeq_epi32(x, y);
	}
	
	NUDGE_FORCEINLINE simd4_float asfloat(simd4_int32 x) {
		return _mm_castsi128_ps(x);
	}
}

#ifdef __AVX2__
typedef __m256 simd8_float;
typedef __m256i simd8_int32;

namespace simd128 {
	NUDGE_FORCEINLINE __m256 unpacklo32(__m256 x, __m256 y) {
		return _mm256_unpacklo_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m256 unpackhi32(__m256 x, __m256 y) {
		return _mm256_unpackhi_ps(x, y);
	}

	NUDGE_FORCEINLINE __m256i unpacklo32(__m256i x, __m256i y) {
		return _mm256_unpacklo_epi32(x, y);
	}

	NUDGE_FORCEINLINE __m256i unpackhi32(__m256i x, __m256i y) {
		return _mm256_unpackhi_epi32(x, y);
	}
	
	template<unsigned x0, unsigned x1, unsigned y0, unsigned y1>
	NUDGE_FORCEINLINE __m256 concat2x32(__m256 x, __m256 y) {
		return _mm256_shuffle_ps(x, y, _MM_SHUFFLE(y1, y0, x1, x0));
	}
	
	template<unsigned i0, unsigned i1, unsigned i2, unsigned i3>
	NUDGE_FORCEINLINE __m256 shuffle32(__m256 x) {
		return _mm256_shuffle_ps(x, x, _MM_SHUFFLE(i3, i2, i1, i0));
	}
	
	template<unsigned i0, unsigned i1, unsigned i2, unsigned i3>
	NUDGE_FORCEINLINE __m256i shuffle32(__m256i x) {
		return _mm256_shuffle_epi32(x, _MM_SHUFFLE(i3, i2, i1, i0));
	}
	
	NUDGE_FORCEINLINE void transpose32(simd8_float& x, simd8_float& y, simd8_float& z, simd8_float& w) {
		__m256 t0 = _mm256_unpacklo_ps(x, y);
		__m256 t1 = _mm256_unpacklo_ps(z, w);
		__m256 t2 = _mm256_unpackhi_ps(x, y);
		__m256 t3 = _mm256_unpackhi_ps(z, w);
		x = _mm256_shuffle_ps(t0, t1, _MM_SHUFFLE(1,0,1,0));
		y = _mm256_shuffle_ps(t0, t1, _MM_SHUFFLE(3,2,3,2));
		z = _mm256_shuffle_ps(t2, t3, _MM_SHUFFLE(1,0,1,0));
		w = _mm256_shuffle_ps(t2, t3, _MM_SHUFFLE(3,2,3,2));
	}
}

namespace simd256 {
	template<unsigned i0, unsigned i1>
	NUDGE_FORCEINLINE simd8_float permute128(simd8_float x, simd8_float y) {
		return _mm256_castsi256_ps(_mm256_permute2x128_si256(_mm256_castps_si256(x), _mm256_castps_si256(y), i0 | (i1 << 4)));
	}

	template<unsigned i0, unsigned i1>
	NUDGE_FORCEINLINE simd8_int32 permute128(simd8_int32 x, simd8_int32 y) {
		return _mm256_permute2x128_si256(x, y, i0 | (i1 << 4));
	}
	
	template<unsigned i0, unsigned i1>
	NUDGE_FORCEINLINE simd8_float shuffle128(simd8_float x) {
		return _mm256_castsi256_ps(_mm256_permute2x128_si256(_mm256_castps_si256(x), _mm256_castps_si256(x), i0 | (i1 << 4)));
	}

	template<unsigned i0, unsigned i1>
	NUDGE_FORCEINLINE simd8_int32 shuffle128(simd8_int32 x) {
		return _mm256_permute2x128_si256(x, x, i0 | (i1 << 4));
	}
	
	NUDGE_FORCEINLINE simd8_float broadcast(simd4_float x) {
		return _mm256_insertf128_ps(_mm256_castps128_ps256(x), x, 1);
	}

	NUDGE_FORCEINLINE simd8_int32 broadcast(simd4_int32 x) {
		return _mm256_insertf128_si256(_mm256_castsi128_si256(x), x, 1);
	}
}

namespace simd {
	NUDGE_FORCEINLINE simd8_float concat(simd4_float x, simd4_float y) {
		return _mm256_insertf128_ps(_mm256_castps128_ps256(x), y, 1);
	}
	
	NUDGE_FORCEINLINE simd4_float extract_low(simd8_float x) {
		return _mm256_castps256_ps128(x);
	}
	
	NUDGE_FORCEINLINE simd4_float extract_high(simd8_float x) {
		return _mm256_extractf128_ps(x, 1);
	}

	NUDGE_FORCEINLINE simd4_int32 extract_low(simd8_int32 x) {
		return _mm256_castsi256_si128(x);
	}

	NUDGE_FORCEINLINE simd4_int32 extract_high(simd8_int32 x) {
		return _mm256_extractf128_si256(x, 1);
	}
	
	NUDGE_FORCEINLINE unsigned signmask32(__m256 x) {
		return _mm256_movemask_ps(x);
	}
	
	NUDGE_FORCEINLINE unsigned signmask32(__m256i x) {
		return _mm256_movemask_ps(_mm256_castsi256_ps(x));
	}
	
	NUDGE_FORCEINLINE __m256 bitwise_xor(__m256 x, __m256 y) {
		return _mm256_xor_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m256 bitwise_or(__m256 x, __m256 y) {
		return _mm256_or_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m256 bitwise_and(__m256 x, __m256 y) {
		return _mm256_and_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m256 bitwise_notand(__m256 x, __m256 y) {
		return _mm256_andnot_ps(x, y);
	}
	
	NUDGE_FORCEINLINE __m256i bitwise_xor(__m256i x, __m256i y) {
		return _mm256_xor_si256(x, y);
	}
	
	NUDGE_FORCEINLINE __m256i bitwise_or(__m256i x, __m256i y) {
		return _mm256_or_si256(x, y);
	}
	
	NUDGE_FORCEINLINE __m256i bitwise_and(__m256i x, __m256i y) {
		return _mm256_and_si256(x, y);
	}
	
	NUDGE_FORCEINLINE __m256i bitwise_notand(__m256i x, __m256i y) {
		return _mm256_andnot_si256(x, y);
	}
	
	NUDGE_FORCEINLINE __m256 blendv32(__m256 x, __m256 y, __m256 s) {
		return _mm256_blendv_ps(x, y, s);
	}

	NUDGE_FORCEINLINE __m256i blendv32(__m256i x, __m256i y, __m256i s) {
		return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), _mm256_castsi256_ps(s)));
	}
}

namespace simd_float {
	NUDGE_FORCEINLINE float extract_first_float(simd8_float x) {
		return _mm_cvtss_f32(_mm256_castps256_ps128(x));
	}
	
	NUDGE_FORCEINLINE simd8_float zero8() {
		return _mm256_setzero_ps();
	}
	
	NUDGE_FORCEINLINE simd8_float make8(float x) {
		return _mm256_set1_ps(x);
	}
	
	NUDGE_FORCEINLINE simd8_float make8(float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1) {
		return _mm256_setr_ps(x0, y0, z0, w0, x1, y1, z1, w1);
	}
	
	NUDGE_FORCEINLINE simd8_float broadcast_load8(const float* p) {
		return _mm256_broadcast_ss(p);
	}
	
	NUDGE_FORCEINLINE simd8_float load8(const float* p) {
		return _mm256_load_ps(p);
	}
	
	NUDGE_FORCEINLINE simd8_float loadu8(const float* p) {
		return _mm256_loadu_ps(p);
	}
	
	NUDGE_FORCEINLINE void store8(float* p, simd8_float x) {
		_mm256_store_ps(p, x);
	}
	
	NUDGE_FORCEINLINE void storeu8(float* p, simd8_float x) {
		_mm256_storeu_ps(p, x);
	}
	
	NUDGE_FORCEINLINE simd8_float madd(simd8_float x, simd8_float y, simd8_float z) {
#ifdef __FMA__
		return _mm256_fmadd_ps(x, y, z);
#else
		return _mm256_add_ps(_mm256_mul_ps(x, y), z);
#endif
	}
	
	NUDGE_FORCEINLINE simd8_float msub(simd8_float x, simd8_float y, simd8_float z) {
#ifdef __FMA__
		return _mm256_fmsub_ps(x, y, z);
#else
		return _mm256_sub_ps(_mm256_mul_ps(x, y), z);
#endif
	}
	
	// Note: First operand is returned on NaN.
	NUDGE_FORCEINLINE simd8_float min(simd8_float x, simd8_float y) {
		return _mm256_min_ps(y, x); // Note: For SSE, second operand is returned on NaN.
	}
	
	// Note: First operand is returned on NaN.
	NUDGE_FORCEINLINE simd8_float max(simd8_float x, simd8_float y) {
		return _mm256_max_ps(y, x); // Note: For SSE, second operand is returned on NaN.
	}
	
	NUDGE_FORCEINLINE simd8_float rsqrt(simd8_float x) {
		return _mm256_rsqrt_ps(x);
	}
	
	NUDGE_FORCEINLINE simd8_float recip(simd8_float x) {
		return _mm256_rcp_ps(x);
	}
	
	NUDGE_FORCEINLINE simd8_float sqrt(simd8_float x) {
		return _mm256_sqrt_ps(x);
	}
	
	NUDGE_FORCEINLINE simd8_float abs(simd8_float x) {
		return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
	}
	
	NUDGE_FORCEINLINE simd8_float cmp_gt(simd8_float x, simd8_float y) {
		return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
	}
	
	NUDGE_FORCEINLINE simd8_float cmp_ge(simd8_float x, simd8_float y) {
		return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
	}
	
	NUDGE_FORCEINLINE simd8_float cmp_le(simd8_float x, simd8_float y) {
		return _mm256_cmp_ps(x, y, _CMP_LE_OQ);
	}
	
	NUDGE_FORCEINLINE simd8_float cmp_eq(simd8_float x, simd8_float y) {
		return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
	}
	
	NUDGE_FORCEINLINE simd8_float cmp_neq(simd8_float x, simd8_float y) {
		return _mm256_cmp_ps(x, y, _CMP_NEQ_OQ);
	}
	
	NUDGE_FORCEINLINE simd8_int32 asint(simd8_float x) {
		return _mm256_castps_si256(x);
	}
	
	NUDGE_FORCEINLINE simd8_int32 toint(simd8_float x) {
		return _mm256_cvttps_epi32(x);
	}
}

namespace simd_int32 {
	NUDGE_FORCEINLINE simd8_int32 zero8() {
		return _mm256_setzero_si256();
	}
	
	NUDGE_FORCEINLINE simd8_int32 make8(int32_t x) {
		return _mm256_set1_epi32(x);
	}
	
	NUDGE_FORCEINLINE simd8_int32 make8(int32_t x0, int32_t y0, int32_t z0, int32_t w0, int32_t x1, int32_t y1, int32_t z1, int32_t w1) {
		return _mm256_setr_epi32(x0, y0, z0, w0, x1, y1, z1, w1);
	}
	
	NUDGE_FORCEINLINE simd8_int32 load8(const int32_t* p) {
		return _mm256_load_si256((const __m256i*)p);
	}
	
	NUDGE_FORCEINLINE simd8_int32 loadu8(const int32_t* p) {
		return _mm256_loadu_si256((const __m256i*)p);
	}
	
	NUDGE_FORCEINLINE void store8(int32_t* p, simd8_int32 x) {
		_mm256_store_si256((__m256i*)p, x);
	}
	
	NUDGE_FORCEINLINE void storeu8(int32_t* p, simd8_int32 x) {
		_mm256_storeu_si256((__m256i*)p, x);
	}
	
	template<unsigned bits>
	NUDGE_FORCEINLINE simd8_int32 shift_left(simd8_int32 x) {
		return _mm256_slli_epi32(x, bits);
	}
	
	template<unsigned bits>
	NUDGE_FORCEINLINE simd8_int32 shift_right(simd8_int32 x) {
		return _mm256_srli_epi32(x, bits);
	}
	
	NUDGE_FORCEINLINE simd8_int32 add(simd8_int32 x, simd8_int32 y) {
		return _mm256_add_epi32(x, y);
	}
	
	NUDGE_FORCEINLINE simd8_int32 cmp_eq(simd8_int32 x, simd8_int32 y) {
		return _mm256_cmpeq_epi32(x, y);
	}
	
	NUDGE_FORCEINLINE simd8_float asfloat(simd8_int32 x) {
		return _mm256_castsi256_ps(x);
	}
}
#endif

#if NUDGE_SIMDV_WIDTH == 128
typedef simd4_float simdv_float;
typedef simd4_int32 simdv_int32;

namespace simd_float {
	NUDGE_FORCEINLINE simdv_float zerov() {
		return zero4();
	}
	
	NUDGE_FORCEINLINE simdv_float makev(float x) {
		return make4(x);
	}
	
	NUDGE_FORCEINLINE simdv_float broadcast_loadv(const float* p) {
		return broadcast_load4(p);
	}
	
	NUDGE_FORCEINLINE simdv_float loadv(const float* p) {
		return load4(p);
	}
	
	NUDGE_FORCEINLINE simdv_float loaduv(const float* p) {
		return loadu4(p);
	}
	
	NUDGE_FORCEINLINE void storev(float* p, simdv_float x) {
		store4(p, x);
	}
	
	NUDGE_FORCEINLINE void storeuv(float* p, simdv_float x) {
		storeu4(p, x);
	}
}

namespace simd_int32 {
	NUDGE_FORCEINLINE simdv_int32 zerov() {
		return zero4();
	}
	
	NUDGE_FORCEINLINE simdv_int32 makev(int32_t x) {
		return make4(x);
	}
	
	NUDGE_FORCEINLINE simdv_int32 loadv(const int32_t* p) {
		return load4(p);
	}
	
	NUDGE_FORCEINLINE simdv_int32 loaduv(const int32_t* p) {
		return loadu4(p);
	}
	
	NUDGE_FORCEINLINE void storev(int32_t* p, simdv_int32 x) {
		store4(p, x);
	}
	
	NUDGE_FORCEINLINE void storeuv(int32_t* p, simdv_int32 x) {
		storeu4(p, x);
	}
}
#elif NUDGE_SIMDV_WIDTH == 256
typedef simd8_float simdv_float;
typedef simd8_int32 simdv_int32;

namespace simd_float {
	NUDGE_FORCEINLINE simdv_float zerov() {
		return zero8();
	}
	
	NUDGE_FORCEINLINE simdv_float makev(float x) {
		return make8(x);
	}
	
	NUDGE_FORCEINLINE simdv_float broadcast_loadv(const float* p) {
		return broadcast_load8(p);
	}
	
	NUDGE_FORCEINLINE simdv_float loadv(const float* p) {
		return load8(p);
	}
	
	NUDGE_FORCEINLINE simdv_float loaduv(const float* p) {
		return loadu8(p);
	}
	
	NUDGE_FORCEINLINE void storev(float* p, simdv_float x) {
		store8(p, x);
	}
	
	NUDGE_FORCEINLINE void storeuv(float* p, simdv_float x) {
		storeu8(p, x);
	}
}

namespace simd_int32 {
	NUDGE_FORCEINLINE simdv_int32 zerov() {
		return zero8();
	}
	
	NUDGE_FORCEINLINE simdv_int32 makev(int32_t x) {
		return make8(x);
	}
	
	NUDGE_FORCEINLINE simdv_int32 loadv(const int32_t* p) {
		return load8(p);
	}
	
	NUDGE_FORCEINLINE simdv_int32 loaduv(const int32_t* p) {
		return loadu8(p);
	}
	
	NUDGE_FORCEINLINE void storev(int32_t* p, simdv_int32 x) {
		store8(p, x);
	}
	
	NUDGE_FORCEINLINE void storeuv(int32_t* p, simdv_int32 x) {
		storeu8(p, x);
	}
}
#endif

namespace simd_aos {
	NUDGE_FORCEINLINE simd4_float dot(simd4_float a, simd4_float b) {
		simd4_float c = a*b;
		return simd128::shuffle32<0,0,0,0>(c) + simd128::shuffle32<1,1,1,1>(c) + simd128::shuffle32<2,2,2,2>(c);
	}
	
	NUDGE_FORCEINLINE simd4_float cross(simd4_float a, simd4_float b) {
		simd4_float c = simd128::shuffle32<1,2,0,0>(a) * simd128::shuffle32<2,0,1,0>(b);
		simd4_float d = simd128::shuffle32<2,0,1,0>(a) * simd128::shuffle32<1,2,0,0>(b);
		return c - d;
	}
}

namespace simd_soa {
	NUDGE_FORCEINLINE void cross(simd4_float ax, simd4_float ay, simd4_float az, simd4_float bx, simd4_float by, simd4_float bz, simd4_float& rx, simd4_float& ry, simd4_float& rz) {
		rx = ay*bz - az*by;
		ry = az*bx - ax*bz;
		rz = ax*by - ay*bx;
	}
	
	NUDGE_FORCEINLINE void normalize(simd4_float& x, simd4_float& y, simd4_float& z) {
		simd4_float f = simd_float::rsqrt(x*x + y*y + z*z);
		x *= f;
		y *= f;
		z *= f;
	}
	
#if NUDGE_SIMDV_WIDTH >= 256
	NUDGE_FORCEINLINE void cross(simd8_float ax, simd8_float ay, simd8_float az, simd8_float bx, simd8_float by, simd8_float bz, simd8_float& rx, simd8_float& ry, simd8_float& rz) {
		rx = ay*bz - az*by;
		ry = az*bx - ax*bz;
		rz = ax*by - ay*bx;
	}
	
	NUDGE_FORCEINLINE void normalize(simd8_float& x, simd8_float& y, simd8_float& z) {
		simd8_float f = simd_float::rsqrt(x*x + y*y + z*z);
		x *= f;
		y *= f;
		z *= f;
	}
#endif
}

namespace {
	struct float3 {
		float x, y, z;
	};
	
	struct float3x3 {
		float3 c0, c1, c2;
	};
	
	struct Rotation {
		float3 v;
		float s;
	};
	
	struct AABB {
		float3 min;
		float unused0;
		float3 max;
		float unused1;
	};
	
	struct AABBV {
		float min_x[simdv_width32];
		float max_x[simdv_width32];
		float min_y[simdv_width32];
		float max_y[simdv_width32];
		float min_z[simdv_width32];
		float max_z[simdv_width32];
	};
	
	struct ContactSlotV {
		uint32_t indices[simdv_width32];
	};
	
	struct ContactPairV {
		uint32_t ab[simdv_width32];
	};
	
	struct ContactConstraintV {
		uint16_t a[simdv_width32];
		uint16_t b[simdv_width32];
		
		float pa_z[simdv_width32];
		float pa_x[simdv_width32];
		float pa_y[simdv_width32];
		
		float pb_z[simdv_width32];
		float pb_x[simdv_width32];
		float pb_y[simdv_width32];
		
		float n_x[simdv_width32];
		float u_x[simdv_width32];
		float v_x[simdv_width32];
		
		float n_y[simdv_width32];
		float u_y[simdv_width32];
		float v_y[simdv_width32];
		
		float n_z[simdv_width32];
		float u_z[simdv_width32];
		float v_z[simdv_width32];
		
		float bias[simdv_width32];
		float friction[simdv_width32];
		float normal_velocity_to_normal_impulse[simdv_width32];
		
		float friction_coefficient_x[simdv_width32];
		float friction_coefficient_y[simdv_width32];
		float friction_coefficient_z[simdv_width32];
		
		float na_x[simdv_width32];
		float na_y[simdv_width32];
		float na_z[simdv_width32];
		
		float nb_x[simdv_width32];
		float nb_y[simdv_width32];
		float nb_z[simdv_width32];
		
		float ua_x[simdv_width32];
		float ua_y[simdv_width32];
		float ua_z[simdv_width32];
		
		float va_x[simdv_width32];
		float va_y[simdv_width32];
		float va_z[simdv_width32];
		
		float ub_x[simdv_width32];
		float ub_y[simdv_width32];
		float ub_z[simdv_width32];
		
		float vb_x[simdv_width32];
		float vb_y[simdv_width32];
		float vb_z[simdv_width32];
	};
	
	struct ContactConstraintStateV {
		float applied_normal_impulse[simdv_width32];
		float applied_friction_impulse_x[simdv_width32];
		float applied_friction_impulse_y[simdv_width32];
	};
	
	struct InertiaTransform {
		float xx;
		float yy;
		float zz;
		float unused0;
		float xy;
		float xz;
		float yz;
		float unused1;
	};
}

#ifdef _WIN32
static inline unsigned first_set_bit(unsigned x) {
	unsigned long r = 0;
	_BitScanForward(&r, x);
	return r;
}
#else
static inline unsigned first_set_bit(unsigned x) {
	return __builtin_ctz(x);
}
#endif

static inline void* align(Arena* arena, uintptr_t alignment) {
	uintptr_t data = (uintptr_t)arena->data;
	uintptr_t end = data + arena->size;
	uintptr_t mask = alignment-1;
	
	data = (data + mask) & ~mask;
	
	arena->data = (void*)data;
	arena->size = end - data;
	
	assert((intptr_t)arena->size >= 0); // Out of memory.
	
	return arena->data;
}

static inline void* allocate(Arena* arena, uintptr_t size) {
	void* data = arena->data;
	arena->data = (void*)((uintptr_t)data + size);
	arena->size -= size;
	
	assert((intptr_t)arena->size >= 0); // Out of memory.
	
	return data;
}

static inline void* allocate(Arena* arena, uintptr_t size, uintptr_t alignment) {
	align(arena, alignment);
	
	void* data = arena->data;
	arena->data = (void*)((uintptr_t)data + size);
	arena->size -= size;
	
	assert((intptr_t)arena->size >= 0); // Out of memory.
	
	return data;
}

template<class T>
static inline T* allocate_struct(Arena* arena, uintptr_t alignment) {
	return static_cast<T*>(allocate(arena, sizeof(T), alignment));
}

template<class T>
static inline T* allocate_array(Arena* arena, uintptr_t count, uintptr_t alignment) {
	return static_cast<T*>(allocate(arena, sizeof(T)*count, alignment));
}

static inline void* reserve(Arena* arena, uintptr_t size, uintptr_t alignment) {
	align(arena, alignment);
	assert(size <= arena->size); // Cannot reserve this amount.
	return arena->data;
}

static inline void commit(Arena* arena, uintptr_t size) {
	allocate(arena, size);
}

template<class T>
static inline T* reserve_array(Arena* arena, uintptr_t count, uintptr_t alignment) {
	return static_cast<T*>(reserve(arena, sizeof(T)*count, alignment));
}

template<class T>
static inline void commit_array(Arena* arena, uintptr_t count) {
	commit(arena, sizeof(T)*count);
}

static inline Rotation make_rotation(const float q[4]) {
	Rotation r = { { q[0], q[1], q[2] }, q[3] };
	return r;
}

static inline float3 make_float3(const float x[3]) {
	float3 r = { x[0], x[1], x[2] };
	return r;
}

static inline float3 make_float3(float x, float y, float z) {
	float3 r = { x, y, z };
	return r;
}

static inline float3 make_float3(float x) {
	float3 r = { x, x, x };
	return r;
}

static inline float3 operator + (float3 a, float3 b) {
	float3 r = { a.x + b.x, a.y + b.y, a.z + b.z };
	return r;
}

static inline float3 operator - (float3 a, float3 b) {
	float3 r = { a.x - b.x, a.y - b.y, a.z - b.z };
	return r;
}

static inline float3 operator * (float a, float3 b) {
	float3 r = { a * b.x, a * b.y, a * b.z };
	return r;
}

static inline float3 operator * (float3 a, float b) {
	float3 r = { a.x * b, a.y * b, a.z * b };
	return r;
}

static inline float3& operator *= (float3& a, float b) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

static inline float dot(float3 a, float3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline float length2(float3 a) {
	return dot(a, a);
}

static inline float3 cross(float3 a, float3 b) {
	float3 v = { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
	return v;
}

static inline float3 operator * (Rotation lhs, float3 rhs) {
	float3 t = 2.0f * cross(lhs.v, rhs);
	return rhs + lhs.s * t + cross(lhs.v, t);
}

static inline Rotation operator * (Rotation lhs, Rotation rhs) {
	float3 v = rhs.v*lhs.s + lhs.v*rhs.s + cross(lhs.v, rhs.v);
	Rotation r = { v, lhs.s*rhs.s - dot(lhs.v, rhs.v) };
	return r;
}

static inline Rotation normalize(Rotation r) {
	float f = 1.0f / sqrtf(r.s*r.s + r.v.x*r.v.x + r.v.y*r.v.y + r.v.z*r.v.z);
	r.v *= f;
	r.s *= f;
	return r;
}

static inline Rotation inverse(Rotation r) {
	r.v.x = -r.v.x;
	r.v.y = -r.v.y;
	r.v.z = -r.v.z;
	return r;
}

static inline float3x3 matrix(Rotation q) {
	float kx = q.v.x + q.v.x;
	float ky = q.v.y + q.v.y;
	float kz = q.v.z + q.v.z;
	
	float xx = kx*q.v.x;
	float yy = ky*q.v.y;
	float zz = kz*q.v.z;
	float xy = kx*q.v.y;
	float xz = kx*q.v.z;
	float yz = ky*q.v.z;
	float sx = kx*q.s;
	float sy = ky*q.s;
	float sz = kz*q.s;
	
	float3x3 m = {
		{ 1.0f - yy - zz, xy + sz, xz - sy },
		{ xy - sz, 1.0f - xx - zz, yz + sx },
		{ xz + sy, yz - sx, 1.0f - xx - yy },
	};
	return m;
}

static inline Transform operator * (Transform lhs, Transform rhs) {
	float3 p = make_rotation(lhs.rotation) * make_float3(rhs.position) + make_float3(lhs.position);
	Rotation q = make_rotation(lhs.rotation) * make_rotation(rhs.rotation);
	
	Transform r = {
		{ p.x, p.y, p.z },
		rhs.body,
		{ q.v.x, q.v.y, q.v.z, q.s },
	};
	return r;
}

static unsigned box_box_collide(uint32_t* pairs, unsigned pair_count, BoxCollider* colliders, Transform* transforms, Contact* contacts, BodyPair* bodies, uint64_t* tags, Arena temporary) {
	// TODO: We may want to batch/chunk this for better cache behavior for repeatedly accessed data.
	// TODO: We should make use of 8-wide SIMD here as well.
	
	float* feature_penetrations = allocate_array<float>(&temporary, pair_count + 7, 32); // Padding is required.
	uint32_t* features = allocate_array<uint32_t>(&temporary, pair_count + 7, 32);
	
	unsigned count = 0;
	
	// Determine most separating face and reject pairs separated by a face.
	{
		pairs[pair_count+0] = 0; // Padding.
		pairs[pair_count+1] = 0;
		pairs[pair_count+2] = 0;
		
		unsigned added = 0;
		
		// Transform each box into the local space of the other in order to quickly determine per-face penetration.
		for (unsigned i = 0; i < pair_count; i += 4) {
			// Load pairs.
			unsigned pair0 = pairs[i+0];
			unsigned pair1 = pairs[i+1];
			unsigned pair2 = pairs[i+2];
			unsigned pair3 = pairs[i+3];
			
			unsigned a0_index = pair0 & 0xffff;
			unsigned b0_index = pair0 >> 16;
			
			unsigned a1_index = pair1 & 0xffff;
			unsigned b1_index = pair1 >> 16;
			
			unsigned a2_index = pair2 & 0xffff;
			unsigned b2_index = pair2 >> 16;
			
			unsigned a3_index = pair3 & 0xffff;
			unsigned b3_index = pair3 >> 16;
			
			// Load rotations.
			simd4_float a_rotation_x = simd_float::load4(transforms[a0_index].rotation);
			simd4_float a_rotation_y = simd_float::load4(transforms[a1_index].rotation);
			simd4_float a_rotation_z = simd_float::load4(transforms[a2_index].rotation);
			simd4_float a_rotation_s = simd_float::load4(transforms[a3_index].rotation);
			
			simd4_float b_rotation_x = simd_float::load4(transforms[b0_index].rotation);
			simd4_float b_rotation_y = simd_float::load4(transforms[b1_index].rotation);
			simd4_float b_rotation_z = simd_float::load4(transforms[b2_index].rotation);
			simd4_float b_rotation_s = simd_float::load4(transforms[b3_index].rotation);
			
			simd128::transpose32(a_rotation_x, a_rotation_y, a_rotation_z, a_rotation_s);
			simd128::transpose32(b_rotation_x, b_rotation_y, b_rotation_z, b_rotation_s);
			
			// Determine quaternion for rotation from a to b.
			simd4_float t_x, t_y, t_z;
			simd_soa::cross(b_rotation_x, b_rotation_y, b_rotation_z, a_rotation_x, a_rotation_y, a_rotation_z, t_x, t_y, t_z);
			
			simd4_float relative_rotation_x = a_rotation_x * b_rotation_s - b_rotation_x * a_rotation_s - t_x;
			simd4_float relative_rotation_y = a_rotation_y * b_rotation_s - b_rotation_y * a_rotation_s - t_y;
			simd4_float relative_rotation_z = a_rotation_z * b_rotation_s - b_rotation_z * a_rotation_s - t_z;
			simd4_float relative_rotation_s = (a_rotation_x * b_rotation_x +
											   a_rotation_y * b_rotation_y +
											   a_rotation_z * b_rotation_z +
											   a_rotation_s * b_rotation_s);
			
			// Compute the corresponding matrix.
			// Note that the b to a matrix is simply the transpose of a to b.
			simd4_float kx = relative_rotation_x + relative_rotation_x;
			simd4_float ky = relative_rotation_y + relative_rotation_y;
			simd4_float kz = relative_rotation_z + relative_rotation_z;
			
			simd4_float xx = kx * relative_rotation_x;
			simd4_float yy = ky * relative_rotation_y;
			simd4_float zz = kz * relative_rotation_z;
			simd4_float xy = kx * relative_rotation_y;
			simd4_float xz = kx * relative_rotation_z;
			simd4_float yz = ky * relative_rotation_z;
			simd4_float sx = kx * relative_rotation_s;
			simd4_float sy = ky * relative_rotation_s;
			simd4_float sz = kz * relative_rotation_s;
			
			simd4_float one = simd_float::make4(1.0f);
			
			simd4_float vx_x = one - yy - zz;
			simd4_float vx_y = xy + sz;
			simd4_float vx_z = xz - sy;
			
			simd4_float vy_x = xy - sz;
			simd4_float vy_y = one - xx - zz;
			simd4_float vy_z = yz + sx;
			
			simd4_float vz_x = xz + sy;
			simd4_float vz_y = yz - sx;
			simd4_float vz_z = one - xx - yy;
			
			// Load sizes.
			simd4_float a_size_x = simd_float::load4(colliders[a0_index].size);
			simd4_float a_size_y = simd_float::load4(colliders[a1_index].size);
			simd4_float a_size_z = simd_float::load4(colliders[a2_index].size);
			simd4_float a_size_w = simd_float::load4(colliders[a3_index].size);
			
			simd4_float b_size_x = simd_float::load4(colliders[b0_index].size);
			simd4_float b_size_y = simd_float::load4(colliders[b1_index].size);
			simd4_float b_size_z = simd_float::load4(colliders[b2_index].size);
			simd4_float b_size_w = simd_float::load4(colliders[b3_index].size);
			
			simd128::transpose32(a_size_x, a_size_y, a_size_z, a_size_w);
			simd128::transpose32(b_size_x, b_size_y, b_size_z, b_size_w);
			
			// Compute the penetration.
			vx_x = simd_float::abs(vx_x);
			vx_y = simd_float::abs(vx_y);
			vx_z = simd_float::abs(vx_z);
			
			vy_x = simd_float::abs(vy_x);
			vy_y = simd_float::abs(vy_y);
			vy_z = simd_float::abs(vy_z);
			
			vz_x = simd_float::abs(vz_x);
			vz_y = simd_float::abs(vz_y);
			vz_z = simd_float::abs(vz_z);
			
			simd4_float pax = b_size_x + vx_x*a_size_x + vy_x*a_size_y + vz_x*a_size_z;
			simd4_float pay = b_size_y + vx_y*a_size_x + vy_y*a_size_y + vz_y*a_size_z;
			simd4_float paz = b_size_z + vx_z*a_size_x + vy_z*a_size_y + vz_z*a_size_z;
			
			simd4_float pbx = a_size_x + vx_x*b_size_x + vx_y*b_size_y + vx_z*b_size_z;
			simd4_float pby = a_size_y + vy_x*b_size_x + vy_y*b_size_y + vy_z*b_size_z;
			simd4_float pbz = a_size_z + vz_x*b_size_x + vz_y*b_size_y + vz_z*b_size_z;
			
			// Load positions.
			simd4_float a_position_x = simd_float::load4(transforms[a0_index].position);
			simd4_float a_position_y = simd_float::load4(transforms[a1_index].position);
			simd4_float a_position_z = simd_float::load4(transforms[a2_index].position);
			simd4_float a_position_w = simd_float::load4(transforms[a3_index].position);
			
			simd4_float b_position_x = simd_float::load4(transforms[b0_index].position);
			simd4_float b_position_y = simd_float::load4(transforms[b1_index].position);
			simd4_float b_position_z = simd_float::load4(transforms[b2_index].position);
			simd4_float b_position_w = simd_float::load4(transforms[b3_index].position);
			
			// Compute relative positions and offset the penetrations.
			simd4_float delta_x = a_position_x - b_position_x;
			simd4_float delta_y = a_position_y - b_position_y;
			simd4_float delta_z = a_position_z - b_position_z;
			simd4_float delta_w = a_position_w - b_position_w;
			
			simd128::transpose32(delta_x, delta_y, delta_z, delta_w);
			
			simd_soa::cross(b_rotation_x, b_rotation_y, b_rotation_z, delta_x, delta_y, delta_z, t_x, t_y, t_z);
			t_x += t_x;
			t_y += t_y;
			t_z += t_z;
			
			simd4_float u_x, u_y, u_z;
			simd_soa::cross(b_rotation_x, b_rotation_y, b_rotation_z, t_x, t_y, t_z, u_x, u_y, u_z);
			
			simd4_float a_offset_x = u_x + delta_x - b_rotation_s * t_x;
			simd4_float a_offset_y = u_y + delta_y - b_rotation_s * t_y;
			simd4_float a_offset_z = u_z + delta_z - b_rotation_s * t_z;
			
			pax -= simd_float::abs(a_offset_x);
			pay -= simd_float::abs(a_offset_y);
			paz -= simd_float::abs(a_offset_z);
			
			simd_soa::cross(delta_x, delta_y, delta_z, a_rotation_x, a_rotation_y, a_rotation_z, t_x, t_y, t_z);
			t_x += t_x;
			t_y += t_y;
			t_z += t_z;
			
			simd_soa::cross(a_rotation_x, a_rotation_y, a_rotation_z, t_x, t_y, t_z, u_x, u_y, u_z);
			
			simd4_float b_offset_x = u_x - delta_x - a_rotation_s * t_x;
			simd4_float b_offset_y = u_y - delta_y - a_rotation_s * t_y;
			simd4_float b_offset_z = u_z - delta_z - a_rotation_s * t_z;
			
			pbx -= simd_float::abs(b_offset_x);
			pby -= simd_float::abs(b_offset_y);
			pbz -= simd_float::abs(b_offset_z);
			
			// Reduce face penetrations.
			simd4_float payz = simd_float::min(pay, paz);
			simd4_float pbyz = simd_float::min(pby, pbz);
			
			simd4_float pa = simd_float::min(pax, payz);
			simd4_float pb = simd_float::min(pbx, pbyz);
			
			simd4_float p = simd_float::min(pa, pb);
			
			// Determine the best aligned face for each collider.
			simd4_float aymf = simd_float::cmp_eq(payz, pa);
			simd4_float azmf = simd_float::cmp_eq(paz, pa);
			
			simd4_float bymf = simd_float::cmp_eq(pbyz, pb);
			simd4_float bzmf = simd_float::cmp_eq(pbz, pb);
			
			simd4_int32 aymi = simd::bitwise_and(simd_float::asint(aymf), simd_int32::make4(1));
			simd4_int32 azmi = simd::bitwise_and(simd_float::asint(azmf), simd_int32::make4(1));
			
			simd4_int32 bymi = simd::bitwise_and(simd_float::asint(bymf), simd_int32::make4(1));
			simd4_int32 bzmi = simd::bitwise_and(simd_float::asint(bzmf), simd_int32::make4(1));
			
			simd4_int32 aface = simd_int32::add(aymi, azmi);
			simd4_int32 bface = simd_int32::add(bymi, bzmi);
			
			// Swap so that collider a has the most separating face.
			simd4_float swap = simd_float::cmp_eq(pa, p);
			
			simd4_float pair_a_b = simd_int32::asfloat(simd_int32::load4((const int32_t*)(pairs + i)));
			simd4_float pair_b_a = simd_int32::asfloat(simd::bitwise_or(simd_int32::shift_left<16>(simd_float::asint(pair_a_b)), simd_int32::shift_right<16>(simd_float::asint(pair_a_b))));
			
			simd4_float face = simd::blendv32(simd_int32::asfloat(bface), simd_int32::asfloat(aface), swap);
			simd4_float pair = simd::blendv32(pair_a_b, pair_b_a, swap);
			
			// Store data for pairs with positive penetration.
			unsigned mask = simd::signmask32(simd_float::cmp_gt(p, simd_float::zero4()));
			
			NUDGE_ALIGNED(16) float face_penetration_array[4];
			NUDGE_ALIGNED(16) uint32_t face_array[4];
			NUDGE_ALIGNED(16) uint32_t pair_array[4];
			
			simd_float::store4(face_penetration_array, p);
			simd_float::store4((float*)face_array, face);
			simd_float::store4((float*)pair_array, pair);
			
			while (mask) {
				unsigned index = first_set_bit(mask);
				mask &= mask-1;
				
				feature_penetrations[added] = face_penetration_array[index];
				features[added] = face_array[index];
				pairs[added] = pair_array[index];
				
				++added;
			}
		}
		
		// Erase padding.
		while (added && !pairs[added-1])
			--added;
		
		pair_count = added;
	}
	
	// Check if edge pairs are more separating.
	// Do face-face test if not.
	{
		pairs[pair_count+0] = 0; // Padding.
		pairs[pair_count+1] = 0;
		pairs[pair_count+2] = 0;
		
		feature_penetrations[pair_count+0] = 0.0f;
		feature_penetrations[pair_count+1] = 0.0f;
		feature_penetrations[pair_count+2] = 0.0f;
		
		unsigned added = 0;
		
		for (unsigned pair_offset = 0; pair_offset < pair_count; pair_offset += 4) {
			// Load pairs.
			unsigned pair0 = pairs[pair_offset+0];
			unsigned pair1 = pairs[pair_offset+1];
			unsigned pair2 = pairs[pair_offset+2];
			unsigned pair3 = pairs[pair_offset+3];
			
			unsigned a0_index = pair0 & 0xffff;
			unsigned b0_index = pair0 >> 16;
			
			unsigned a1_index = pair1 & 0xffff;
			unsigned b1_index = pair1 >> 16;
			
			unsigned a2_index = pair2 & 0xffff;
			unsigned b2_index = pair2 >> 16;
			
			unsigned a3_index = pair3 & 0xffff;
			unsigned b3_index = pair3 >> 16;
			
			// Load rotations.
			simd4_float a_rotation_x = simd_float::load4(transforms[a0_index].rotation);
			simd4_float a_rotation_y = simd_float::load4(transforms[a1_index].rotation);
			simd4_float a_rotation_z = simd_float::load4(transforms[a2_index].rotation);
			simd4_float a_rotation_s = simd_float::load4(transforms[a3_index].rotation);
			
			simd4_float b_rotation_x = simd_float::load4(transforms[b0_index].rotation);
			simd4_float b_rotation_y = simd_float::load4(transforms[b1_index].rotation);
			simd4_float b_rotation_z = simd_float::load4(transforms[b2_index].rotation);
			simd4_float b_rotation_s = simd_float::load4(transforms[b3_index].rotation);
			
			simd128::transpose32(a_rotation_x, a_rotation_y, a_rotation_z, a_rotation_s);
			simd128::transpose32(b_rotation_x, b_rotation_y, b_rotation_z, b_rotation_s);
			
			// Determine quaternion for rotation from a to b.
			simd4_float t_x, t_y, t_z;
			simd_soa::cross(b_rotation_x, b_rotation_y, b_rotation_z, a_rotation_x, a_rotation_y, a_rotation_z, t_x, t_y, t_z);
			
			simd4_float relative_rotation_x = a_rotation_x * b_rotation_s - b_rotation_x * a_rotation_s - t_x;
			simd4_float relative_rotation_y = a_rotation_y * b_rotation_s - b_rotation_y * a_rotation_s - t_y;
			simd4_float relative_rotation_z = a_rotation_z * b_rotation_s - b_rotation_z * a_rotation_s - t_z;
			simd4_float relative_rotation_s = (a_rotation_x * b_rotation_x +
											   a_rotation_y * b_rotation_y +
											   a_rotation_z * b_rotation_z +
											   a_rotation_s * b_rotation_s);
			
			// Compute the corresponding matrix.
			// Note that the b to a matrix is simply the transpose of a to b.
			simd4_float kx = relative_rotation_x + relative_rotation_x;
			simd4_float ky = relative_rotation_y + relative_rotation_y;
			simd4_float kz = relative_rotation_z + relative_rotation_z;
			
			simd4_float xx = kx * relative_rotation_x;
			simd4_float yy = ky * relative_rotation_y;
			simd4_float zz = kz * relative_rotation_z;
			simd4_float xy = kx * relative_rotation_y;
			simd4_float xz = kx * relative_rotation_z;
			simd4_float yz = ky * relative_rotation_z;
			simd4_float sx = kx * relative_rotation_s;
			simd4_float sy = ky * relative_rotation_s;
			simd4_float sz = kz * relative_rotation_s;
			
			simd4_float one = simd_float::make4(1.0f);
			
			simd4_float vx_x = one - yy - zz;
			simd4_float vx_y = xy + sz;
			simd4_float vx_z = xz - sy;
			
			simd4_float vy_x = xy - sz;
			simd4_float vy_y = one - xx - zz;
			simd4_float vy_z = yz + sx;
			
			simd4_float vz_x = xz + sy;
			simd4_float vz_y = yz - sx;
			simd4_float vz_z = one - xx - yy;
			
			NUDGE_ALIGNED(16) float a_to_b[4*9];
			
			simd_float::store4(a_to_b + 0, vx_x);
			simd_float::store4(a_to_b + 4, vx_y);
			simd_float::store4(a_to_b + 8, vx_z);
			
			simd_float::store4(a_to_b + 12, vy_x);
			simd_float::store4(a_to_b + 16, vy_y);
			simd_float::store4(a_to_b + 20, vy_z);
			
			simd_float::store4(a_to_b + 24, vz_x);
			simd_float::store4(a_to_b + 28, vz_y);
			simd_float::store4(a_to_b + 32, vz_z);
			
			// Load sizes.
			simd4_float a_size_x = simd_float::load4(colliders[a0_index].size);
			simd4_float a_size_y = simd_float::load4(colliders[a1_index].size);
			simd4_float a_size_z = simd_float::load4(colliders[a2_index].size);
			simd4_float a_size_w = simd_float::load4(colliders[a3_index].size);
			
			simd4_float b_size_x = simd_float::load4(colliders[b0_index].size);
			simd4_float b_size_y = simd_float::load4(colliders[b1_index].size);
			simd4_float b_size_z = simd_float::load4(colliders[b2_index].size);
			simd4_float b_size_w = simd_float::load4(colliders[b3_index].size);
			
			simd128::transpose32(a_size_x, a_size_y, a_size_z, a_size_w);
			simd128::transpose32(b_size_x, b_size_y, b_size_z, b_size_w);
			
			// Load positions.
			simd4_float a_position_x = simd_float::load4(transforms[a0_index].position);
			simd4_float a_position_y = simd_float::load4(transforms[a1_index].position);
			simd4_float a_position_z = simd_float::load4(transforms[a2_index].position);
			simd4_float a_position_w = simd_float::load4(transforms[a3_index].position);
			
			simd4_float b_position_x = simd_float::load4(transforms[b0_index].position);
			simd4_float b_position_y = simd_float::load4(transforms[b1_index].position);
			simd4_float b_position_z = simd_float::load4(transforms[b2_index].position);
			simd4_float b_position_w = simd_float::load4(transforms[b3_index].position);
			
			// Compute relative positions and offset the penetrations.
			simd4_float delta_x = a_position_x - b_position_x;
			simd4_float delta_y = a_position_y - b_position_y;
			simd4_float delta_z = a_position_z - b_position_z;
			simd4_float delta_w = a_position_w - b_position_w;
			
			simd128::transpose32(delta_x, delta_y, delta_z, delta_w);
			
			simd_soa::cross(delta_x, delta_y, delta_z, a_rotation_x, a_rotation_y, a_rotation_z, t_x, t_y, t_z);
			t_x += t_x;
			t_y += t_y;
			t_z += t_z;
			
			simd4_float u_x, u_y, u_z;
			simd_soa::cross(a_rotation_x, a_rotation_y, a_rotation_z, t_x, t_y, t_z, u_x, u_y, u_z);
			
			simd4_float b_offset_x = u_x - delta_x - a_rotation_s * t_x;
			simd4_float b_offset_y = u_y - delta_y - a_rotation_s * t_y;
			simd4_float b_offset_z = u_z - delta_z - a_rotation_s * t_z;
			
			NUDGE_ALIGNED(16) float b_offset_array[3*4];
			
			simd_float::store4(b_offset_array + 0, b_offset_x);
			simd_float::store4(b_offset_array + 4, b_offset_y);
			simd_float::store4(b_offset_array + 8, b_offset_z);
			
			simd4_float face_penetration = simd_float::load4(feature_penetrations + pair_offset);
			
			// Is an edge pair more separating?
			NUDGE_ALIGNED(16) float edge_penetration_a[4*9];
			NUDGE_ALIGNED(16) float edge_penetration_b[4*9];
			
			for (unsigned i = 0; i < 3; ++i) {
				simd4_float acx = simd_float::load4(a_to_b + (0*3 + i)*4);
				simd4_float acy = simd_float::load4(a_to_b + (1*3 + i)*4);
				simd4_float acz = simd_float::load4(a_to_b + (2*3 + i)*4);
				
				simd4_float bcx = simd_float::load4(a_to_b + (i*3 + 0)*4);
				simd4_float bcy = simd_float::load4(a_to_b + (i*3 + 1)*4);
				simd4_float bcz = simd_float::load4(a_to_b + (i*3 + 2)*4);
				
				simd4_float ac2x = acx*acx;
				simd4_float ac2y = acy*acy;
				simd4_float ac2z = acz*acz;
				
				simd4_float bc2x = bcx*bcx;
				simd4_float bc2y = bcy*bcy;
				simd4_float bc2z = bcz*bcz;
				
				simd4_float aacx = simd_float::abs(acx);
				simd4_float aacy = simd_float::abs(acy);
				simd4_float aacz = simd_float::abs(acz);
				
				simd4_float abcx = simd_float::abs(bcx);
				simd4_float abcy = simd_float::abs(bcy);
				simd4_float abcz = simd_float::abs(bcz);
				
				simd4_float r_a0 = ac2y + ac2z;
				simd4_float r_a1 = ac2z + ac2x;
				simd4_float r_a2 = ac2x + ac2y;
				
				simd4_float r_b0 = bc2y + bc2z;
				simd4_float r_b1 = bc2z + bc2x;
				simd4_float r_b2 = bc2x + bc2y;
				
				simd4_float nan_threshold = simd_float::make4(1e-3f);
				
				r_a0 = simd::bitwise_or(simd_float::rsqrt(r_a0), simd_float::cmp_le(r_a0, nan_threshold));
				r_a1 = simd::bitwise_or(simd_float::rsqrt(r_a1), simd_float::cmp_le(r_a1, nan_threshold));
				r_a2 = simd::bitwise_or(simd_float::rsqrt(r_a2), simd_float::cmp_le(r_a2, nan_threshold));
				
				r_b0 = simd::bitwise_or(simd_float::rsqrt(r_b0), simd_float::cmp_le(r_b0, nan_threshold));
				r_b1 = simd::bitwise_or(simd_float::rsqrt(r_b1), simd_float::cmp_le(r_b1, nan_threshold));
				r_b2 = simd::bitwise_or(simd_float::rsqrt(r_b2), simd_float::cmp_le(r_b2, nan_threshold));
				
				simd4_float pa0 = aacy*a_size_z + aacz*a_size_y;
				simd4_float pa1 = aacz*a_size_x + aacx*a_size_z;
				simd4_float pa2 = aacx*a_size_y + aacy*a_size_x;
				
				simd4_float pb0 = abcy*b_size_z + abcz*b_size_y;
				simd4_float pb1 = abcz*b_size_x + abcx*b_size_z;
				simd4_float pb2 = abcx*b_size_y + abcy*b_size_x;
				
				simd4_float o0 = simd_float::abs(acy*b_offset_z - acz*b_offset_y);
				simd4_float o1 = simd_float::abs(acz*b_offset_x - acx*b_offset_z);
				simd4_float o2 = simd_float::abs(acx*b_offset_y - acy*b_offset_x);
				
				simd_float::store4(edge_penetration_a + (i*3 + 0)*4, (pa0 - o0) * r_a0);
				simd_float::store4(edge_penetration_a + (i*3 + 1)*4, (pa1 - o1) * r_a1);
				simd_float::store4(edge_penetration_a + (i*3 + 2)*4, (pa2 - o2) * r_a2);
				
				simd_float::store4(edge_penetration_b + (i*3 + 0)*4, pb0 * r_b0);
				simd_float::store4(edge_penetration_b + (i*3 + 1)*4, pb1 * r_b1);
				simd_float::store4(edge_penetration_b + (i*3 + 2)*4, pb2 * r_b2);
			}
			
			simd4_int32 a_edge = simd_int32::make4(0);
			simd4_int32 b_edge = simd_int32::make4(0);
			
			simd4_float penetration = face_penetration;
			
			for (unsigned i = 0; i < 3; ++i) {
				for (unsigned j = 0; j < 3; ++j) {
					simd4_float p = simd_float::load4(edge_penetration_a + (i*3 + j)*4) + simd_float::load4(edge_penetration_b + (j*3 + i)*4);
					
					simd4_float mask = simd_float::cmp_gt(penetration, p);
					
					penetration = simd_float::min(penetration, p); // Note: First operand is returned on NaN.
					a_edge = simd::blendv32(a_edge, simd_int32::make4(j), simd_float::asint(mask));
					b_edge = simd::blendv32(b_edge, simd_int32::make4(i), simd_float::asint(mask));
				}
			}
			
			simd4_float face_bias = simd_float::make4(1e-3f);
			
			unsigned edge = simd::signmask32(simd_float::cmp_gt(face_penetration, penetration + face_bias));
			unsigned overlapping = simd::signmask32(simd_float::cmp_gt(penetration, simd_float::zero4()));
			
			unsigned face = ~edge;
			
			edge &= overlapping;
			face &= overlapping;
			
			NUDGE_ALIGNED(16) float penetration_array[4];
			NUDGE_ALIGNED(16) int32_t a_edge_array[4];
			NUDGE_ALIGNED(16) int32_t b_edge_array[4];
			
			simd_float::store4(penetration_array, penetration);
			simd_int32::store4(a_edge_array, a_edge);
			simd_int32::store4(b_edge_array, b_edge);
			
			// Do face-face tests.
			while (face) {
				unsigned index = first_set_bit(face);
				face &= face-1;
				
				unsigned pair = pairs[pair_offset + index];
				unsigned a_face = features[pair_offset + index];
				
				unsigned a_index = pair & 0xffff;
				unsigned b_index = pair >> 16;
				
				// Gather.
				simd4_float dirs = simd_float::make4(a_to_b[(a_face*3 + 0)*4 + index],
													 a_to_b[(a_face*3 + 1)*4 + index],
													 a_to_b[(a_face*3 + 2)*4 + index],
													 0.0f);
				
				simd4_float c0 = simd_float::make4(a_to_b[(0*3 + 0)*4 + index],
												   a_to_b[(1*3 + 0)*4 + index],
												   a_to_b[(2*3 + 0)*4 + index],
												   0.0f);
				
				simd4_float c1 = simd_float::make4(a_to_b[(0*3 + 1)*4 + index],
												   a_to_b[(1*3 + 1)*4 + index],
												   a_to_b[(2*3 + 1)*4 + index],
												   0.0f);
				
				simd4_float c2 = simd_float::make4(a_to_b[(0*3 + 2)*4 + index],
												   a_to_b[(1*3 + 2)*4 + index],
												   a_to_b[(2*3 + 2)*4 + index],
												   0.0f);
				
				simd4_float b_offset = simd_float::make4(b_offset_array[0*4 + index],
														 b_offset_array[1*4 + index],
														 b_offset_array[2*4 + index],
														 0.0f);
				
				// Load sizes.
				simd4_float a_size = simd_float::load4(colliders[a_index].size);
				simd4_float b_size = simd_float::load4(colliders[b_index].size);
				
				// Find most aligned face of b.
				dirs = simd_float::abs(dirs);
				
				simd4_float max_dir = simd_float::max(simd128::shuffle32<0,2,1,3>(dirs), simd128::shuffle32<0,0,0,0>(dirs));
				
				unsigned dir_mask = simd::signmask32(simd_float::cmp_ge(dirs, max_dir));
				
				// Compute the coordinates of the two quad faces.
				c0 *= simd128::shuffle32<0,0,0,0>(b_size);
				c1 *= simd128::shuffle32<1,1,1,1>(b_size);
				c2 *= simd128::shuffle32<2,2,2,2>(b_size);
				
				unsigned b_face = 0;
				
				if (dir_mask & 4) {
					simd4_float t = c0;
					c0 = c2;
					c2 = c1;
					c1 = t;
					b_face = 2;
				}
				else if (dir_mask & 2) {
					simd4_float t = c0;
					c0 = c1;
					c1 = c2;
					c2 = t;
					b_face = 1;
				}
				
				simd4_float c = c0;
				simd4_float dx = c1;
				simd4_float dy = c2;
				
				unsigned b_positive_face_bit = simd::signmask32(simd::bitwise_xor(b_offset, c)) & (1 << a_face);
				unsigned b_offset_neg = simd::signmask32(b_offset) & (1 << a_face);
				
				if (!b_positive_face_bit)
					c = -c;
				
				c += b_offset;
				
				// Quad coordinate packing:
				// Size of quad a, center of quad b, x-axis of quad b, y-axis of quad b.
				// a.size.x, c.x, dx.x, dy.x
				// a.size.y, c.y, dx.y, dy.y
				// a.size.z, c.z, dx.z, dy.z
				NUDGE_ALIGNED(16) float quads[4*3];
				
				simd4_float q0 = simd128::unpacklo32(a_size, c);
				simd4_float q1 = simd128::unpackhi32(a_size, c);
				simd4_float q2 = simd128::unpacklo32(dx, dy);
				simd4_float q3 = simd128::unpackhi32(dx, dy);
				
				simd_float::store4(quads + 0, simd128::concat2x32<0,1,0,1>(q0, q2));
				simd_float::store4(quads + 4, simd128::concat2x32<2,3,2,3>(q0, q2));
				simd_float::store4(quads + 8, simd128::concat2x32<0,1,0,1>(q1, q3));
				
				// Transform so that overlap testing can be done in two dimensions.
				const float* transformed_x = quads + 4*((a_face+1) % 3);
				const float* transformed_y = quads + 4*((a_face+2) % 3);
				const float* transformed_z = quads + 4*a_face;
				
				// Find support points for the overlap between the quad faces in two dimensions.
				NUDGE_ALIGNED(32) float support[16*3];
				NUDGE_ALIGNED(32) uint32_t support_tags[16];
				unsigned mask; // Indicates valid points.
				{
					float* support_x = support + 0;
					float* support_y = support + 16;
					
					simd4_float tx = simd_float::load4(transformed_x);
					simd4_float ty = simd_float::load4(transformed_y);
					
					simd4_float sxycxy = simd128::unpacklo32(tx, ty);
					simd4_float dxy = simd128::unpackhi32(tx, ty);
					
					simd4_float sx = simd128::shuffle32<0,0,0,0>(sxycxy);
					simd4_float sy = simd128::shuffle32<1,1,1,1>(sxycxy);
					simd4_float cx = simd128::shuffle32<2,2,2,2>(sxycxy);
					simd4_float cy = simd128::shuffle32<3,3,3,3>(sxycxy);
					
					simd4_float sign_npnp = simd_float::make4(-0.0f, 0.0f, -0.0f, 0.0f);
					
					// Add corner points to the support if they are part of the intersection.
					__m128i corner_mask;
					__m128i edge_mask;
					{
						simd4_float sign_pnpn = simd_float::make4(0.0f, -0.0f, 0.0f, -0.0f);
						simd4_float sign_nnpp = simd_float::make4(-0.0f, -0.0f, 0.0f, 0.0f);
						
						simd4_float corner0x = simd::bitwise_xor(sx, sign_pnpn);
						simd4_float corner0y = simd::bitwise_xor(sy, sign_nnpp);
						
						simd4_float corner1x = cx + simd::bitwise_xor(simd128::shuffle32<0,0,0,0>(dxy), sign_npnp) + simd::bitwise_xor(simd128::shuffle32<2,2,2,2>(dxy), sign_nnpp);
						simd4_float corner1y = cy + simd::bitwise_xor(simd128::shuffle32<1,1,1,1>(dxy), sign_npnp) + simd::bitwise_xor(simd128::shuffle32<3,3,3,3>(dxy), sign_nnpp);
						
						simd4_float k = (simd128::concat2x32<2,2,0,0>(sxycxy, dxy) * simd128::shuffle32<3,1,3,1>(dxy) -
										 simd128::concat2x32<3,3,1,1>(sxycxy, dxy) * simd128::shuffle32<2,0,2,0>(dxy));
						
						simd4_float ox = simd128::shuffle32<0,0,0,0>(k);
						simd4_float oy = simd128::shuffle32<1,1,1,1>(k);
						simd4_float delta_max = simd_float::abs(simd128::shuffle32<2,2,2,2>(k));
						
						simd4_float sdxy = dxy * simd128::shuffle32<1,0,1,0>(sxycxy);
						
						simd4_float delta_x = ox + simd::bitwise_xor(simd128::shuffle32<2,2,2,2>(sdxy), sign_nnpp) + simd::bitwise_xor(simd128::shuffle32<3,3,3,3>(sdxy), sign_npnp);
						simd4_float delta_y = oy + simd::bitwise_xor(simd128::shuffle32<0,0,0,0>(sdxy), sign_nnpp) + simd::bitwise_xor(simd128::shuffle32<1,1,1,1>(sdxy), sign_npnp);
						
						simd4_float inside_x = simd_float::cmp_le(simd_float::abs(corner1x), sx);
						simd4_float inside_y = simd_float::cmp_le(simd_float::abs(corner1y), sy);
						
						simd4_float mask0 = simd_float::cmp_le(simd_float::max(simd_float::abs(delta_x), simd_float::abs(delta_y)), delta_max);
						simd4_float mask1 = simd::bitwise_and(inside_x, inside_y);
						
						corner_mask = _mm_packs_epi32(simd_float::asint(mask0), simd_float::asint(mask1));
						
						// Don't allow edge intersections if both vertices are inside.
						edge_mask = _mm_packs_epi32(simd_float::asint(simd::bitwise_and(simd128::shuffle32<3,2,0,2>(mask0), simd128::shuffle32<1,0,1,3>(mask0))),
													simd_float::asint(simd::bitwise_and(simd128::shuffle32<1,3,2,3>(mask1), simd128::shuffle32<0,2,0,1>(mask1))));
						
						simd_float::store4(support_x + 0, corner0x);
						simd_float::store4(support_y + 0, corner0y);
						simd_float::store4(support_x + 4, corner1x);
						simd_float::store4(support_y + 4, corner1y);
					}
					
					// Find additional support points by intersecting the edges of the second quad against the bounds of the first.
					unsigned edge_axis_near;
					unsigned edge_axis_far;
					{
						simd4_float one = simd_float::make4(1.0f);
						simd4_float rdxy = one/dxy;
						
						simd4_float offset_x = simd128::shuffle32<0,0,2,2>(dxy);
						simd4_float offset_y = simd128::shuffle32<1,1,3,3>(dxy);
						
						simd4_float pivot_x = cx + simd::bitwise_xor(simd128::shuffle32<2,2,0,0>(dxy), sign_npnp);
						simd4_float pivot_y = cy + simd::bitwise_xor(simd128::shuffle32<3,3,1,1>(dxy), sign_npnp);
						
						simd4_float sign_mask = simd_float::make4(-0.0f);
						simd4_float pos_x = simd::bitwise_or(simd::bitwise_and(offset_x, sign_mask), sx); // Copy sign.
						simd4_float pos_y = simd::bitwise_or(simd::bitwise_and(offset_y, sign_mask), sy);
						
						simd4_float rx = simd128::shuffle32<0,0,2,2>(rdxy);
						simd4_float ry = simd128::shuffle32<1,1,3,3>(rdxy);
						
						simd4_float near_x = (pos_x + pivot_x) * rx;
						simd4_float far_x = (pos_x - pivot_x) * rx;
						
						simd4_float near_y = (pos_y + pivot_y) * ry;
						simd4_float far_y = (pos_y - pivot_y) * ry;
						
						simd4_float a = simd_float::min(one, near_x); // First operand is returned on NaN.
						simd4_float b = simd_float::min(one, far_x);
						
						edge_axis_near = simd::signmask32(simd_float::cmp_gt(a, near_y));
						edge_axis_far = simd::signmask32(simd_float::cmp_gt(b, far_y));
						
						a = simd_float::min(a, near_y);
						b = simd_float::min(b, far_y);
						
						simd4_float ax = pivot_x - offset_x * a;
						simd4_float ay = pivot_y - offset_y * a;
						simd4_float bx = pivot_x + offset_x * b;
						simd4_float by = pivot_y + offset_y * b;
						
						simd4_float mask = simd_float::cmp_gt(a + b, simd_float::zero4()); // Make sure -a < b.
						
						simd4_float mask_a = simd_float::cmp_neq(a, one);
						simd4_float mask_b = simd_float::cmp_neq(b, one);
						
						mask_a = simd::bitwise_and(mask_a, mask);
						mask_b = simd::bitwise_and(mask_b, mask);
						
						edge_mask = simd::bitwise_notand(edge_mask, _mm_packs_epi32(simd_float::asint(mask_a), simd_float::asint(mask_b)));
						
						simd_float::store4(support_x + 8, ax);
						simd_float::store4(support_y + 8, ay);
						simd_float::store4(support_x + 12, bx);
						simd_float::store4(support_y + 12, by);
					}
					
					mask = _mm_movemask_epi8(_mm_packs_epi16(corner_mask, edge_mask));
					
					// Calculate and store vertex labels.
					// The 8 vertices are tagged using the sign bit of each axis.
					// Bit rotation is used to "transform" the coordinates.
					unsigned a_sign_face_bit = b_offset_neg ? (1 << a_face) : 0;
					unsigned b_sign_face_bit = b_positive_face_bit ? 0 : (1 << b_face);
					
					unsigned a_vertices = 0x12003624 >> (3 - a_face); // Rotates all vertices in parallel.
					unsigned b_vertices = 0x00122436 >> (3 - b_face);
					
					unsigned a_face_bits = 0xffff0000 | a_sign_face_bit;
					unsigned b_face_bits = 0x0000ffff | (b_sign_face_bit << 16);
					
					support_tags[0] = ((a_vertices >>  0) & 0x7) | a_face_bits;
					support_tags[1] = ((a_vertices >>  8) & 0x7) | a_face_bits;
					support_tags[2] = ((a_vertices >> 16) & 0x7) | a_face_bits;
					support_tags[3] = ((a_vertices >> 24) & 0x7) | a_face_bits;
					
					support_tags[4] = ((b_vertices << 16) & 0x70000) | b_face_bits;
					support_tags[5] = ((b_vertices <<  8) & 0x70000) | b_face_bits;
					support_tags[6] = ((b_vertices >>  0) & 0x70000) | b_face_bits;
					support_tags[7] = ((b_vertices >>  8) & 0x70000) | b_face_bits;
					
					// Calculate edge numbers in the local coordinate frame.
					unsigned edge_axis_winding = simd::signmask32(dxy);
					
					unsigned y_near0 = (edge_axis_near >> 0) & 1;
					unsigned y_near1 = (edge_axis_near >> 1) & 1;
					unsigned y_near2 = (edge_axis_near >> 2) & 1;
					unsigned y_near3 = (edge_axis_near >> 3) & 1;
					
					unsigned y_far0 = (edge_axis_far >> 0) & 1;
					unsigned y_far1 = (edge_axis_far >> 1) & 1;
					unsigned y_far2 = (edge_axis_far >> 2) & 1;
					unsigned y_far3 = (edge_axis_far >> 3) & 1;
					
					unsigned a_near_edge0 = y_near0*2 + ((edge_axis_winding >> (0 + y_near0)) & 1);
					unsigned a_near_edge1 = y_near1*2 + ((edge_axis_winding >> (0 + y_near1)) & 1);
					unsigned a_near_edge2 = y_near2*2 + ((edge_axis_winding >> (2 + y_near2)) & 1);
					unsigned a_near_edge3 = y_near3*2 + ((edge_axis_winding >> (2 + y_near3)) & 1);
					
					edge_axis_winding ^= 0xf;
					
					unsigned a_far_edge0 = y_far0*2 + ((edge_axis_winding >> (0 + y_far0)) & 1);
					unsigned a_far_edge1 = y_far1*2 + ((edge_axis_winding >> (0 + y_far1)) & 1);
					unsigned a_far_edge2 = y_far2*2 + ((edge_axis_winding >> (2 + y_far2)) & 1);
					unsigned a_far_edge3 = y_far3*2 + ((edge_axis_winding >> (2 + y_far3)) & 1);
					
					// Map local edges to labels (so that faces can share an edge).
					// The 12 edges are tagged using two ordered points.
					// We use the same trick as the vertex transform but do it for pairs of vertices (in correct order).
					uint64_t a_edge_map = 0x1200362424003612llu >> (3 - a_face);
					uint64_t b_edge_map = 0x2400361212003624llu >> (3 - b_face);
					
					unsigned face_bits = a_sign_face_bit | (a_sign_face_bit << 8) | (b_sign_face_bit << 16) | (b_sign_face_bit << 24);
					
					unsigned b_edge0 = ((unsigned)((b_edge_map >> (0<<4)) & 0x0707) << 16) | face_bits;
					unsigned b_edge1 = ((unsigned)((b_edge_map >> (1<<4)) & 0x0707) << 16) | face_bits;
					unsigned b_edge2 = ((unsigned)((b_edge_map >> (2<<4)) & 0x0707) << 16) | face_bits;
					unsigned b_edge3 = ((unsigned)((b_edge_map >> (3<<4)) & 0x0707) << 16) | face_bits;
					
					support_tags[ 8] = (unsigned)((a_edge_map >> (a_near_edge0<<4)) & 0x0707) | b_edge0;
					support_tags[ 9] = (unsigned)((a_edge_map >> (a_near_edge1<<4)) & 0x0707) | b_edge1;
					support_tags[10] = (unsigned)((a_edge_map >> (a_near_edge2<<4)) & 0x0707) | b_edge2;
					support_tags[11] = (unsigned)((a_edge_map >> (a_near_edge3<<4)) & 0x0707) | b_edge3;
					
					support_tags[12] = (unsigned)((a_edge_map >> (a_far_edge0<<4)) & 0x0707) | b_edge0;
					support_tags[13] = (unsigned)((a_edge_map >> (a_far_edge1<<4)) & 0x0707) | b_edge1;
					support_tags[14] = (unsigned)((a_edge_map >> (a_far_edge2<<4)) & 0x0707) | b_edge2;
					support_tags[15] = (unsigned)((a_edge_map >> (a_far_edge3<<4)) & 0x0707) | b_edge3;
				}
				
				// Compute z-plane through face b and calculate z for the support points.
				simd4_float a_size_transformed = simd_float::load4(transformed_x);
				simd4_float c_transformed = simd_float::load4(transformed_y);
				simd4_float dx_transformed = simd_float::load4(transformed_z);
				simd4_float dy_transformed = simd_float::zero4();
				
				simd128::transpose32(a_size_transformed, c_transformed, dx_transformed, dy_transformed);
				
				simd4_float zn = simd_aos::cross(dx_transformed, dy_transformed);
				simd4_float plane = simd128::concat2x32<0,1,0,1>(simd::bitwise_xor(zn, simd_float::make4(-0.0f)), simd_aos::dot(c_transformed, zn));
				plane *= simd_float::make4(1.0f)/simd128::shuffle32<2,2,2,2>(zn);
				
				NUDGE_ALIGNED(32) float penetrations[16];
				
				simdv_float z_sign = simd_float::zerov();
				
				if (b_offset_neg)
					z_sign = simd_float::makev(-0.0f);
				
#if NUDGE_SIMDV_WIDTH == 256
				simdv_float penetration_offset = simd256::broadcast(simd128::shuffle32<2,2,2,2>(a_size_transformed));
				simdv_float plane256 = simd256::broadcast(plane);
#else
				simdv_float penetration_offset = simd128::shuffle32<2,2,2,2>(a_size_transformed);
#endif
				unsigned penetration_mask = 0;
				
				for (unsigned i = 0; i < 16; i += simdv_width32) {
#if NUDGE_SIMDV_WIDTH == 256
					simdv_float plane = plane256;
#endif
					
					simdv_float x = simd_float::loadv(support + 0 + i);
					simdv_float y = simd_float::loadv(support + 16 + i);
					simdv_float z = x*simd128::shuffle32<0,0,0,0>(plane) + y*simd128::shuffle32<1,1,1,1>(plane) + simd128::shuffle32<2,2,2,2>(plane);
					
					simdv_float penetration = penetration_offset - simd::bitwise_xor(z, z_sign);
					
					z += penetration * simd::bitwise_xor(simd_float::makev(0.5f), z_sign);
					
					penetration_mask |= simd::signmask32(simd_float::cmp_gt(penetration, simd_float::zerov())) << i;
					
					simd_float::storev(penetrations + i, penetration);
					simd_float::storev(support + 32 + i, z);
				}
				
				mask &= penetration_mask;
				
				// Inverse transform.
				unsigned a_face_inverse = (a_face ^ 1) ^ (a_face >> 1);
				
				const float* support_x = support + 16*((a_face_inverse+1) % 3);
				const float* support_y = support + 16*((a_face_inverse+2) % 3);
				const float* support_z = support + 16*a_face_inverse;
				
				// Setup rotation matrix from a to world.
				simd4_float a_to_world0, a_to_world1, a_to_world2;
				{
					simd4_float qx_qy_qz_qs = simd_float::load4(transforms[a_index].rotation);
					simd4_float kx_ky_kz_ks = qx_qy_qz_qs + qx_qy_qz_qs;
					
					// Make ks negative so that we can create +sx from kx*qs and -sx from ks*qx.
					kx_ky_kz_ks = simd::bitwise_xor(kx_ky_kz_ks, simd_float::make4(0.0f, 0.0f, 0.0f, -0.0f));
					
					//  1.0f - yy - zz, xy + sz, xz - sy
					a_to_world0 = (simd128::shuffle32<1,0,0,3>(kx_ky_kz_ks) * simd128::shuffle32<1,1,2,3>(qx_qy_qz_qs) +
								   simd128::shuffle32<2,2,3,3>(kx_ky_kz_ks) * simd128::shuffle32<2,3,1,3>(qx_qy_qz_qs));
					
					// xy - sz, 1.0f - zz - xx, yz + sx
					a_to_world1 = (simd128::shuffle32<0,2,1,3>(kx_ky_kz_ks) * simd128::shuffle32<1,2,2,3>(qx_qy_qz_qs) +
								   simd128::shuffle32<3,0,0,3>(kx_ky_kz_ks) * simd128::shuffle32<2,0,3,3>(qx_qy_qz_qs));
					
					// xz + sy, yz - sx, 1.0f - xx - yy
					a_to_world2 = (simd128::shuffle32<0,1,0,3>(kx_ky_kz_ks) * simd128::shuffle32<2,2,0,3>(qx_qy_qz_qs) +
								   simd128::shuffle32<1,3,1,3>(kx_ky_kz_ks) * simd128::shuffle32<3,0,1,3>(qx_qy_qz_qs));
					
					a_to_world0 = a_to_world0 - simd_float::make4(1.0f, 0.0f, 0.0f, 0.0f);
					a_to_world1 = a_to_world1 - simd_float::make4(0.0f, 1.0f, 0.0f, 0.0f);
					a_to_world2 = a_to_world2 - simd_float::make4(0.0f, 0.0f, 1.0f, 0.0f);
					
					a_to_world0 = simd::bitwise_xor(a_to_world0, simd_float::make4(-0.0f, 0.0f, 0.0f, 0.0f));
					a_to_world1 = simd::bitwise_xor(a_to_world1, simd_float::make4(0.0f, -0.0f, 0.0f, 0.0f));
					a_to_world2 = simd::bitwise_xor(a_to_world2, simd_float::make4(0.0f, 0.0f, -0.0f, 0.0f));
				}
				
				// Add valid support points as contacts.
				simd4_float wn = a_face == 0 ? a_to_world0 : (a_face == 1 ? a_to_world1 : a_to_world2);
				
				if (b_offset_neg)
					wn = simd::bitwise_xor(wn, simd_float::make4(-0.0f));
				
				simd4_float a_position = simd_float::load4(transforms[a_index].position);
				
				uint16_t a_body = (uint16_t)transforms[a_index].body;
				uint16_t b_body = (uint16_t)transforms[b_index].body;
				
				a_index = transforms[a_index].body >> 16;
				b_index = transforms[b_index].body >> 16;
				
				unsigned tag_swap = 0;
				
				if (b_index > a_index) {
					unsigned tc = a_index;
					uint16_t tb = a_body;
					
					a_index = b_index;
					b_index = tc;
					
					a_body = b_body;
					b_body = tb;
					
					tag_swap = 16;
					
					wn = simd::bitwise_xor(wn, simd_float::make4(-0.0f));;
				}
				
				uint64_t high_tag = ((uint64_t)a_index << 32) | ((uint64_t)b_index << 48);
				
				while (mask) {
					unsigned index = first_set_bit(mask);
					mask &= mask-1;
					
					simd4_float wp = (a_to_world0 * simd_float::broadcast_load4(support_x + index) +
									  a_to_world1 * simd_float::broadcast_load4(support_y + index) +
									  a_to_world2 * simd_float::broadcast_load4(support_z + index) + a_position);
					
					float penetration = penetrations[index];
					
					simd_float::store4(contacts[count].position, wp);
					simd_float::store4(contacts[count].normal, wn);
					
					contacts[count].penetration = penetration;
					contacts[count].friction = 0.5f;
					bodies[count].a = a_body;
					bodies[count].b = b_body;
					tags[count] = (uint32_t)(support_tags[index] >> tag_swap) | (uint32_t)(support_tags[index] << tag_swap) | high_tag;
					
					++count;
				}
			}
			
			// Batch edge pairs.
			// Note: We need to output the edge pairs after handling the faces since we read from the pairs array during face processing.
			while (edge) {
				unsigned index = first_set_bit(edge);
				edge &= edge-1;
				
				unsigned pair = pairs[pair_offset + index];
				unsigned edge_a = a_edge_array[index];
				unsigned edge_b = b_edge_array[index];
				
				unsigned a = pair & 0xffff;
				unsigned b = pair >> 16;
				
				a = transforms[a].body >> 16;
				b = transforms[b].body >> 16;
				
				feature_penetrations[added] = penetration_array[index];
				features[added] = a > b ? edge_a | (edge_b << 16) : edge_b | (edge_a << 16);
				pairs[added] = a > b ? pair : (pair >> 16) | (pair << 16);
				
				++added;
			}
		}
		
		assert(!added || pairs[added-1]); // There should be no padding.
		
		pair_count = added;
	}
	
	// Do edge-edge tests.
	{
		pairs[pair_count+0] = 0; // Padding.
		pairs[pair_count+1] = 0;
		pairs[pair_count+2] = 0;
		
		features[pair_count+0] = 0;
		features[pair_count+1] = 0;
		features[pair_count+2] = 0;
		
		feature_penetrations[pair_count+0] = 0.0f;
		feature_penetrations[pair_count+1] = 0.0f;
		feature_penetrations[pair_count+2] = 0.0f;
		
		for (unsigned i = 0; i < pair_count; i += 4) {
			// Load pairs.
			unsigned pair0 = pairs[i + 0];
			unsigned pair1 = pairs[i + 1];
			unsigned pair2 = pairs[i + 2];
			unsigned pair3 = pairs[i + 3];
			
			unsigned a0_index = pair0 & 0xffff;
			unsigned b0_index = pair0 >> 16;
			
			unsigned a1_index = pair1 & 0xffff;
			unsigned b1_index = pair1 >> 16;
			
			unsigned a2_index = pair2 & 0xffff;
			unsigned b2_index = pair2 >> 16;
			
			unsigned a3_index = pair3 & 0xffff;
			unsigned b3_index = pair3 >> 16;
			
			// Load rotations.
			simd4_float a_rotation_x = simd_float::load4(transforms[a0_index].rotation);
			simd4_float a_rotation_y = simd_float::load4(transforms[a1_index].rotation);
			simd4_float a_rotation_z = simd_float::load4(transforms[a2_index].rotation);
			simd4_float a_rotation_s = simd_float::load4(transforms[a3_index].rotation);
			
			simd4_float b_rotation_x = simd_float::load4(transforms[b0_index].rotation);
			simd4_float b_rotation_y = simd_float::load4(transforms[b1_index].rotation);
			simd4_float b_rotation_z = simd_float::load4(transforms[b2_index].rotation);
			simd4_float b_rotation_s = simd_float::load4(transforms[b3_index].rotation);
			
			simd128::transpose32(a_rotation_x, a_rotation_y, a_rotation_z, a_rotation_s);
			simd128::transpose32(b_rotation_x, b_rotation_y, b_rotation_z, b_rotation_s);
			
			// Compute rotation matrices.
			simd4_float a_basis_xx, a_basis_xy, a_basis_xz;
			simd4_float a_basis_yx, a_basis_yy, a_basis_yz;
			simd4_float a_basis_zx, a_basis_zy, a_basis_zz;
			{
				simd4_float kx = a_rotation_x + a_rotation_x;
				simd4_float ky = a_rotation_y + a_rotation_y;
				simd4_float kz = a_rotation_z + a_rotation_z;
				
				simd4_float xx = kx*a_rotation_x;
				simd4_float yy = ky*a_rotation_y;
				simd4_float zz = kz*a_rotation_z;
				simd4_float xy = kx*a_rotation_y;
				simd4_float xz = kx*a_rotation_z;
				simd4_float yz = ky*a_rotation_z;
				simd4_float sx = kx*a_rotation_s;
				simd4_float sy = ky*a_rotation_s;
				simd4_float sz = kz*a_rotation_s;
				
				a_basis_xx = simd_float::make4(1.0f) - yy - zz;
				a_basis_xy = xy + sz;
				a_basis_xz = xz - sy;
				
				a_basis_yx = xy - sz;
				a_basis_yy = simd_float::make4(1.0f) - xx - zz;
				a_basis_yz = yz + sx;
				
				a_basis_zx = xz + sy;
				a_basis_zy = yz - sx;
				a_basis_zz = simd_float::make4(1.0f) - xx - yy;
			}
			
			simd4_float b_basis_xx, b_basis_xy, b_basis_xz;
			simd4_float b_basis_yx, b_basis_yy, b_basis_yz;
			simd4_float b_basis_zx, b_basis_zy, b_basis_zz;
			{
				simd4_float kx = b_rotation_x + b_rotation_x;
				simd4_float ky = b_rotation_y + b_rotation_y;
				simd4_float kz = b_rotation_z + b_rotation_z;
				
				simd4_float xx = kx*b_rotation_x;
				simd4_float yy = ky*b_rotation_y;
				simd4_float zz = kz*b_rotation_z;
				simd4_float xy = kx*b_rotation_y;
				simd4_float xz = kx*b_rotation_z;
				simd4_float yz = ky*b_rotation_z;
				simd4_float sx = kx*b_rotation_s;
				simd4_float sy = ky*b_rotation_s;
				simd4_float sz = kz*b_rotation_s;
				
				b_basis_xx = simd_float::make4(1.0f) - yy - zz;
				b_basis_xy = xy + sz;
				b_basis_xz = xz - sy;
				
				b_basis_yx = xy - sz;
				b_basis_yy = simd_float::make4(1.0f) - xx - zz;
				b_basis_yz = yz + sx;
				
				b_basis_zx = xz + sy;
				b_basis_zy = yz - sx;
				b_basis_zz = simd_float::make4(1.0f) - xx - yy;
			}
			
			// Load edges.
			simd4_int32 edge = simd_int32::load4((const int32_t*)(features + i));
			
			// Select edge directions.
#ifdef NUDGE_NATIVE_BLENDV32
			simd4_int32 a_select_y = simd_int32::shift_left<32-1>(edge); // Shifts the relevant bit to the top.
			simd4_int32 a_select_z = simd_int32::shift_left<32-2>(edge);
			
			simd4_int32 b_select_y = simd_int32::shift_left<16-1>(edge);
			simd4_int32 b_select_z = simd_int32::shift_left<16-2>(edge);
			
			simd4_float u_x = simd::blendv32(a_basis_xx, a_basis_yx, simd_int32::asfloat(a_select_y));
			simd4_float u_y = simd::blendv32(a_basis_xy, a_basis_yy, simd_int32::asfloat(a_select_y));
			simd4_float u_z = simd::blendv32(a_basis_xz, a_basis_yz, simd_int32::asfloat(a_select_y));
			
			simd4_float v_x = simd::blendv32(b_basis_xx, b_basis_yx, simd_int32::asfloat(b_select_y));
			simd4_float v_y = simd::blendv32(b_basis_xy, b_basis_yy, simd_int32::asfloat(b_select_y));
			simd4_float v_z = simd::blendv32(b_basis_xz, b_basis_yz, simd_int32::asfloat(b_select_y));
			
			u_x = simd::blendv32(u_x, a_basis_zx, simd_int32::asfloat(a_select_z));
			u_y = simd::blendv32(u_y, a_basis_zy, simd_int32::asfloat(a_select_z));
			u_z = simd::blendv32(u_z, a_basis_zz, simd_int32::asfloat(a_select_z));
			
			v_x = simd::blendv32(v_x, b_basis_zx, simd_int32::asfloat(b_select_z));
			v_y = simd::blendv32(v_y, b_basis_zy, simd_int32::asfloat(b_select_z));
			v_z = simd::blendv32(v_z, b_basis_zz, simd_int32::asfloat(b_select_z));
#else
			simd4_int32 a_edge = simd::bitwise_and(edge, simd_int32::make4(0xffff));
			simd4_int32 b_edge = simd_int32::shift_right<16>(edge);
			
			simd4_float a_select_x = simd_int32::asfloat(simd_int32::cmp_eq(a_edge, simd_int32::zero4()));
			simd4_float a_select_y = simd_int32::asfloat(simd_int32::cmp_eq(a_edge, simd_int32::make4(1)));
			simd4_float a_select_z = simd_int32::asfloat(simd_int32::cmp_eq(a_edge, simd_int32::make4(2)));
			
			simd4_float b_select_x = simd_int32::asfloat(simd_int32::cmp_eq(b_edge, simd_int32::zero4()));
			simd4_float b_select_y = simd_int32::asfloat(simd_int32::cmp_eq(b_edge, simd_int32::make4(1)));
			simd4_float b_select_z = simd_int32::asfloat(simd_int32::cmp_eq(b_edge, simd_int32::make4(2)));
			
			simd4_float u_x = simd::bitwise_and(a_basis_xx, a_select_x);
			simd4_float u_y = simd::bitwise_and(a_basis_xy, a_select_x);
			simd4_float u_z = simd::bitwise_and(a_basis_xz, a_select_x);
			
			simd4_float v_x = simd::bitwise_and(b_basis_xx, b_select_x);
			simd4_float v_y = simd::bitwise_and(b_basis_xy, b_select_x);
			simd4_float v_z = simd::bitwise_and(b_basis_xz, b_select_x);
			
			u_x = simd::bitwise_or(u_x, simd::bitwise_and(a_basis_yx, a_select_y));
			u_y = simd::bitwise_or(u_y, simd::bitwise_and(a_basis_yy, a_select_y));
			u_z = simd::bitwise_or(u_z, simd::bitwise_and(a_basis_yz, a_select_y));
			
			v_x = simd::bitwise_or(v_x, simd::bitwise_and(b_basis_yx, b_select_y));
			v_y = simd::bitwise_or(v_y, simd::bitwise_and(b_basis_yy, b_select_y));
			v_z = simd::bitwise_or(v_z, simd::bitwise_and(b_basis_yz, b_select_y));
			
			u_x = simd::bitwise_or(u_x, simd::bitwise_and(a_basis_zx, a_select_z));
			u_y = simd::bitwise_or(u_y, simd::bitwise_and(a_basis_zy, a_select_z));
			u_z = simd::bitwise_or(u_z, simd::bitwise_and(a_basis_zz, a_select_z));
			
			v_x = simd::bitwise_or(v_x, simd::bitwise_and(b_basis_zx, b_select_z));
			v_y = simd::bitwise_or(v_y, simd::bitwise_and(b_basis_zy, b_select_z));
			v_z = simd::bitwise_or(v_z, simd::bitwise_and(b_basis_zz, b_select_z));
#endif
			
			// Compute axis.
			simd4_float n_x, n_y, n_z;
			simd_soa::cross(u_x, u_y, u_z, v_x, v_y, v_z, n_x, n_y, n_z);
			
			// Load positions.
			simd4_float a_position_x = simd_float::load4(transforms[a0_index].position);
			simd4_float a_position_y = simd_float::load4(transforms[a1_index].position);
			simd4_float a_position_z = simd_float::load4(transforms[a2_index].position);
			simd4_float a_position_w = simd_float::load4(transforms[a3_index].position);
			
			simd4_float b_position_x = simd_float::load4(transforms[b0_index].position);
			simd4_float b_position_y = simd_float::load4(transforms[b1_index].position);
			simd4_float b_position_z = simd_float::load4(transforms[b2_index].position);
			simd4_float b_position_w = simd_float::load4(transforms[b3_index].position);
			
			simd128::transpose32(a_position_x, a_position_y, a_position_z, a_position_w);
			simd128::transpose32(b_position_x, b_position_y, b_position_z, b_position_w);
			
			// Compute relative position.
			simd4_float delta_x = b_position_x - a_position_x;
			simd4_float delta_y = b_position_y - a_position_y;
			simd4_float delta_z = b_position_z - a_position_z;
			
			// Flip normal?
			simd4_float sign_mask = simd_float::make4(-0.0f);
			simd4_float flip_sign = simd::bitwise_and(n_x*delta_x + n_y*delta_y + n_z*delta_z, sign_mask);
			
			n_x = simd::bitwise_xor(n_x, flip_sign);
			n_y = simd::bitwise_xor(n_y, flip_sign);
			n_z = simd::bitwise_xor(n_z, flip_sign);
			
			// Load sizes.
			simd4_float a_size_x = simd_float::load4(colliders[a0_index].size);
			simd4_float a_size_y = simd_float::load4(colliders[a1_index].size);
			simd4_float a_size_z = simd_float::load4(colliders[a2_index].size);
			simd4_float a_size_w = simd_float::load4(colliders[a3_index].size);
			
			simd4_float b_size_x = simd_float::load4(colliders[b0_index].size);
			simd4_float b_size_y = simd_float::load4(colliders[b1_index].size);
			simd4_float b_size_z = simd_float::load4(colliders[b2_index].size);
			simd4_float b_size_w = simd_float::load4(colliders[b3_index].size);
			
			simd128::transpose32(a_size_x, a_size_y, a_size_z, a_size_w);
			simd128::transpose32(b_size_x, b_size_y, b_size_z, b_size_w);
			
			// Compute direction to the edge.
			simd4_float a_sign_x = a_basis_xx*n_x + a_basis_xy*n_y + a_basis_xz*n_z;
			simd4_float a_sign_y = a_basis_yx*n_x + a_basis_yy*n_y + a_basis_yz*n_z;
			simd4_float a_sign_z = a_basis_zx*n_x + a_basis_zy*n_y + a_basis_zz*n_z;
			
			simd4_float b_sign_x = b_basis_xx*n_x + b_basis_xy*n_y + b_basis_xz*n_z;
			simd4_float b_sign_y = b_basis_yx*n_x + b_basis_yy*n_y + b_basis_yz*n_z;
			simd4_float b_sign_z = b_basis_zx*n_x + b_basis_zy*n_y + b_basis_zz*n_z;
			
			a_sign_x = simd::bitwise_and(a_sign_x, sign_mask);
			a_sign_y = simd::bitwise_and(a_sign_y, sign_mask);
			a_sign_z = simd::bitwise_and(a_sign_z, sign_mask);
			
			b_sign_x = simd::bitwise_and(b_sign_x, sign_mask);
			b_sign_y = simd::bitwise_and(b_sign_y, sign_mask);
			b_sign_z = simd::bitwise_and(b_sign_z, sign_mask);
			
			simd4_int32 edge_x = simd::bitwise_or(simd_int32::shift_right<31-0>(simd_float::asint(a_sign_x)), simd_int32::shift_right<31-16>(simd_float::asint(simd::bitwise_xor(b_sign_x, simd_float::make4(-0.0f)))));
			simd4_int32 edge_y = simd::bitwise_or(simd_int32::shift_right<31-1>(simd_float::asint(a_sign_y)), simd_int32::shift_right<31-17>(simd_float::asint(simd::bitwise_xor(b_sign_y, simd_float::make4(-0.0f)))));
			simd4_int32 edge_z = simd::bitwise_or(simd_int32::shift_right<31-2>(simd_float::asint(a_sign_z)), simd_int32::shift_right<31-18>(simd_float::asint(simd::bitwise_xor(b_sign_z, simd_float::make4(-0.0f)))));
			simd4_int32 edge_w = _mm_add_epi16(_mm_add_epi16(edge, _mm_set1_epi16(1)), _mm_srli_epi16(edge, 1)); // Calculates 1 << edge (valid for 0-2).
			
			simd4_int32 edge_xy = simd::bitwise_or(edge_x, edge_y);
			simd4_int32 edge_zw = simd::bitwise_or(edge_z, edge_w);
			
			simd4_int32 tag_hi = simd::bitwise_or(edge_xy, edge_zw);
			simd4_int32 tag_lo = simd::bitwise_notand(edge_w, tag_hi);
			tag_hi = simd_int32::shift_left<8>(tag_hi);
			
			simd4_int32 tag = simd::bitwise_or(tag_lo, tag_hi);
			
			a_size_x = simd::bitwise_xor(a_size_x, a_sign_x);
			a_size_y = simd::bitwise_xor(a_size_y, a_sign_y);
			a_size_z = simd::bitwise_xor(a_size_z, a_sign_z);
			
			b_size_x = simd::bitwise_xor(b_size_x, b_sign_x);
			b_size_y = simd::bitwise_xor(b_size_y, b_sign_y);
			b_size_z = simd::bitwise_xor(b_size_z, b_sign_z);
			
			a_basis_xx *= a_size_x;
			a_basis_xy *= a_size_x;
			a_basis_xz *= a_size_x;
			
			a_basis_yx *= a_size_y;
			a_basis_yy *= a_size_y;
			a_basis_yz *= a_size_y;
			
			a_basis_zx *= a_size_z;
			a_basis_zy *= a_size_z;
			a_basis_zz *= a_size_z;
			
			b_basis_xx *= b_size_x;
			b_basis_xy *= b_size_x;
			b_basis_xz *= b_size_x;
			
			b_basis_yx *= b_size_y;
			b_basis_yy *= b_size_y;
			b_basis_yz *= b_size_y;
			
			b_basis_zx *= b_size_z;
			b_basis_zy *= b_size_z;
			b_basis_zz *= b_size_z;
			
			simd4_float ca_x = a_basis_xx + a_basis_yx + a_basis_zx + a_position_x;
			simd4_float ca_y = a_basis_xy + a_basis_yy + a_basis_zy + a_position_y;
			simd4_float ca_z = a_basis_xz + a_basis_yz + a_basis_zz + a_position_z;
			
			simd4_float cb_x = b_basis_xx + b_basis_yx + b_basis_zx - b_position_x; // Note that cb really is negated to save some operations.
			simd4_float cb_y = b_basis_xy + b_basis_yy + b_basis_zy - b_position_y;
			simd4_float cb_z = b_basis_xz + b_basis_yz + b_basis_zz - b_position_z;
			
			// Calculate closest point between the two lines.
			simd4_float o_x = ca_x + cb_x;
			simd4_float o_y = ca_y + cb_y;
			simd4_float o_z = ca_z + cb_z;
			
			simd4_float ia = u_x*u_x + u_y*u_y + u_z*u_z;
			simd4_float ib = u_x*v_x + u_y*v_y + u_z*v_z;
			simd4_float ic = v_x*v_x + v_y*v_y + v_z*v_z;
			simd4_float id = o_x*u_x + o_y*u_y + o_z*u_z;
			simd4_float ie = o_x*v_x + o_y*v_y + o_z*v_z;
			
			simd4_float half = simd_float::make4(0.5f);
			simd4_float ir = half / (ia*ic - ib*ib);
			
			simd4_float sa = (ib*ie - ic*id) * ir;
			simd4_float sb = (ia*ie - ib*id) * ir;
			
			simd4_float p_x = (ca_x - cb_x)*half + u_x*sa + v_x*sb;
			simd4_float p_y = (ca_y - cb_y)*half + u_y*sa + v_y*sb;
			simd4_float p_z = (ca_z - cb_z)*half + u_z*sa + v_z*sb;
			
			simd_soa::normalize(n_x, n_y, n_z);
			
			simd4_float p_w = simd_float::load4(feature_penetrations + i);
			simd4_float n_w = simd_float::make4(0.5f);
			
			simd128::transpose32(p_x, p_y, p_z, p_w);
			simd128::transpose32(n_x, n_y, n_z, n_w);
			
			simd_float::store4(contacts[count + 0].position, p_x);
			simd_float::store4(contacts[count + 0].normal, n_x);
			simd_float::store4(contacts[count + 1].position, p_y);
			simd_float::store4(contacts[count + 1].normal, n_y);
			simd_float::store4(contacts[count + 2].position, p_z);
			simd_float::store4(contacts[count + 2].normal, n_z);
			simd_float::store4(contacts[count + 3].position, p_w);
			simd_float::store4(contacts[count + 3].normal, n_w);
			
			simd4_float body_pair = simd::bitwise_or(simd::bitwise_and(a_position_w, simd_int32::asfloat(simd_int32::make4(0xffff))), simd_int32::asfloat(simd_int32::shift_left<16>(simd_float::asint(b_position_w))));
			simd_float::storeu4((float*)(bodies + count), body_pair);
			
			simd4_int32 pair = simd_float::asint(simd::bitwise_or(simd::bitwise_and(b_position_w, simd_int32::asfloat(simd_int32::make4(0xffff0000))), simd_int32::asfloat(simd_int32::shift_right<16>(simd_float::asint(a_position_w)))));
			
			simd_int32::storeu4((int32_t*)tags + count*2 + 0, simd128::unpacklo32(tag, pair));
			simd_int32::storeu4((int32_t*)tags + count*2 + 4, simd128::unpackhi32(tag, pair));
			
			count += 4;
		}
		
		// Get rid of padding.
		while (count && bodies[count-1].a == bodies[count-1].b)
			--count;
	}
	
	return count;
}

static inline unsigned sphere_sphere_collide(SphereCollider a, SphereCollider b, Transform a_transform, Transform b_transform, Contact* contacts, BodyPair* bodies) {
	float r = a.radius + b.radius;
	
	float3 dp = make_float3(b_transform.position) - make_float3(a_transform.position);
	float l2 = length2(dp);
	
	if (l2 > r*r)
		return 0;
	
	float3 n;
	float l = sqrtf(l2);
	
	if (l2 > 1e-4f)
		n = dp * (1.0f / l);
	else
		n = make_float3(1.0f, 0.0f, 0.0f);
	
	float3 p = make_float3(a_transform.position) + n * (l - b.radius);
	
	contacts[0].position[0] = p.x;
	contacts[0].position[1] = p.y;
	contacts[0].position[2] = p.z;
	contacts[0].penetration = r - l;
	contacts[0].normal[0] = n.x;
	contacts[0].normal[1] = n.y;
	contacts[0].normal[2] = n.z;
	contacts[0].friction = 0.5f;
	
	bodies[0].a = (uint16_t)a_transform.body;
	bodies[0].b = (uint16_t)b_transform.body;
	
	return 1;
}

static inline unsigned box_sphere_collide(BoxCollider a, SphereCollider b, Transform a_transform, Transform b_transform, Contact* contacts, BodyPair* bodies) {
	Rotation a_to_world = make_rotation(a_transform.rotation);
	Rotation world_to_a = inverse(a_to_world);
	float3 offset_b = world_to_a * (make_float3(b_transform.position) - make_float3(a_transform.position));
	
	float dx = fabsf(offset_b.x);
	float dy = fabsf(offset_b.y);
	float dz = fabsf(offset_b.z);
	
	float w = a.size[0] + b.radius;
	float h = a.size[1] + b.radius;
	float d = a.size[2] + b.radius;
	
	if (dx >= w || dy >= h || dz >= d)
		return 0;
	
	float3 n;
	float penetration;
	
	float r = b.radius;
	
	unsigned outside_x = dx > a.size[0];
	unsigned outside_y = dy > a.size[1];
	unsigned outside_z = dz > a.size[2];
	
	if (outside_x + outside_y + outside_z >= 2) {
		float3 corner = {
			outside_x ? (offset_b.x > 0.0f ? a.size[0] : -a.size[0]) : offset_b.x,
			outside_y ? (offset_b.y > 0.0f ? a.size[1] : -a.size[1]) : offset_b.y,
			outside_z ? (offset_b.z > 0.0f ? a.size[2] : -a.size[2]) : offset_b.z,
		};
		
		float3 dp = offset_b - corner;
		float l2 = length2(dp);
		
		if (l2 > r*r)
			return 0;
		
		float l = sqrtf(l2);
		float m = 1.0f / l;
		
		n = dp * m;
		penetration = r - l;
	}
	else if (w - dx < h - dy && w - dx < d - dz) {
		n.x = offset_b.x > 0.0f ? 1.0f : -1.0f;
		n.y = 0.0f;
		n.z = 0.0f;
		penetration = w - dx;
	}
	else if (h - dy < d - dz) {
		n.x = 0.0f;
		n.y = offset_b.y > 0.0f ? 1.0f : -1.0f;
		n.z = 0.0f;
		penetration = h - dy;
	}
	else {
		n.x = 0.0f;
		n.y = 0.0f;
		n.z = offset_b.z > 0.0f ? 1.0f : -1.0f;
		penetration = d - dz;
	}
	
	float3 p = offset_b - n*r;
	
	p = a_to_world * p + make_float3(a_transform.position);
	n = a_to_world * n;
	
	contacts[0].position[0] = p.x;
	contacts[0].position[1] = p.y;
	contacts[0].position[2] = p.z;
	contacts[0].penetration = penetration;
	contacts[0].normal[0] = n.x;
	contacts[0].normal[1] = n.y;
	contacts[0].normal[2] = n.z;
	contacts[0].friction = 0.5f;
	
	bodies[0].a = (uint16_t)a_transform.body;
	bodies[0].b = (uint16_t)b_transform.body;
	
	return 1;
}

template<unsigned offset>
static inline void dilate_3(simdv_int32 x, simdv_int32& lo32, simdv_int32& hi32) {
	simdv_int32 mask0 = simd_int32::makev(0xff);
	simdv_int32 mask1 = simd_int32::makev(0x0f00f00f);
	simdv_int32 mask2 = simd_int32::makev(0xc30c30c3);
	simdv_int32 mask3 = simd_int32::makev(0x49249249);
	
	simdv_int32 lo24 = x;
	simdv_int32 hi24 = simd_int32::shift_right<8>(x);
	lo24 = simd::bitwise_and(lo24, mask0);
	hi24 = simd::bitwise_and(hi24, mask0);
	
	lo24 = simd::bitwise_or(lo24, simd_int32::shift_left<8>(lo24));
	hi24 = simd::bitwise_or(hi24, simd_int32::shift_left<8>(hi24));
	lo24 = simd::bitwise_and(lo24, mask1);
	hi24 = simd::bitwise_and(hi24, mask1);
	
	lo24 = simd::bitwise_or(lo24, simd_int32::shift_left<4>(lo24));
	hi24 = simd::bitwise_or(hi24, simd_int32::shift_left<4>(hi24));
	lo24 = simd::bitwise_and(lo24, mask2);
	hi24 = simd::bitwise_and(hi24, mask2);
	
	lo24 = simd::bitwise_or(lo24, simd_int32::shift_left<2>(lo24));
	hi24 = simd::bitwise_or(hi24, simd_int32::shift_left<2>(hi24));
	lo24 = simd::bitwise_and(lo24, mask3);
	hi24 = simd::bitwise_and(hi24, mask3);
	
	lo32 = simd::bitwise_or(simd_int32::shift_left<offset>(lo24), simd_int32::shift_left<24+offset>(hi24));
	hi32 = simd_int32::shift_right<8-offset>(hi24);
}

static inline void morton(simdv_int32 x, simdv_int32 y, simdv_int32 z, simdv_int32& lo32, simdv_int32& hi32) {
	simdv_int32 lx, hx, ly, hy, lz, hz;
	dilate_3<2>(x, lx, hx);
	dilate_3<1>(y, ly, hy);
	dilate_3<0>(z, lz, hz);
	
	lo32 = simd::bitwise_or(simd::bitwise_or(lx, ly), lz);
	hi32 = simd::bitwise_or(simd::bitwise_or(hx, hy), hz);
}

static inline void radix_sort_uint64_low48(uint64_t* data, unsigned count, Arena temporary) {
	uint64_t* temp = allocate_array<uint64_t>(&temporary, count, 16);
	
	unsigned buckets0[257] = {};
	unsigned buckets1[257] = {};
	unsigned buckets2[257] = {};
	unsigned buckets3[257] = {};
	unsigned buckets4[257] = {};
	unsigned buckets5[257] = {};
	
	unsigned* histogram0 = buckets0+1;
	unsigned* histogram1 = buckets1+1;
	unsigned* histogram2 = buckets2+1;
	unsigned* histogram3 = buckets3+1;
	unsigned* histogram4 = buckets4+1;
	unsigned* histogram5 = buckets5+1;
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = data[i];
		
		++histogram0[(d >> (0 << 3)) & 0xff];
		++histogram1[(d >> (1 << 3)) & 0xff];
		++histogram2[(d >> (2 << 3)) & 0xff];
		++histogram3[(d >> (3 << 3)) & 0xff];
		++histogram4[(d >> (4 << 3)) & 0xff];
		++histogram5[(d >> (5 << 3)) & 0xff];
	}
	
	for (unsigned i = 1; i < 256; ++i) {
		buckets0[i] += buckets0[i-1];
		buckets1[i] += buckets1[i-1];
		buckets2[i] += buckets2[i-1];
		buckets3[i] += buckets3[i-1];
		buckets4[i] += buckets4[i-1];
		buckets5[i] += buckets5[i-1];
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = data[i];
		unsigned index = buckets0[(d >> (0 << 3)) & 0xff]++;
		temp[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = temp[i];
		unsigned index = buckets1[(d >> (1 << 3)) & 0xff]++;
		data[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = data[i];
		unsigned index = buckets2[(d >> (2 << 3)) & 0xff]++;
		temp[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = temp[i];
		unsigned index = buckets3[(d >> (3 << 3)) & 0xff]++;
		data[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = data[i];
		unsigned index = buckets4[(d >> (4 << 3)) & 0xff]++;
		temp[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t d = temp[i];
		unsigned index = buckets5[(d >> (5 << 3)) & 0xff]++;
		data[index] = d;
	}
}

static inline void radix_sort_uint32_x2(uint32_t* data, uint32_t* data2, unsigned count, Arena temporary) {
	uint32_t* temp = allocate_array<uint32_t>(&temporary, count, 16);
	uint32_t* temp2 = allocate_array<uint32_t>(&temporary, count, 16);
	
	unsigned buckets0[257] = {};
	unsigned buckets1[257] = {};
	unsigned buckets2[257] = {};
	unsigned buckets3[257] = {};
	
	unsigned* histogram0 = buckets0+1;
	unsigned* histogram1 = buckets1+1;
	unsigned* histogram2 = buckets2+1;
	unsigned* histogram3 = buckets3+1;
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = data[i];
		
		++histogram0[(d >> (0 << 3)) & 0xff];
		++histogram1[(d >> (1 << 3)) & 0xff];
		++histogram2[(d >> (2 << 3)) & 0xff];
		++histogram3[(d >> (3 << 3)) & 0xff];
	}
	
	for (unsigned i = 1; i < 256; ++i) {
		buckets0[i] += buckets0[i-1];
		buckets1[i] += buckets1[i-1];
		buckets2[i] += buckets2[i-1];
		buckets3[i] += buckets3[i-1];
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = data[i];
		uint32_t d2 = data2[i];
		unsigned index = buckets0[(d >> (0 << 3)) & 0xff]++;
		temp[index] = d;
		temp2[index] = d2;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = temp[i];
		uint32_t d2 = temp2[i];
		unsigned index = buckets1[(d >> (1 << 3)) & 0xff]++;
		data[index] = d;
		data2[index] = d2;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = data[i];
		uint32_t d2 = data2[i];
		unsigned index = buckets2[(d >> (2 << 3)) & 0xff]++;
		temp[index] = d;
		temp2[index] = d2;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = temp[i];
		uint32_t d2 = temp2[i];
		unsigned index = buckets3[(d >> (3 << 3)) & 0xff]++;
		data[index] = d;
		data2[index] = d2;
	}
}

static inline void radix_sort_uint32(uint32_t* data, unsigned count, Arena temporary) {
	uint32_t* temp = allocate_array<uint32_t>(&temporary, count, 16);
	
	unsigned buckets0[257] = {};
	unsigned buckets1[257] = {};
	unsigned buckets2[257] = {};
	unsigned buckets3[257] = {};
	
	unsigned* histogram0 = buckets0+1;
	unsigned* histogram1 = buckets1+1;
	unsigned* histogram2 = buckets2+1;
	unsigned* histogram3 = buckets3+1;
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = data[i];
		
		++histogram0[(d >> (0 << 3)) & 0xff];
		++histogram1[(d >> (1 << 3)) & 0xff];
		++histogram2[(d >> (2 << 3)) & 0xff];
		++histogram3[(d >> (3 << 3)) & 0xff];
	}
	
	for (unsigned i = 1; i < 256; ++i) {
		buckets0[i] += buckets0[i-1];
		buckets1[i] += buckets1[i-1];
		buckets2[i] += buckets2[i-1];
		buckets3[i] += buckets3[i-1];
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = data[i];
		unsigned index = buckets0[(d >> (0 << 3)) & 0xff]++;
		temp[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = temp[i];
		unsigned index = buckets1[(d >> (1 << 3)) & 0xff]++;
		data[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = data[i];
		unsigned index = buckets2[(d >> (2 << 3)) & 0xff]++;
		temp[index] = d;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t d = temp[i];
		unsigned index = buckets3[(d >> (3 << 3)) & 0xff]++;
		data[index] = d;
	}
}

template<unsigned data_stride, unsigned index_stride, class T>
NUDGE_FORCEINLINE static void load4(const float* data, const T* indices,
									simdv_float& d0, simdv_float& d1, simdv_float& d2, simdv_float& d3) {
	static const unsigned stride_in_floats = data_stride/sizeof(float);
	
#if NUDGE_SIMDV_WIDTH == 256
	unsigned i0 = indices[0*index_stride];
	unsigned i1 = indices[1*index_stride];
	unsigned i2 = indices[2*index_stride];
	unsigned i3 = indices[3*index_stride];
	
	simd4_float t0 = simd_float::load4(data + i0*stride_in_floats);
	simd4_float t1 = simd_float::load4(data + i1*stride_in_floats);
	simd4_float t2 = simd_float::load4(data + i2*stride_in_floats);
	simd4_float t3 = simd_float::load4(data + i3*stride_in_floats);
	
	unsigned i4 = indices[4*index_stride];
	unsigned i5 = indices[5*index_stride];
	unsigned i6 = indices[6*index_stride];
	unsigned i7 = indices[7*index_stride];
	
	simd4_float t4 = simd_float::load4(data + i4*stride_in_floats);
	simd4_float t5 = simd_float::load4(data + i5*stride_in_floats);
	simd4_float t6 = simd_float::load4(data + i6*stride_in_floats);
	simd4_float t7 = simd_float::load4(data + i7*stride_in_floats);
	
	d0 = simd::concat(t0, t4);
	d1 = simd::concat(t1, t5);
	d2 = simd::concat(t2, t6);
	d3 = simd::concat(t3, t7);
#else
	unsigned i0 = indices[0*index_stride];
	unsigned i1 = indices[1*index_stride];
	unsigned i2 = indices[2*index_stride];
	unsigned i3 = indices[3*index_stride];
	
	d0 = simd_float::load4(data + i0*stride_in_floats);
	d1 = simd_float::load4(data + i1*stride_in_floats);
	d2 = simd_float::load4(data + i2*stride_in_floats);
	d3 = simd_float::load4(data + i3*stride_in_floats);
#endif
	
	simd128::transpose32(d0, d1, d2, d3);
}

template<unsigned data_stride, unsigned index_stride, class T>
NUDGE_FORCEINLINE static void load8(const float* data, const T* indices,
									simdv_float& d0, simdv_float& d1, simdv_float& d2, simdv_float& d3,
									simdv_float& d4, simdv_float& d5, simdv_float& d6, simdv_float& d7) {
	static const unsigned stride_in_floats = data_stride/sizeof(float);
	
#if NUDGE_SIMDV_WIDTH == 256
	unsigned i0 = indices[0*index_stride];
	unsigned i1 = indices[1*index_stride];
	unsigned i2 = indices[2*index_stride];
	unsigned i3 = indices[3*index_stride];
	
	simdv_float t0 = simd_float::load8(data + i0*stride_in_floats);
	simdv_float t1 = simd_float::load8(data + i1*stride_in_floats);
	simdv_float t2 = simd_float::load8(data + i2*stride_in_floats);
	simdv_float t3 = simd_float::load8(data + i3*stride_in_floats);
	
	unsigned i4 = indices[4*index_stride];
	unsigned i5 = indices[5*index_stride];
	unsigned i6 = indices[6*index_stride];
	unsigned i7 = indices[7*index_stride];
	
	simdv_float t4 = simd_float::load8(data + i4*stride_in_floats);
	simdv_float t5 = simd_float::load8(data + i5*stride_in_floats);
	simdv_float t6 = simd_float::load8(data + i6*stride_in_floats);
	simdv_float t7 = simd_float::load8(data + i7*stride_in_floats);
	
	d0 = simd256::permute128<0,2>(t0, t4);
	d1 = simd256::permute128<0,2>(t1, t5);
	d2 = simd256::permute128<0,2>(t2, t6);
	d3 = simd256::permute128<0,2>(t3, t7);
	
	d4 = simd256::permute128<1,3>(t0, t4);
	d5 = simd256::permute128<1,3>(t1, t5);
	d6 = simd256::permute128<1,3>(t2, t6);
	d7 = simd256::permute128<1,3>(t3, t7);
#else
	unsigned i0 = indices[0*index_stride];
	unsigned i1 = indices[1*index_stride];
	unsigned i2 = indices[2*index_stride];
	unsigned i3 = indices[3*index_stride];
	
	d0 = simd_float::load4(data + i0*stride_in_floats);
	d1 = simd_float::load4(data + i1*stride_in_floats);
	d2 = simd_float::load4(data + i2*stride_in_floats);
	d3 = simd_float::load4(data + i3*stride_in_floats);
	
	d4 = simd_float::load4(data + i0*stride_in_floats + 4);
	d5 = simd_float::load4(data + i1*stride_in_floats + 4);
	d6 = simd_float::load4(data + i2*stride_in_floats + 4);
	d7 = simd_float::load4(data + i3*stride_in_floats + 4);
#endif
	
	simd128::transpose32(d0, d1, d2, d3);
	simd128::transpose32(d4, d5, d6, d7);
}

template<unsigned data_stride, unsigned index_stride, class T>
NUDGE_FORCEINLINE static void store8(float* data, const T* indices,
									 simdv_float d0, simdv_float d1, simdv_float d2, simdv_float d3,
									 simdv_float d4, simdv_float d5, simdv_float d6, simdv_float d7) {
	static const unsigned stride_in_floats = data_stride/sizeof(float);
	
#if NUDGE_SIMDV_WIDTH == 256
	simdv_float t0 = simd256::permute128<0,2>(d0, d4);
	simdv_float t1 = simd256::permute128<0,2>(d1, d5);
	simdv_float t2 = simd256::permute128<0,2>(d2, d6);
	simdv_float t3 = simd256::permute128<0,2>(d3, d7);
	
	simdv_float t4 = simd256::permute128<1,3>(d0, d4);
	simdv_float t5 = simd256::permute128<1,3>(d1, d5);
	simdv_float t6 = simd256::permute128<1,3>(d2, d6);
	simdv_float t7 = simd256::permute128<1,3>(d3, d7);
	
	simd128::transpose32(t0, t1, t2, t3);
	simd128::transpose32(t4, t5, t6, t7);
	
	unsigned i0 = indices[0*index_stride];
	unsigned i1 = indices[1*index_stride];
	unsigned i2 = indices[2*index_stride];
	unsigned i3 = indices[3*index_stride];
	
	simd_float::store8(data + i0*stride_in_floats, t0);
	simd_float::store8(data + i1*stride_in_floats, t1);
	simd_float::store8(data + i2*stride_in_floats, t2);
	simd_float::store8(data + i3*stride_in_floats, t3);
	
	unsigned i4 = indices[4*index_stride];
	unsigned i5 = indices[5*index_stride];
	unsigned i6 = indices[6*index_stride];
	unsigned i7 = indices[7*index_stride];
	
	simd_float::store8(data + i4*stride_in_floats, t4);
	simd_float::store8(data + i5*stride_in_floats, t5);
	simd_float::store8(data + i6*stride_in_floats, t6);
	simd_float::store8(data + i7*stride_in_floats, t7);
#else
	simd128::transpose32(d0, d1, d2, d3);
	simd128::transpose32(d4, d5, d6, d7);
	
	unsigned i0 = indices[0*index_stride];
	unsigned i1 = indices[1*index_stride];
	unsigned i2 = indices[2*index_stride];
	unsigned i3 = indices[3*index_stride];
	
	simd_float::store4(data + i0*stride_in_floats, d0);
	simd_float::store4(data + i1*stride_in_floats, d1);
	simd_float::store4(data + i2*stride_in_floats, d2);
	simd_float::store4(data + i3*stride_in_floats, d3);
	
	simd_float::store4(data + i0*stride_in_floats + 4, d4);
	simd_float::store4(data + i1*stride_in_floats + 4, d5);
	simd_float::store4(data + i2*stride_in_floats + 4, d6);
	simd_float::store4(data + i3*stride_in_floats + 4, d7);
#endif
}

void collide(ActiveBodies* active_bodies, ContactData* contacts, BodyData bodies, ColliderData colliders, BodyConnections body_connections, Arena temporary) {
	contacts->count = 0;
	contacts->sleeping_count = 0;
	active_bodies->count = 0;
	
	const Transform* body_transforms = bodies.transforms;
	
	unsigned count = colliders.spheres.count + colliders.boxes.count;
	unsigned aligned_count = (count + 7) & (~7);
	
	assert(count <= (1 << 13)); // Too many colliders. 2^13 is currently the maximum.
	
	AABB* aos_bounds = allocate_array<AABB>(&temporary, aligned_count, 32);
	
	unsigned box_bounds_offset = 0;
	unsigned sphere_bounds_offset = colliders.boxes.count;
	
	Transform* transforms = allocate_array<Transform>(&temporary, count, 32);
	uint16_t* collider_tags = allocate_array<uint16_t>(&temporary, count, 32);
	uint16_t* collider_bodies = allocate_array<uint16_t>(&temporary, count, 32);
	
	if (colliders.boxes.count) {
		for (unsigned i = 0; i < colliders.boxes.count; ++i) {
			Transform transform = colliders.boxes.transforms[i];
			transform = body_transforms[transform.body] * transform;
			transform.body |= (uint32_t)colliders.boxes.tags[i] << 16;
			
			float3x3 m = matrix(make_rotation(transform.rotation));
			
			m.c0 *= colliders.boxes.data[i].size[0];
			m.c1 *= colliders.boxes.data[i].size[1];
			m.c2 *= colliders.boxes.data[i].size[2];
			
			float3 size = {
				fabsf(m.c0.x) + fabsf(m.c1.x) + fabsf(m.c2.x),
				fabsf(m.c0.y) + fabsf(m.c1.y) + fabsf(m.c2.y),
				fabsf(m.c0.z) + fabsf(m.c1.z) + fabsf(m.c2.z),
			};
			
			float3 min = make_float3(transform.position) - size;
			float3 max = make_float3(transform.position) + size;
			
			AABB aabb = {
				min, 0.0f,
				max, 0.0f,
			};
			
			transforms[i + box_bounds_offset] = transform;
			aos_bounds[i + box_bounds_offset] = aabb;
			collider_tags[i + box_bounds_offset] = colliders.boxes.tags[i];
			collider_bodies[i + box_bounds_offset] = colliders.boxes.transforms[i].body;
		}
		
		colliders.boxes.transforms = transforms + box_bounds_offset;
	}
	
	if (colliders.spheres.count) {
		for (unsigned i = 0; i < colliders.spheres.count; ++i) {
			Transform transform = colliders.spheres.transforms[i];
			transform = body_transforms[transform.body] * transform;
			transform.body |= (uint32_t)colliders.spheres.tags[i] << 16;
			
			float radius = colliders.spheres.data[i].radius;
			
			float3 min = make_float3(transform.position) - make_float3(radius);
			float3 max = make_float3(transform.position) + make_float3(radius);
			
			AABB aabb = {
				min, 0.0f,
				max, 0.0f,
			};
			
			transforms[i + sphere_bounds_offset] = transform;
			aos_bounds[i + sphere_bounds_offset] = aabb;
			collider_tags[i + sphere_bounds_offset] = colliders.spheres.tags[i];
			collider_bodies[i + sphere_bounds_offset] = colliders.spheres.transforms[i].body;
		}
		
		colliders.spheres.transforms = transforms + sphere_bounds_offset;
	}
	
	for (unsigned i = count; i < aligned_count; ++i) {
		AABB zero = {};
		aos_bounds[i] = zero;
	}
	
	// Morton order using the min corner should improve coherence: After some point, all BBs' min points will be outside a BB's max.
	simd4_float scene_min128 = simd_float::load4(&aos_bounds[0].min.x);
	simd4_float scene_max128 = scene_min128;
	
	for (unsigned i = 1; i < count; ++i) {
		simd4_float p = simd_float::load4(&aos_bounds[i].min.x);
		scene_min128 = simd_float::min(scene_min128, p);
		scene_max128 = simd_float::max(scene_max128, p);
	}
	
	simd4_float scene_scale128 = simd_float::make4((1<<16)-1) * simd_float::recip(scene_max128 - scene_min128);
	
	scene_scale128 = simd_float::min(simd128::shuffle32<0,1,2,2>(scene_scale128), simd128::shuffle32<2,2,0,1>(scene_scale128));
	scene_scale128 = simd_float::min(scene_scale128, simd128::shuffle32<1,0,3,2>(scene_scale128));
	scene_min128 = scene_min128 * scene_scale128;
	
#ifdef DEBUG
	if (simd_float::extract_first_float(scene_scale128) < 2.0f)
		printf("Warning: World bounds are very large, which may decrease performance. Perhaps there's a body in free fall?\n");
#endif
	
#if NUDGE_SIMDV_WIDTH == 256
	simdv_float scene_min = simd256::broadcast(scene_min128);
	simdv_float scene_scale = simd256::broadcast(scene_scale128);
	simdv_int32 index = simd_int32::make8(0 << 16, 1 << 16, 2 << 16, 3 << 16, 4 << 16, 5 << 16, 6 << 16, 7 << 16);
#else
	simdv_float scene_min = scene_min128;
	simdv_float scene_scale = scene_scale128;
	simdv_int32 index = simd_int32::make4(0 << 16, 1 << 16, 2 << 16, 3 << 16);
#endif
	
	simdv_float scene_min_x = simd128::shuffle32<0,0,0,0>(scene_min);
	simdv_float scene_min_y = simd128::shuffle32<1,1,1,1>(scene_min);
	simdv_float scene_min_z = simd128::shuffle32<2,2,2,2>(scene_min);
	
	uint64_t* morton_codes = allocate_array<uint64_t>(&temporary, aligned_count, 32);
	
	for (unsigned i = 0; i < count; i += simdv_width32) {
#if NUDGE_SIMDV_WIDTH == 256
		simd4_float pos_xl = simd_float::load4(&aos_bounds[i+0].min.x);
		simd4_float pos_yl = simd_float::load4(&aos_bounds[i+1].min.x);
		simd4_float pos_zl = simd_float::load4(&aos_bounds[i+2].min.x);
		simd4_float pos_wl = simd_float::load4(&aos_bounds[i+3].min.x);
		
		simdv_float pos_x = simd::concat(pos_xl, simd_float::load4(&aos_bounds[i+4].min.x));
		simdv_float pos_y = simd::concat(pos_yl, simd_float::load4(&aos_bounds[i+5].min.x));
		simdv_float pos_z = simd::concat(pos_zl, simd_float::load4(&aos_bounds[i+6].min.x));
		simdv_float pos_w = simd::concat(pos_wl, simd_float::load4(&aos_bounds[i+7].min.x));
#else
		simd4_float pos_x = simd_float::load4(&aos_bounds[i+0].min.x);
		simd4_float pos_y = simd_float::load4(&aos_bounds[i+1].min.x);
		simd4_float pos_z = simd_float::load4(&aos_bounds[i+2].min.x);
		simd4_float pos_w = simd_float::load4(&aos_bounds[i+3].min.x);
#endif
		
		simd128::transpose32(pos_x, pos_y, pos_z, pos_w);
		
		pos_x = simd_float::msub(pos_x, scene_scale, scene_min_x);
		pos_y = simd_float::msub(pos_y, scene_scale, scene_min_y);
		pos_z = simd_float::msub(pos_z, scene_scale, scene_min_z);
		
		simdv_int32 lm, hm;
		morton(simd_float::toint(pos_x), simd_float::toint(pos_y), simd_float::toint(pos_z), lm, hm);
		hm = simd::bitwise_or(hm, index);
		
		simdv_int32 mi0 = simd128::unpacklo32(lm, hm);
		simdv_int32 mi1 = simd128::unpackhi32(lm, hm);
		
#if NUDGE_SIMDV_WIDTH == 256
		simd_int32::store8((int32_t*)(morton_codes + i) + 0, simd256::permute128<0,2>(mi0, mi1));
		simd_int32::store8((int32_t*)(morton_codes + i) + 8, simd256::permute128<1,3>(mi0, mi1));
#else
		simd_int32::store4((int32_t*)(morton_codes + i) + 0, mi0);
		simd_int32::store4((int32_t*)(morton_codes + i) + 4, mi1);
#endif
		
		index = simd_int32::add(index, simd_int32::makev(simdv_width32 << 16));
	}
	
	radix_sort_uint64_low48(morton_codes, count, temporary);
	uint16_t* sorted_indices = allocate_array<uint16_t>(&temporary, aligned_count, 32);
	
	for (unsigned i = 0; i < count; ++i)
		sorted_indices[i] = (uint16_t)(morton_codes[i] >> 48);
	
	for (unsigned i = count; i < aligned_count; ++i)
		sorted_indices[i] = 0;
	
	unsigned bounds_count = aligned_count >> simdv_width32_log2;
	AABBV* bounds = allocate_array<AABBV>(&temporary, bounds_count, 32);
	
	for (unsigned i = 0; i < count; i += simdv_width32) {
		simdv_float min_x, min_y, min_z, min_w;
		simdv_float max_x, max_y, max_z, max_w;
		load8<sizeof(aos_bounds[0]), 1>(&aos_bounds[0].min.x, sorted_indices + i,
										min_x, min_y, min_z, min_w,
										max_x, max_y, max_z, max_w);
		
		simd_float::storev(bounds[i >> simdv_width32_log2].min_x, min_x);
		simd_float::storev(bounds[i >> simdv_width32_log2].max_x, max_x);
		simd_float::storev(bounds[i >> simdv_width32_log2].min_y, min_y);
		simd_float::storev(bounds[i >> simdv_width32_log2].max_y, max_y);
		simd_float::storev(bounds[i >> simdv_width32_log2].min_z, min_z);
		simd_float::storev(bounds[i >> simdv_width32_log2].max_z, max_z);
	}
	
	for (unsigned i = count; i < aligned_count; ++i) {
		unsigned bounds_group = i >> simdv_width32_log2;
		unsigned bounds_lane = i & (simdv_width32-1);
		
		bounds[bounds_group].min_x[bounds_lane] = NAN;
		bounds[bounds_group].max_x[bounds_lane] = NAN;
		bounds[bounds_group].min_y[bounds_lane] = NAN;
		bounds[bounds_group].max_y[bounds_lane] = NAN;
		bounds[bounds_group].min_z[bounds_lane] = NAN;
		bounds[bounds_group].max_z[bounds_lane] = NAN;
	}
	
	// Pack each set of 8 consecutive AABBs into coarse AABBs.
	unsigned coarse_count = aligned_count >> 3;
	unsigned aligned_coarse_count = (coarse_count + (simdv_width32-1)) & (~(simdv_width32-1));
	
	unsigned coarse_bounds_count = aligned_coarse_count >> simdv_width32_log2;
	AABBV* coarse_bounds = allocate_array<AABBV>(&temporary, coarse_bounds_count, 32);
	
	for (unsigned i = 0; i < coarse_count; ++i) {
		unsigned start = i << (3 - simdv_width32_log2);
		
		simd4_float coarse_min_x = simd_float::load4(bounds[start].min_x);
		simd4_float coarse_max_x = simd_float::load4(bounds[start].max_x);
		simd4_float coarse_min_y = simd_float::load4(bounds[start].min_y);
		simd4_float coarse_max_y = simd_float::load4(bounds[start].max_y);
		simd4_float coarse_min_z = simd_float::load4(bounds[start].min_z);
		simd4_float coarse_max_z = simd_float::load4(bounds[start].max_z);
		
		// Note that the first operand is returned on NaN. The last padded bounds are NaN, so the earlier bounds should be in the first operand.
#if NUDGE_SIMDV_WIDTH == 256
		coarse_min_x = simd_float::min(coarse_min_x, simd_float::load4(bounds[start].min_x + 4));
		coarse_max_x = simd_float::max(coarse_max_x, simd_float::load4(bounds[start].max_x + 4));
		coarse_min_y = simd_float::min(coarse_min_y, simd_float::load4(bounds[start].min_y + 4));
		coarse_max_y = simd_float::max(coarse_max_y, simd_float::load4(bounds[start].max_y + 4));
		coarse_min_z = simd_float::min(coarse_min_z, simd_float::load4(bounds[start].min_z + 4));
		coarse_max_z = simd_float::max(coarse_max_z, simd_float::load4(bounds[start].max_z + 4));
#else
		coarse_min_x = simd_float::min(coarse_min_x, simd_float::load4(bounds[start+1].min_x));
		coarse_max_x = simd_float::max(coarse_max_x, simd_float::load4(bounds[start+1].max_x));
		coarse_min_y = simd_float::min(coarse_min_y, simd_float::load4(bounds[start+1].min_y));
		coarse_max_y = simd_float::max(coarse_max_y, simd_float::load4(bounds[start+1].max_y));
		coarse_min_z = simd_float::min(coarse_min_z, simd_float::load4(bounds[start+1].min_z));
		coarse_max_z = simd_float::max(coarse_max_z, simd_float::load4(bounds[start+1].max_z));
#endif
		
		coarse_min_x = simd_float::min(coarse_min_x, simd128::shuffle32<2,3,0,1>(coarse_min_x));
		coarse_max_x = simd_float::max(coarse_max_x, simd128::shuffle32<2,3,0,1>(coarse_max_x));
		coarse_min_y = simd_float::min(coarse_min_y, simd128::shuffle32<2,3,0,1>(coarse_min_y));
		coarse_max_y = simd_float::max(coarse_max_y, simd128::shuffle32<2,3,0,1>(coarse_max_y));
		coarse_min_z = simd_float::min(coarse_min_z, simd128::shuffle32<2,3,0,1>(coarse_min_z));
		coarse_max_z = simd_float::max(coarse_max_z, simd128::shuffle32<2,3,0,1>(coarse_max_z));
		
		coarse_min_x = simd_float::min(coarse_min_x, simd128::shuffle32<1,0,3,2>(coarse_min_x));
		coarse_max_x = simd_float::max(coarse_max_x, simd128::shuffle32<1,0,3,2>(coarse_max_x));
		coarse_min_y = simd_float::min(coarse_min_y, simd128::shuffle32<1,0,3,2>(coarse_min_y));
		coarse_max_y = simd_float::max(coarse_max_y, simd128::shuffle32<1,0,3,2>(coarse_max_y));
		coarse_min_z = simd_float::min(coarse_min_z, simd128::shuffle32<1,0,3,2>(coarse_min_z));
		coarse_max_z = simd_float::max(coarse_max_z, simd128::shuffle32<1,0,3,2>(coarse_max_z));
		
		unsigned bounds_group = i >> simdv_width32_log2;
		unsigned bounds_lane = i & (simdv_width32-1);
		
		coarse_bounds[bounds_group].min_x[bounds_lane] = simd_float::extract_first_float(coarse_min_x);
		coarse_bounds[bounds_group].max_x[bounds_lane] = simd_float::extract_first_float(coarse_max_x);
		coarse_bounds[bounds_group].min_y[bounds_lane] = simd_float::extract_first_float(coarse_min_y);
		coarse_bounds[bounds_group].max_y[bounds_lane] = simd_float::extract_first_float(coarse_max_y);
		coarse_bounds[bounds_group].min_z[bounds_lane] = simd_float::extract_first_float(coarse_min_z);
		coarse_bounds[bounds_group].max_z[bounds_lane] = simd_float::extract_first_float(coarse_max_z);
	}
	
	for (unsigned i = coarse_count; i < aligned_coarse_count; ++i) {
		unsigned bounds_group = i >> simdv_width32_log2;
		unsigned bounds_lane = i & (simdv_width32-1);
		
		coarse_bounds[bounds_group].min_x[bounds_lane] = NAN;
		coarse_bounds[bounds_group].max_x[bounds_lane] = NAN;
		coarse_bounds[bounds_group].min_y[bounds_lane] = NAN;
		coarse_bounds[bounds_group].max_y[bounds_lane] = NAN;
		coarse_bounds[bounds_group].min_z[bounds_lane] = NAN;
		coarse_bounds[bounds_group].max_z[bounds_lane] = NAN;
	}
	
	// Test all coarse groups against each other and generate pairs with potential overlap.
	uint32_t* coarse_groups = reserve_array<uint32_t>(&temporary, coarse_count*coarse_count, 32);
	unsigned coarse_group_count = 0;
	
	for (unsigned i = 0; i < coarse_count; ++i) {
		unsigned bounds_group = i >> simdv_width32_log2;
		unsigned bounds_lane = i & (simdv_width32-1);
		
		simdv_float min_a_x = simd_float::broadcast_loadv(coarse_bounds[bounds_group].min_x + bounds_lane);
		simdv_float max_a_x = simd_float::broadcast_loadv(coarse_bounds[bounds_group].max_x + bounds_lane);
		simdv_float min_a_y = simd_float::broadcast_loadv(coarse_bounds[bounds_group].min_y + bounds_lane);
		simdv_float max_a_y = simd_float::broadcast_loadv(coarse_bounds[bounds_group].max_y + bounds_lane);
		simdv_float min_a_z = simd_float::broadcast_loadv(coarse_bounds[bounds_group].min_z + bounds_lane);
		simdv_float max_a_z = simd_float::broadcast_loadv(coarse_bounds[bounds_group].max_z + bounds_lane);
		
		unsigned first = coarse_group_count;
		
		// Maximum number of colliders is 2^13, i.e., 13 bit indices.
		// i needs 10 bits.
		// j needs 7 or 8 bits.
		// mask needs 4 or 8 bits.
		unsigned ij_bits = (bounds_group << 8) | (i << 16);
		
		for (unsigned j = bounds_group; j < coarse_bounds_count; ++j) {
			simdv_float min_b_x = simd_float::loadv(coarse_bounds[j].min_x);
			simdv_float max_b_x = simd_float::loadv(coarse_bounds[j].max_x);
			simdv_float min_b_y = simd_float::loadv(coarse_bounds[j].min_y);
			simdv_float max_b_y = simd_float::loadv(coarse_bounds[j].max_y);
			simdv_float min_b_z = simd_float::loadv(coarse_bounds[j].min_z);
			simdv_float max_b_z = simd_float::loadv(coarse_bounds[j].max_z);
			
			simdv_float inside_x = simd::bitwise_and(simd_float::cmp_gt(max_b_x, min_a_x), simd_float::cmp_gt(max_a_x, min_b_x));
			simdv_float inside_y = simd::bitwise_and(simd_float::cmp_gt(max_b_y, min_a_y), simd_float::cmp_gt(max_a_y, min_b_y));
			simdv_float inside_z = simd::bitwise_and(simd_float::cmp_gt(max_b_z, min_a_z), simd_float::cmp_gt(max_a_z, min_b_z));
			
			unsigned mask = simd::signmask32(simd::bitwise_and(simd::bitwise_and(inside_x, inside_y), inside_z));
			
			coarse_groups[coarse_group_count] = mask | ij_bits;
			coarse_group_count += mask != 0;
			
			ij_bits += 1 << 8;
		}
		
		// Mask out collisions already handled.
		coarse_groups[first] &= ~((1 << bounds_lane) - 1);
	}
	
	commit_array<uint32_t>(&temporary, coarse_group_count);
	
	uint32_t* coarse_pairs = reserve_array<uint32_t>(&temporary, coarse_group_count*simdv_width32, 32);
	unsigned coarse_pair_count = 0;
	
	for (unsigned i = 0; i < coarse_group_count; ++i) {
		unsigned group = coarse_groups[i];
		unsigned mask = group & 0xff;
		
		unsigned batch = (group & 0xff00) >> (8 - simdv_width32_log2);
		unsigned other = group & 0xffff0000;
		
		while (mask) {
			unsigned index = first_set_bit(mask);
			mask &= mask-1;
			
			coarse_pairs[coarse_pair_count++] = other | (batch + index);
		}
	}
	
	commit_array<uint32_t>(&temporary, coarse_pair_count);
	
	// Test AABBs within the coarse pairs.
	uint32_t* groups = reserve_array<uint32_t>(&temporary, coarse_pair_count*16, 32);
	unsigned group_count = 0;
	
#if NUDGE_SIMDV_WIDTH == 256
	for (unsigned n = 0; n < coarse_pair_count; ++n) {
		unsigned pair = coarse_pairs[n];
		
		unsigned a = pair >> 16;
		unsigned b = pair & 0xffff;
		
		unsigned lane_count = 8;
		
		if (a == b)
			--lane_count;
		
		if (lane_count + (a << 3) > count)
			lane_count = count - (a << 3);
		
		// Maximum number of colliders is 2^13, i.e., 13 bit indices.
		// i needs 13 bits.
		// j needs 10 or 11 bits.
		// mask needs 4 or 8 bits.
		unsigned ij_bits = (b << 8) | (a << 22);
		
		unsigned lower_lane_mask = a == b ? 0xfe00 : 0xffff;
		
		simdv_float min_b_x = simd_float::loadv(bounds[b].min_x);
		simdv_float max_b_x = simd_float::loadv(bounds[b].max_x);
		simdv_float min_b_y = simd_float::loadv(bounds[b].min_y);
		simdv_float max_b_y = simd_float::loadv(bounds[b].max_y);
		simdv_float min_b_z = simd_float::loadv(bounds[b].min_z);
		simdv_float max_b_z = simd_float::loadv(bounds[b].max_z);
		
		for (unsigned i = 0; i < lane_count; ++i, ij_bits += (1 << 19)) {
			simdv_float min_a_x = simd_float::broadcast_loadv(bounds[a].min_x + i);
			simdv_float max_a_x = simd_float::broadcast_loadv(bounds[a].max_x + i);
			simdv_float min_a_y = simd_float::broadcast_loadv(bounds[a].min_y + i);
			simdv_float max_a_y = simd_float::broadcast_loadv(bounds[a].max_y + i);
			simdv_float min_a_z = simd_float::broadcast_loadv(bounds[a].min_z + i);
			simdv_float max_a_z = simd_float::broadcast_loadv(bounds[a].max_z + i);
			
			simdv_float inside_x = simd::bitwise_and(simd_float::cmp_gt(max_b_x, min_a_x), simd_float::cmp_gt(max_a_x, min_b_x));
			simdv_float inside_y = simd::bitwise_and(simd_float::cmp_gt(max_b_y, min_a_y), simd_float::cmp_gt(max_a_y, min_b_y));
			simdv_float inside_z = simd::bitwise_and(simd_float::cmp_gt(max_b_z, min_a_z), simd_float::cmp_gt(max_a_z, min_b_z));
			
			unsigned mask = simd::signmask32(simd::bitwise_and(simd::bitwise_and(inside_x, inside_y), inside_z));
			
			// Mask out collisions already handled.
			mask &= lower_lane_mask >> 8;
			lower_lane_mask <<= 1;
			
			groups[group_count] = mask | ij_bits;
			group_count += mask != 0;
		}
	}
#else
	// TODO: This version is currently much worse than the 256-bit version. We should fix it.
	for (unsigned n = 0; n < coarse_pair_count; ++n) {
		unsigned pair = coarse_pairs[n];
		
		unsigned a = pair >> 16;
		unsigned b = pair & 0xffff;
		
		unsigned a_start = a << 3;
		unsigned a_end = a_start + (1 << 3);
		
		if (a_end > count)
			a_end = count;
		
		unsigned b_start = b << (3 - simdv_width32_log2);
		unsigned b_end = b_start + (1 << (3 - simdv_width32_log2));
		
		if (b_end > bounds_count)
			b_end = bounds_count;
		
		for (unsigned i = a_start; i < a_end; ++i) {
			unsigned bounds_group = i >> simdv_width32_log2;
			unsigned bounds_lane = i & (simdv_width32-1);
			
			simdv_float min_a_x = simd_float::broadcast_loadv(bounds[bounds_group].min_x + bounds_lane);
			simdv_float max_a_x = simd_float::broadcast_loadv(bounds[bounds_group].max_x + bounds_lane);
			simdv_float min_a_y = simd_float::broadcast_loadv(bounds[bounds_group].min_y + bounds_lane);
			simdv_float max_a_y = simd_float::broadcast_loadv(bounds[bounds_group].max_y + bounds_lane);
			simdv_float min_a_z = simd_float::broadcast_loadv(bounds[bounds_group].min_z + bounds_lane);
			simdv_float max_a_z = simd_float::broadcast_loadv(bounds[bounds_group].max_z + bounds_lane);
			
			unsigned first = group_count;
			
			unsigned start = (i+1) >> simdv_width32_log2;
			
			if (start < b_start)
				start = b_start;
			
			// Maximum number of colliders is 2^13, i.e., 13 bit indices.
			// i needs 13 bits.
			// j needs 10 or 11 bits.
			// mask needs 4 or 8 bits.
			unsigned ij_bits = (start << 8) | (i << 19);
			
			for (unsigned j = start; j < b_end; ++j) {
				simdv_float min_b_x = simd_float::loadv(bounds[j].min_x);
				simdv_float max_b_x = simd_float::loadv(bounds[j].max_x);
				simdv_float min_b_y = simd_float::loadv(bounds[j].min_y);
				simdv_float max_b_y = simd_float::loadv(bounds[j].max_y);
				simdv_float min_b_z = simd_float::loadv(bounds[j].min_z);
				simdv_float max_b_z = simd_float::loadv(bounds[j].max_z);
				
				simdv_float inside_x = simd::bitwise_and(simd_float::cmp_gt(max_b_x, min_a_x), simd_float::cmp_gt(max_a_x, min_b_x));
				simdv_float inside_y = simd::bitwise_and(simd_float::cmp_gt(max_b_y, min_a_y), simd_float::cmp_gt(max_a_y, min_b_y));
				simdv_float inside_z = simd::bitwise_and(simd_float::cmp_gt(max_b_z, min_a_z), simd_float::cmp_gt(max_a_z, min_b_z));
				
				unsigned mask = simd::signmask32(simd::bitwise_and(simd::bitwise_and(inside_x, inside_y), inside_z));
				
				groups[group_count] = mask | ij_bits;
				group_count += mask != 0;
				
				ij_bits += 1 << 8;
			}
			
			// Mask out collisions already handled.
			if (first < group_count && (groups[first] & 0x7ff00) == (bounds_group << 8))
				groups[first] &= ~((2 << bounds_lane) - 1);
		}
	}
#endif
	
	commit_array<uint32_t>(&temporary, group_count);
	
	uint32_t* pairs = reserve_array<uint32_t>(&temporary, group_count*simdv_width32, 32);
	unsigned pair_count = 0;
	
	for (unsigned i = 0; i < group_count; ++i) {
		unsigned group = groups[i];
		unsigned mask = group & 0xff;
		
		unsigned batch = (group & 0x7ff00) >> (8 - simdv_width32_log2);
		unsigned base = ((uint32_t)(group >> 19) << 16) | batch;
		
		while (mask) {
			unsigned index = first_set_bit(mask);
			mask &= mask-1;
			
			pairs[pair_count++] = base + index;
		}
	}
	
	commit_array<uint32_t>(&temporary, pair_count);
	
	for (unsigned i = 0; i < pair_count; ++i) {
		unsigned pair = pairs[i];
		pairs[i] = sorted_indices[pair & 0xffff] | ((uint32_t)sorted_indices[pair >> 16] << 16);
	}
	
	radix_sort_uint32(pairs, pair_count, temporary);
	
	// Discard islands of inactive objects at a coarse level, before detailed collisions.
	{
		NUDGE_ARENA_SCOPE(temporary);
		
		// Find connected sets.
		uint16_t* heights = allocate_array<uint16_t>(&temporary, bodies.count, 16);
		uint16_t* parents = allocate_array<uint16_t>(&temporary, bodies.count, 16);
		
		memset(heights, 0, sizeof(heights[0])*bodies.count);
		memset(parents, 0xff, sizeof(parents[0])*bodies.count);
		
		for (unsigned i = 0; i < body_connections.count; ++i) {
			BodyPair pair = body_connections.data[i];
			
			unsigned a = pair.a;
			unsigned b = pair.b;
			
			// Body 0 is the static world and is ignored.
			if (!a || !b)
				continue;
			
			// Determine the root of a and b.
			unsigned a_root = a;
			unsigned a_parent = parents[a];
			
			for (unsigned parent = a_parent; parent != 0xffff; parent = parents[a_root])
				a_root = parent;
			
			unsigned b_root = b;
			unsigned b_parent = parents[b];
			
			for (unsigned parent = b_parent; parent != 0xffff; parent = parents[b_root])
				b_root = parent;
			
			if (a_root == b_root)
				continue;
			
			// Put a and b under the same root.
			unsigned a_height = heights[a_root];
			unsigned b_height = heights[b_root];
			
			unsigned root;
			
			if (a_height < b_height) {
				parents[a_root] = b_root;
				root = b_root;
			}
			else {
				parents[b_root] = a_root;
				root = a_root;
			}
			
			if (a_height == b_height) // Height of subtree increased.
				heights[a_root] = a_height+1;
			
			// Propagate the root to make subsequent iterations faster.
			if (a_root != a) {
				while (a_parent != a_root) {
					unsigned next = parents[a_parent];
					parents[a] = root;
					
					a = a_parent;
					a_parent = next;
				}
			}
			
			if (b_root != b) {
				while (b_parent != b_root) {
					unsigned next = parents[b_parent];
					parents[b] = root;
					
					b = b_parent;
					b_parent = next;
				}
			}
		}
		
		for (unsigned i = 0; i < pair_count; ++i) {
			unsigned pair = pairs[i];
			
			unsigned a = collider_bodies[pair & 0xffff];
			unsigned b = collider_bodies[pair >> 16];
			
			// Body 0 is the static world and is ignored.
			if (!a || !b)
				continue;
			
			// Determine the root of a and b.
			unsigned a_root = a;
			unsigned a_parent = parents[a];
			
			for (unsigned parent = a_parent; parent != 0xffff; parent = parents[a_root])
				a_root = parent;
			
			unsigned b_root = b;
			unsigned b_parent = parents[b];
			
			for (unsigned parent = b_parent; parent != 0xffff; parent = parents[b_root])
				b_root = parent;
			
			if (a_root == b_root)
				continue;
			
			// Put a and b under the same root.
			unsigned a_height = heights[a_root];
			unsigned b_height = heights[b_root];
			
			unsigned root;
			
			if (a_height < b_height) {
				parents[a_root] = b_root;
				root = b_root;
			}
			else {
				parents[b_root] = a_root;
				root = a_root;
			}
			
			if (a_height == b_height) // Height of subtree increased.
				heights[a_root] = a_height+1;
			
			// Propagate the root to make subsequent iterations faster.
			if (a_root != a) {
				while (a_parent != a_root) {
					unsigned next = parents[a_parent];
					parents[a] = root;
					
					a = a_parent;
					a_parent = next;
				}
			}
			
			if (b_root != b) {
				while (b_parent != b_root) {
					unsigned next = parents[b_parent];
					parents[b] = root;
					
					b = b_parent;
					b_parent = next;
				}
			}
		}
		
		// Identify a numbered set for each body.
		unsigned set_count = 0;
		uint16_t* sets = heights;
		memset(sets, 0xff, sizeof(sets[0])*bodies.count);
		
		for (unsigned i = 1; i < bodies.count; ++i) {
			unsigned root = parents[i];
			
			for (unsigned parent = root; parent != 0xffff; parent = parents[root])
				root = parent;
			
			if (root == 0xffff)
				root = i;
			
			if (sets[root] == 0xffff)
				sets[root] = set_count++;
			
			sets[i] = sets[root];
		}
		
		sets[0] = 0;
		
		// Determine active sets.
		uint8_t* active = allocate_array<uint8_t>(&temporary, set_count, 16);
		memset(active, 0, sizeof(active[0])*set_count);
		
		for (unsigned i = 1; i < bodies.count; ++i) {
			if (bodies.idle_counters[i] != 0xff)
				active[sets[i]] = 1;
		}
		
		// Remove inactive pairs.
		unsigned removed = 0;
		
		for (unsigned i = 0; i < pair_count; ++i) {
			unsigned pair = pairs[i];
			
			unsigned a = collider_bodies[pair & 0xffff];
			unsigned b = collider_bodies[pair >> 16];
			
			if (a == b) {
				++removed;
				continue;
			}
			
			unsigned set = sets[a] | sets[b];
			
			if (active[set]) {
				pairs[i-removed] = pair;
			}
			else {
				unsigned a = collider_tags[pair & 0xffff];
				unsigned b = collider_tags[pair >> 16];
				
				contacts->sleeping_pairs[contacts->sleeping_count++] = a > b ? a | (b << 16): b | (a << 16);
				++removed;
			}
		}
		
		pair_count -= removed;
	}
	
	uint32_t bucket_sizes[4] = {};
	
	for (unsigned i = 0; i < pair_count; ++i) {
		unsigned pair = pairs[i];
		
		unsigned a = pair & 0xffff;
		unsigned b = pair >> 16;
		
		a = a >= colliders.boxes.count ? 1 : 0;
		b = b >= colliders.boxes.count ? 2 : 0;
		
		unsigned ab = a | b;
		
		++bucket_sizes[ab];
	}
	
	uint32_t bucket_offsets[4] = {
		0,
		((bucket_sizes[0] + 7) & ~3),
		((bucket_sizes[0] + 7) & ~3) + bucket_sizes[1],
		((bucket_sizes[0] + 7) & ~3) + bucket_sizes[1] + bucket_sizes[2],
	};
	
	uint32_t written_per_bucket[4] = { bucket_offsets[0], bucket_offsets[1], bucket_offsets[2], bucket_offsets[3] };
	
	uint32_t* partitioned_pairs = allocate_array<uint32_t>(&temporary, pair_count + 7, 16); // Padding is required.
	
	for (unsigned i = 0; i < pair_count; ++i) {
		unsigned pair = pairs[i];
		
		unsigned a = pair & 0xffff;
		unsigned b = pair >> 16;
		
		a = a >= colliders.boxes.count ? 1 : 0;
		b = b >= colliders.boxes.count ? 2 : 0;
		
		unsigned ab = a | b;
		
		partitioned_pairs[written_per_bucket[ab]++] = pair;
	}
	
	for (unsigned i = 0; i < bucket_sizes[2]; ++i) {
		unsigned index = bucket_offsets[2] + i;
		unsigned pair = partitioned_pairs[index];
		
		partitioned_pairs[index] = (pair >> 16) | (pair << 16);
	}
	
	contacts->count += box_box_collide(partitioned_pairs, bucket_sizes[0], colliders.boxes.data, colliders.boxes.transforms, contacts->data + contacts->count, contacts->bodies + contacts->count, contacts->tags + contacts->count, temporary);
	
	// TODO: SIMD-optimize this loop.
	for (unsigned i = 0; i < bucket_sizes[1] + bucket_sizes[2]; ++i) {
		unsigned pair = partitioned_pairs[bucket_offsets[1] + i];
		
		unsigned a = pair >> 16;
		unsigned b = pair & 0xffff;
		
		b -= colliders.boxes.count;
		
		BoxCollider box = colliders.boxes.data[a];
		SphereCollider sphere = colliders.spheres.data[b];
		
		contacts->tags[contacts->count] = (uint64_t)((colliders.boxes.transforms[a].body >> 16) | (colliders.spheres.transforms[b].body & 0xffff0000)) << 32;
		contacts->count += box_sphere_collide(box, sphere, colliders.boxes.transforms[a], colliders.spheres.transforms[b], contacts->data + contacts->count, contacts->bodies + contacts->count);
	}
	
	// TODO: SIMD-optimize this loop.
	for (unsigned i = 0; i < bucket_sizes[3]; ++i) {
		unsigned pair = partitioned_pairs[bucket_offsets[3] + i];
		
		unsigned a = pair >> 16;
		unsigned b = pair & 0xffff;
		
		a -= colliders.boxes.count;
		b -= colliders.boxes.count;
		
		SphereCollider sphere_a = colliders.spheres.data[a];
		SphereCollider sphere_b = colliders.spheres.data[b];
		
		contacts->tags[contacts->count] = (uint64_t)((colliders.spheres.transforms[a].body >> 16) | (colliders.spheres.transforms[b].body & 0xffff0000)) << 32;
		contacts->count += sphere_sphere_collide(sphere_a, sphere_b, colliders.spheres.transforms[a], colliders.spheres.transforms[b], contacts->data + contacts->count, contacts->bodies + contacts->count);
	}
	
	// Discard islands of inactive objects at a fine level.
	{
		NUDGE_ARENA_SCOPE(temporary);
		
		// Find connected sets.
		uint16_t* heights = allocate_array<uint16_t>(&temporary, bodies.count, 16);
		uint16_t* parents = allocate_array<uint16_t>(&temporary, bodies.count, 16);
		
		memset(heights, 0, sizeof(heights[0])*bodies.count);
		memset(parents, 0xff, sizeof(parents[0])*bodies.count);
		
		for (unsigned i = 0; i < body_connections.count; ++i) {
			BodyPair pair = body_connections.data[i];
			
			unsigned a = pair.a;
			unsigned b = pair.b;
			
			// Body 0 is the static world and is ignored.
			if (!a || !b)
				continue;
			
			// Determine the root of a and b.
			unsigned a_root = a;
			unsigned a_parent = parents[a];
			
			for (unsigned parent = a_parent; parent != 0xffff; parent = parents[a_root])
				a_root = parent;
			
			unsigned b_root = b;
			unsigned b_parent = parents[b];
			
			for (unsigned parent = b_parent; parent != 0xffff; parent = parents[b_root])
				b_root = parent;
			
			if (a_root == b_root)
				continue;
			
			// Put a and b under the same root.
			unsigned a_height = heights[a_root];
			unsigned b_height = heights[b_root];
			
			unsigned root;
			
			if (a_height < b_height) {
				parents[a_root] = b_root;
				root = b_root;
			}
			else {
				parents[b_root] = a_root;
				root = a_root;
			}
			
			if (a_height == b_height) // Height of subtree increased.
				heights[a_root] = a_height+1;
			
			// Propagate the root to make subsequent iterations faster.
			if (a_root != a) {
				while (a_parent != a_root) {
					unsigned next = parents[a_parent];
					parents[a] = root;
					
					a = a_parent;
					a_parent = next;
				}
			}
			
			if (b_root != b) {
				while (b_parent != b_root) {
					unsigned next = parents[b_parent];
					parents[b] = root;
					
					b = b_parent;
					b_parent = next;
				}
			}
		}
		
		for (unsigned i = 0; i < contacts->count; ) {
			unsigned a = contacts->bodies[i].a;
			unsigned b = contacts->bodies[i].b;
			
			do {
				++i;
			}
			while (i < contacts->count && contacts->bodies[i].a == a && contacts->bodies[i].b == b);
			
			// Body 0 is the static world and is ignored.
			if (!a || !b)
				continue;
			
			// Determine the root of a and b.
			unsigned a_root = a;
			unsigned a_parent = parents[a];
			
			for (unsigned parent = a_parent; parent != 0xffff; parent = parents[a_root])
				a_root = parent;
			
			unsigned b_root = b;
			unsigned b_parent = parents[b];
			
			for (unsigned parent = b_parent; parent != 0xffff; parent = parents[b_root])
				b_root = parent;
			
			if (a_root == b_root)
				continue;
			
			// Put a and b under the same root.
			unsigned a_height = heights[a_root];
			unsigned b_height = heights[b_root];
			
			unsigned root;
			
			if (a_height < b_height) {
				parents[a_root] = b_root;
				root = b_root;
			}
			else {
				parents[b_root] = a_root;
				root = a_root;
			}
			
			if (a_height == b_height) // Height of subtree increased.
				heights[a_root] = a_height+1;
			
			// Propagate the root to make subsequent iterations faster.
			if (a_root != a) {
				while (a_parent != a_root) {
					unsigned next = parents[a_parent];
					parents[a] = root;
					
					a = a_parent;
					a_parent = next;
				}
			}
			
			if (b_root != b) {
				while (b_parent != b_root) {
					unsigned next = parents[b_parent];
					parents[b] = root;
					
					b = b_parent;
					b_parent = next;
				}
			}
		}
		
		// Identify a numbered set for each body.
		unsigned set_count = 0;
		uint16_t* sets = heights;
		memset(sets, 0xff, sizeof(sets[0])*bodies.count);
		
		for (unsigned i = 1; i < bodies.count; ++i) {
			unsigned root = parents[i];
			
			for (unsigned parent = root; parent != 0xffff; parent = parents[root])
				root = parent;
			
			if (root == 0xffff)
				root = i;
			
			if (sets[root] == 0xffff)
				sets[root] = set_count++;
			
			sets[i] = sets[root];
		}
		
		sets[0] = 0;
		
		// Determine active sets.
		uint8_t* active = allocate_array<uint8_t>(&temporary, set_count, 16);
		memset(active, 0, sizeof(active[0])*set_count);
		
		for (unsigned i = 1; i < bodies.count; ++i) {
			if (bodies.idle_counters[i] != 0xff)
				active[sets[i]] = 1;
		}
		
		// Determine active bodies.
		for (unsigned i = 1; i < bodies.count; ++i) {
			unsigned set = sets[i];
			
			if (active[set])
				active_bodies->indices[active_bodies->count++] = i;
		}
		
		// Remove inactive contacts.
		unsigned removed = 0;
		
		for (unsigned i = 0; i < contacts->count; ) {
			unsigned a = contacts->bodies[i].a;
			unsigned b = contacts->bodies[i].b;
			unsigned tag = contacts->tags[i] >> 32;
			
			unsigned span = 0;
			
			do {
				++span;
			}
			while (i+span < contacts->count && (contacts->tags[i+span] >> 32) == tag);
			
			unsigned set = sets[a] | sets[b];
			
			if (active[set]) {
				for (unsigned j = 0; j < span; ++j) {
					contacts->tags[i+j-removed] = contacts->tags[i+j];
					contacts->data[i+j-removed] = contacts->data[i+j];
					contacts->bodies[i+j-removed] = contacts->bodies[i+j];
				}
			}
			else {
				contacts->sleeping_pairs[contacts->sleeping_count++] = tag;
				removed += span;
			}
			
			i += span;
		}
		
		contacts->count -= removed;
	}
	
	radix_sort_uint32(contacts->sleeping_pairs, contacts->sleeping_count, temporary);
}

struct ContactImpulseData {
	uint32_t* sorted_contacts;
	
	CachedContactImpulse* culled_data;
	uint64_t* culled_tags;
	unsigned culled_count;
	
	CachedContactImpulse* data;
};

ContactImpulseData* read_cached_impulses(ContactCache contact_cache, ContactData contacts, Arena* memory) {
	ContactImpulseData* data = allocate_struct<ContactImpulseData>(memory, 64);
	
	// Sort contacts based on tag so that they can be quickly matched against the contact cache.
	uint32_t* sorted_contacts = allocate_array<uint32_t>(memory, contacts.count, 16);
	data->sorted_contacts = sorted_contacts;
	{
		Arena temporary = *memory;
		uint32_t* contact_keys = allocate_array<uint32_t>(&temporary, contacts.count, 16);
		
		for (unsigned i = 0; i < contacts.count; ++i) {
			sorted_contacts[i] = i;
			contact_keys[i] = (uint32_t)contacts.tags[i];
		}
		
		radix_sort_uint32_x2(contact_keys, sorted_contacts, contacts.count, temporary);
		
		for (unsigned i = 0; i < contacts.count; ++i) {
			unsigned index = sorted_contacts[i];
			contact_keys[i] = (uint32_t)(contacts.tags[index] >> 32);
		}
		
		radix_sort_uint32_x2(contact_keys, sorted_contacts, contacts.count, temporary);
	}
	
	// Gather warm start impulses and store away culled impulses for sleeping pairs.
	CachedContactImpulse* culled_data = allocate_array<CachedContactImpulse>(memory, contact_cache.count, 16);
	uint64_t* culled_tags = allocate_array<uint64_t>(memory, contact_cache.count, 16);
	unsigned culled_count = 0;
	
	CachedContactImpulse* contact_impulses = allocate_array<CachedContactImpulse>(memory, contacts.count, 32);
	data->data = contact_impulses;
	
	unsigned cached_contact_offset = 0;
	unsigned sleeping_pair_offset = 0;
	
	for (unsigned i = 0; i < contacts.count; ++i) {
		unsigned index = sorted_contacts[i];
		uint64_t tag = contacts.tags[index];
		
		CachedContactImpulse cached_impulse = {};
		
		uint64_t cached_tag;
		while (cached_contact_offset < contact_cache.count && (cached_tag = contact_cache.tags[cached_contact_offset]) < tag) {
			unsigned cached_pair = cached_tag >> 32;
			
			while (sleeping_pair_offset < contacts.sleeping_count && contacts.sleeping_pairs[sleeping_pair_offset] < cached_pair)
				++sleeping_pair_offset;
			
			if (sleeping_pair_offset < contacts.sleeping_count && contacts.sleeping_pairs[sleeping_pair_offset] == cached_pair) {
				culled_data[culled_count] = contact_cache.data[cached_contact_offset];
				culled_tags[culled_count] = contact_cache.tags[cached_contact_offset];
				++culled_count;
			}
			
			++cached_contact_offset;
		}
		
		if (cached_contact_offset < contact_cache.count && contact_cache.tags[cached_contact_offset] == tag)
			cached_impulse = contact_cache.data[cached_contact_offset];
		
		contact_impulses[index] = cached_impulse;
	}
	
	for (; cached_contact_offset < contact_cache.count && sleeping_pair_offset < contacts.sleeping_count; ) {
		unsigned a = contact_cache.tags[cached_contact_offset] >> 32;
		unsigned b = contacts.sleeping_pairs[sleeping_pair_offset];
		
		if (a < b) {
			++cached_contact_offset;
		}
		else if (a == b) {
			culled_data[culled_count] = contact_cache.data[cached_contact_offset];
			culled_tags[culled_count] = contact_cache.tags[cached_contact_offset];
			++culled_count;
			++cached_contact_offset;
		}
		else {
			++sleeping_pair_offset;
		}
	}
	
	data->culled_data = culled_data;
	data->culled_tags = culled_tags;
	data->culled_count = culled_count;
	
	return data;
}

void write_cached_impulses(ContactCache* contact_cache, ContactData contacts, ContactImpulseData* contact_impulses) {
	uint32_t* sorted_contacts = contact_impulses->sorted_contacts;
	
	CachedContactImpulse* culled_data = contact_impulses->culled_data;
	uint64_t* culled_tags = contact_impulses->culled_tags;
	unsigned culled_count = contact_impulses->culled_count;
	
	// Cache impulses.
	assert(contact_cache->capacity >= contacts.count + culled_count); // Out of space in contact cache.
	contact_cache->count = contacts.count + culled_count;
	{
		// Pick sort from contacts and culled impulses.
		unsigned i = 0, j = 0, k = 0;
		
		while (i < contacts.count && j < culled_count) {
			unsigned index = sorted_contacts[i];
			
			uint64_t a = contacts.tags[index];
			uint64_t b = culled_tags[j];
			
			if (a < b) {
				contact_cache->tags[k] = contacts.tags[index];
				contact_cache->data[k] = contact_impulses->data[index];
				++i;
			}
			else {
				contact_cache->tags[k] = culled_tags[j];
				contact_cache->data[k] = culled_data[j];
				++j;
			}
			
			++k;
		}
		
		for (; i < contacts.count; ++i) {
			unsigned index = sorted_contacts[i];
			
			contact_cache->tags[k] = contacts.tags[index];
			contact_cache->data[k] = contact_impulses->data[index];
			++k;
		}
		
		for (; j < culled_count; ++j) {
			contact_cache->tags[k] = culled_tags[j];
			contact_cache->data[k] = culled_data[j];
			++k;
		}
	}
}

struct ContactConstraintData {
	unsigned contact_count;
	InertiaTransform* momentum_to_velocity;
	uint32_t* constraint_to_contact;
	
	ContactConstraintV* constraints;
	ContactConstraintStateV* constraint_states;
	unsigned constraint_batches;
};

ContactConstraintData* setup_contact_constraints(ActiveBodies active_bodies, ContactData contacts, BodyData bodies, ContactImpulseData* contact_impulses, Arena* memory) {
	// TODO: We should investigate better evaluation order for contacts.
	uint32_t* contact_order = contact_impulses->sorted_contacts;
	
	ContactConstraintData* data = allocate_struct<ContactConstraintData>(memory, 64);
	data->contact_count = contacts.count;
	
	InertiaTransform* momentum_to_velocity = allocate_array<InertiaTransform>(memory, bodies.count, 32);
	data->momentum_to_velocity = momentum_to_velocity;
	
	// TODO: Consider SIMD-optimizing this loop.
	// TODO: Don't compute anything for inactive bodies.
	for (unsigned i = 0; i < bodies.count; ++i) {
		Rotation rotation = make_rotation(bodies.transforms[i].rotation);
		float3 inertia_inverse = make_float3(bodies.properties[i].inertia_inverse);
		
		float3x3 m = matrix(rotation);
		
		InertiaTransform transform = {};
		
		transform.xx = inertia_inverse.x*m.c0.x*m.c0.x + inertia_inverse.y*m.c1.x*m.c1.x + inertia_inverse.z*m.c2.x*m.c2.x;
		transform.yy = inertia_inverse.x*m.c0.y*m.c0.y + inertia_inverse.y*m.c1.y*m.c1.y + inertia_inverse.z*m.c2.y*m.c2.y;
		transform.zz = inertia_inverse.x*m.c0.z*m.c0.z + inertia_inverse.y*m.c1.z*m.c1.z + inertia_inverse.z*m.c2.z*m.c2.z;
		transform.xy = inertia_inverse.x*m.c0.x*m.c0.y + inertia_inverse.y*m.c1.x*m.c1.y + inertia_inverse.z*m.c2.x*m.c2.y;
		transform.xz = inertia_inverse.x*m.c0.x*m.c0.z + inertia_inverse.y*m.c1.x*m.c1.z + inertia_inverse.z*m.c2.x*m.c2.z;
		transform.yz = inertia_inverse.x*m.c0.y*m.c0.z + inertia_inverse.y*m.c1.y*m.c1.z + inertia_inverse.z*m.c2.y*m.c2.z;
		
		momentum_to_velocity[i] = transform;
		bodies.momentum[i].unused0 = bodies.properties[i].mass_inverse;
	}
	
	CachedContactImpulse* impulses = contact_impulses->data;
	
	uint32_t* constraint_to_contact = allocate_array<uint32_t>(memory, contacts.count*simdv_width32, 32);
	data->constraint_to_contact = constraint_to_contact;
	
	// Schedule contacts so there are no conflicts within a SIMD width.
	ContactSlotV* contact_slots = reserve_array<ContactSlotV>(memory, contacts.count, 32);
	unsigned contact_slot_count = 0;
	{
		Arena temporary = *memory;
		commit_array<ContactSlotV>(&temporary, contacts.count);
		
		static const unsigned bucket_count = 16;
		
		ContactPairV* vacant_pair_buckets[bucket_count];
		ContactSlotV* vacant_slot_buckets[bucket_count];
		unsigned bucket_vacancy_count[bucket_count] = {};
		
		simdv_int32 invalid_index = simd_int32::makev(~0u);
		
		for (unsigned i = 0; i < bucket_count; ++i) {
			vacant_pair_buckets[i] = allocate_array<ContactPairV>(&temporary, contacts.count+1, 32);
			vacant_slot_buckets[i] = allocate_array<ContactSlotV>(&temporary, contacts.count, 32);
			
			// Add padding with invalid data so we don't have to range check.
			simd_int32::storev((int32_t*)vacant_pair_buckets[i]->ab, invalid_index);
		}
		
		for (unsigned i = 0; i < contacts.count; ++i) {
			unsigned index = contact_order[i];
			BodyPair bodies = contacts.bodies[index];
			
			unsigned bucket = i % bucket_count;
			ContactPairV* vacant_pairs = vacant_pair_buckets[bucket];
			ContactSlotV* vacant_slots = vacant_slot_buckets[bucket];
			unsigned vacancy_count = bucket_vacancy_count[bucket];
			
			// Ignore dependencies on body 0.
			unsigned ca = bodies.a ? bodies.a : bodies.b;
			unsigned cb = bodies.b ? bodies.b : bodies.a;
			
#ifdef __AVX2__
			__m256i a = _mm256_set1_epi16(ca);
			__m256i b = _mm256_set1_epi16(cb);
			
			__m256i scheduled_a_b;
			
			unsigned j = 0;
			
			for (;; ++j) {
				scheduled_a_b = _mm256_load_si256((const __m256i*)vacant_pairs[j].ab);
				
				__m256i conflict = _mm256_packs_epi16(_mm256_cmpeq_epi16(a, scheduled_a_b), _mm256_cmpeq_epi16(b, scheduled_a_b));
				
				if (!_mm256_movemask_epi8(conflict))
					break;
			}
			
			unsigned lane = first_set_bit((unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(scheduled_a_b, invalid_index))));
#else
			__m128i a = _mm_set1_epi16(ca);
			__m128i b = _mm_set1_epi16(cb);
			
			__m128i scheduled_a_b;
			
			unsigned j = 0;
			
			for (;; ++j) {
				scheduled_a_b = _mm_load_si128((const __m128i*)vacant_pairs[j].ab);
				
				__m128i conflict = _mm_packs_epi16(_mm_cmpeq_epi16(a, scheduled_a_b), _mm_cmpeq_epi16(b, scheduled_a_b));
				
				if (!_mm_movemask_epi8(conflict))
					break;
			}
			
			unsigned lane = first_set_bit((unsigned)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(scheduled_a_b, invalid_index))));
#endif
			
			ContactSlotV* slot = vacant_slots + j;
			ContactPairV* pair = vacant_pairs + j;
			
			slot->indices[lane] = index;
			
#ifdef __AVX2__
			_mm_store_ss((float*)pair->ab + lane, _mm_castsi128_ps(_mm_unpacklo_epi16(simd::extract_low(a), simd::extract_low(b))));
#else
			_mm_store_ss((float*)pair->ab + lane, _mm_castsi128_ps(_mm_unpacklo_epi16(a, b)));
#endif
			
			if (j == vacancy_count) {
				++vacancy_count;
			}
			else if (lane == simdv_width32-1) {
				simdv_int32 indices = simd_int32::loadv((const int32_t*)slot->indices);
				
				--vacancy_count;
				
				ContactPairV* last_pair = vacant_pairs + vacancy_count;
				ContactSlotV* last_slot = vacant_slots + vacancy_count;
				
				simd_int32::storev((int32_t*)contact_slots[contact_slot_count++].indices, indices);
				
				*pair = *last_pair;
				*slot = *last_slot;
			}
			else {
				continue;
			}
			
			// Store count and maintain padding.
			bucket_vacancy_count[bucket] = vacancy_count;
			simd_int32::storev((int32_t*)vacant_pairs[vacancy_count].ab, invalid_index);
		}
		
		for (unsigned i = 0; i < bucket_count; ++i) {
			ContactPairV* vacant_pairs = vacant_pair_buckets[i];
			ContactSlotV* vacant_slots = vacant_slot_buckets[i];
			unsigned vacancy_count = bucket_vacancy_count[i];
			
			// Replace any unset indices with the first one, which is always valid.
			// This is safe because the slots will just overwrite each other.
			for (unsigned i = 0; i < vacancy_count; ++i) {
				simdv_int32 ab = simd_int32::loadv((int32_t*)vacant_pairs[i].ab);
				simdv_int32 indices = simd_int32::loadv((const int32_t*)vacant_slots[i].indices);
				
				simdv_int32 mask = simd_int32::cmp_eq(ab, invalid_index);
				simdv_int32 first_index = simd128::shuffle32<0, 0, 0, 0>(indices);
				
#if NUDGE_SIMDV_WIDTH == 256
				first_index = simd256::shuffle128<0,0>(first_index);
#endif
				
				indices = simd::blendv32(indices, first_index, mask);
				
				simd_int32::storev((int32_t*)contact_slots[contact_slot_count++].indices, indices);
			}
		}
	}
	commit_array<ContactSlotV>(memory, contact_slot_count);
	
	ContactConstraintV* constraints = allocate_array<ContactConstraintV>(memory, contact_slot_count, 32);
	ContactConstraintStateV* constraint_states = allocate_array<ContactConstraintStateV>(memory, contact_slot_count, 32);
	
	data->constraints = constraints;
	data->constraint_states = constraint_states;
	
	memset(constraint_states, 0, sizeof(ContactConstraintStateV)*contact_slot_count);
	
	for (unsigned i = 0; i < contact_slot_count; ++i) {
		ContactSlotV slot = contact_slots[i];
		
		for (unsigned j = 0; j < simdv_width32; ++j)
			constraint_to_contact[i*simdv_width32 + j] = slot.indices[j];
		
		simdv_float position_x, position_y, position_z, penetration;
		simdv_float normal_x, normal_y, normal_z, friction;
		load8<sizeof(contacts.data[0]), 1>((const float*)contacts.data, slot.indices,
										   position_x, position_y, position_z, penetration,
										   normal_x, normal_y, normal_z, friction);
		
		NUDGE_SIMDV_ALIGNED uint16_t ab_array[simdv_width32*2];
		
		for (unsigned j = 0; j < simdv_width32; ++j) {
			BodyPair pair = contacts.bodies[slot.indices[j]];
			ab_array[j*2 + 0] = pair.a;
			ab_array[j*2 + 1] = pair.b;
		}
		
		unsigned a0 = ab_array[0]; unsigned a1 = ab_array[2]; unsigned a2 = ab_array[4]; unsigned a3 = ab_array[6];
		unsigned b0 = ab_array[1]; unsigned b1 = ab_array[3]; unsigned b2 = ab_array[5]; unsigned b3 = ab_array[7];
		
#if NUDGE_SIMDV_WIDTH == 256
		unsigned a4 = ab_array[8]; unsigned a5 = ab_array[10]; unsigned a6 = ab_array[12]; unsigned a7 = ab_array[14];
		unsigned b4 = ab_array[9]; unsigned b5 = ab_array[11]; unsigned b6 = ab_array[13]; unsigned b7 = ab_array[15];
		
		simdv_float a_mass_inverse = simd_float::make8(bodies.momentum[a0].unused0, bodies.momentum[a1].unused0, bodies.momentum[a2].unused0, bodies.momentum[a3].unused0,
													 bodies.momentum[a4].unused0, bodies.momentum[a5].unused0, bodies.momentum[a6].unused0, bodies.momentum[a7].unused0);
		simdv_float b_mass_inverse = simd_float::make8(bodies.momentum[b0].unused0, bodies.momentum[b1].unused0, bodies.momentum[b2].unused0, bodies.momentum[b3].unused0,
													 bodies.momentum[b4].unused0, bodies.momentum[b5].unused0, bodies.momentum[b6].unused0, bodies.momentum[b7].unused0);
#else
		simdv_float a_mass_inverse = simd_float::make4(bodies.momentum[a0].unused0, bodies.momentum[a1].unused0, bodies.momentum[a2].unused0, bodies.momentum[a3].unused0);
		simdv_float b_mass_inverse = simd_float::make4(bodies.momentum[b0].unused0, bodies.momentum[b1].unused0, bodies.momentum[b2].unused0, bodies.momentum[b3].unused0);
#endif
		
		simdv_float a_position_x, a_position_y, a_position_z, a_position_w;
		simdv_float b_position_x, b_position_y, b_position_z, b_position_w;
		load4<sizeof(bodies.transforms[0]), 2>(bodies.transforms[0].position, ab_array,
											   a_position_x, a_position_y, a_position_z, a_position_w);
		load4<sizeof(bodies.transforms[0]), 2>(bodies.transforms[0].position, ab_array + 1,
											   b_position_x, b_position_y, b_position_z, b_position_w);
		
		simdv_float pa_x = position_x - a_position_x;
		simdv_float pa_y = position_y - a_position_y;
		simdv_float pa_z = position_z - a_position_z;
		
		simdv_float pb_x = position_x - b_position_x;
		simdv_float pb_y = position_y - b_position_y;
		simdv_float pb_z = position_z - b_position_z;
		
		simdv_float a_momentum_to_velocity_xx, a_momentum_to_velocity_yy, a_momentum_to_velocity_zz, a_momentum_to_velocity_u0;
		simdv_float a_momentum_to_velocity_xy, a_momentum_to_velocity_xz, a_momentum_to_velocity_yz, a_momentum_to_velocity_u1;
		load8<sizeof(momentum_to_velocity[0]), 2>((const float*)momentum_to_velocity, ab_array,
												  a_momentum_to_velocity_xx, a_momentum_to_velocity_yy, a_momentum_to_velocity_zz, a_momentum_to_velocity_u0,
												  a_momentum_to_velocity_xy, a_momentum_to_velocity_xz, a_momentum_to_velocity_yz, a_momentum_to_velocity_u1);
		
		simdv_float na_xt, na_yt, na_zt;
		simd_soa::cross(pa_x, pa_y, pa_z, normal_x, normal_y, normal_z, na_xt, na_yt, na_zt);
		
		simdv_float na_x = a_momentum_to_velocity_xx*na_xt + a_momentum_to_velocity_xy*na_yt + a_momentum_to_velocity_xz*na_zt;
		simdv_float na_y = a_momentum_to_velocity_xy*na_xt + a_momentum_to_velocity_yy*na_yt + a_momentum_to_velocity_yz*na_zt;
		simdv_float na_z = a_momentum_to_velocity_xz*na_xt + a_momentum_to_velocity_yz*na_yt + a_momentum_to_velocity_zz*na_zt;
		
		simdv_float b_momentum_to_velocity_xx, b_momentum_to_velocity_yy, b_momentum_to_velocity_zz, b_momentum_to_velocity_u0;
		simdv_float b_momentum_to_velocity_xy, b_momentum_to_velocity_xz, b_momentum_to_velocity_yz, b_momentum_to_velocity_u1;
		load8<sizeof(momentum_to_velocity[0]), 2>((const float*)momentum_to_velocity, ab_array + 1,
												  b_momentum_to_velocity_xx, b_momentum_to_velocity_yy, b_momentum_to_velocity_zz, b_momentum_to_velocity_u0,
												  b_momentum_to_velocity_xy, b_momentum_to_velocity_xz, b_momentum_to_velocity_yz, b_momentum_to_velocity_u1);
		
		simdv_float nb_xt, nb_yt, nb_zt;
		simd_soa::cross(pb_x, pb_y, pb_z, normal_x, normal_y, normal_z, nb_xt, nb_yt, nb_zt);
		
		simdv_float nb_x = b_momentum_to_velocity_xx*nb_xt + b_momentum_to_velocity_xy*nb_yt + b_momentum_to_velocity_xz*nb_zt;
		simdv_float nb_y = b_momentum_to_velocity_xy*nb_xt + b_momentum_to_velocity_yy*nb_yt + b_momentum_to_velocity_yz*nb_zt;
		simdv_float nb_z = b_momentum_to_velocity_xz*nb_xt + b_momentum_to_velocity_yz*nb_yt + b_momentum_to_velocity_zz*nb_zt;
		
		simd_soa::cross(na_x, na_y, na_z, pa_x, pa_y, pa_z, na_xt, na_yt, na_zt);
		simd_soa::cross(nb_x, nb_y, nb_z, pb_x, pb_y, pb_z, nb_xt, nb_yt, nb_zt);
		
		simdv_float normal_impulse_to_rotational_velocity_x = na_xt + nb_xt;
		simdv_float normal_impulse_to_rotational_velocity_y = na_yt + nb_yt;
		simdv_float normal_impulse_to_rotational_velocity_z = na_zt + nb_zt;
		
		simdv_float r_dot_n = normal_impulse_to_rotational_velocity_x*normal_x + normal_impulse_to_rotational_velocity_y*normal_y + normal_impulse_to_rotational_velocity_z*normal_z;
		
		simdv_float mass_inverse = a_mass_inverse + b_mass_inverse;
		simdv_float normal_velocity_to_normal_impulse = mass_inverse + r_dot_n;
		
		simdv_float nonzero = simd_float::cmp_neq(normal_velocity_to_normal_impulse, simd_float::zerov());
		normal_velocity_to_normal_impulse = simd::bitwise_and(simd_float::makev(-1.0f) / normal_velocity_to_normal_impulse, nonzero);
		
		simdv_float bias = simd_float::makev(-bias_factor) * simd_float::max(penetration - simd_float::makev(allowed_penetration), simd_float::zerov()) * normal_velocity_to_normal_impulse;
		
		// Compute a tangent from the normal. Care is taken to compute a smoothly varying basis to improve stability.
		simdv_float s = simd_float::abs(normal_x);
		
		simdv_float u_x = normal_z*s;
		simdv_float u_y = u_x - normal_z;
		simdv_float u_z = simd_float::madd(normal_x - normal_y, s, normal_y);
		
		u_x = simd::bitwise_xor(u_x, simd_float::makev(-0.0f));
		simd_soa::normalize(u_x, u_y, u_z);
		
		// Compute the rest of the basis.
		simdv_float v_x, v_y, v_z;
		simd_soa::cross(u_x, u_y, u_z, normal_x, normal_y, normal_z, v_x, v_y, v_z);
		
		simdv_float ua_x, ua_y, ua_z, va_x, va_y, va_z;
		simd_soa::cross(pa_x, pa_y, pa_z, u_x, u_y, u_z, ua_x, ua_y, ua_z);
		simd_soa::cross(pa_x, pa_y, pa_z, v_x, v_y, v_z, va_x, va_y, va_z);
		
		simdv_float ub_x, ub_y, ub_z, vb_x, vb_y, vb_z;
		simd_soa::cross(pb_x, pb_y, pb_z, u_x, u_y, u_z, ub_x, ub_y, ub_z);
		simd_soa::cross(pb_x, pb_y, pb_z, v_x, v_y, v_z, vb_x, vb_y, vb_z);
		
		simdv_float a_duu = a_momentum_to_velocity_xx*ua_x*ua_x + a_momentum_to_velocity_yy*ua_y*ua_y + a_momentum_to_velocity_zz*ua_z*ua_z;
		simdv_float a_dvv = a_momentum_to_velocity_xx*va_x*va_x + a_momentum_to_velocity_yy*va_y*va_y + a_momentum_to_velocity_zz*va_z*va_z;
		simdv_float a_duv = a_momentum_to_velocity_xx*ua_x*va_x + a_momentum_to_velocity_yy*ua_y*va_y + a_momentum_to_velocity_zz*ua_z*va_z;
		
		simdv_float a_suu = a_momentum_to_velocity_xy*ua_x*ua_y + a_momentum_to_velocity_xz*ua_x*ua_z + a_momentum_to_velocity_yz*ua_y*ua_z;
		simdv_float a_svv = a_momentum_to_velocity_xy*va_x*va_y + a_momentum_to_velocity_xz*va_x*va_z + a_momentum_to_velocity_yz*va_y*va_z;
		simdv_float a_suv = a_momentum_to_velocity_xy*(ua_x*va_y + ua_y*va_x) + a_momentum_to_velocity_xz*(ua_x*va_z + ua_z*va_x) + a_momentum_to_velocity_yz*(ua_y*va_z + ua_z*va_y);
		
		simdv_float b_duu = b_momentum_to_velocity_xx*ub_x*ub_x + b_momentum_to_velocity_yy*ub_y*ub_y + b_momentum_to_velocity_zz*ub_z*ub_z;
		simdv_float b_dvv = b_momentum_to_velocity_xx*vb_x*vb_x + b_momentum_to_velocity_yy*vb_y*vb_y + b_momentum_to_velocity_zz*vb_z*vb_z;
		simdv_float b_duv = b_momentum_to_velocity_xx*ub_x*vb_x + b_momentum_to_velocity_yy*ub_y*vb_y + b_momentum_to_velocity_zz*ub_z*vb_z;
		
		simdv_float b_suu = b_momentum_to_velocity_xy*ub_x*ub_y + b_momentum_to_velocity_xz*ub_x*ub_z + b_momentum_to_velocity_yz*ub_y*ub_z;
		simdv_float b_svv = b_momentum_to_velocity_xy*vb_x*vb_y + b_momentum_to_velocity_xz*vb_x*vb_z + b_momentum_to_velocity_yz*vb_y*vb_z;
		simdv_float b_suv = b_momentum_to_velocity_xy*(ub_x*vb_y + ub_y*vb_x) + b_momentum_to_velocity_xz*(ub_x*vb_z + ub_z*vb_x) + b_momentum_to_velocity_yz*(ub_y*vb_z + ub_z*vb_y);
		
		simdv_float friction_x = mass_inverse + a_duu + a_suu + a_suu + b_duu + b_suu + b_suu;
		simdv_float friction_y = mass_inverse + a_dvv + a_svv + a_svv + b_dvv + b_svv + b_svv;
		simdv_float friction_z = a_duv + a_duv + a_suv + a_suv + b_duv + b_duv + b_suv + b_suv;
		
		simdv_float ua_xt = a_momentum_to_velocity_xx*ua_x + a_momentum_to_velocity_xy*ua_y + a_momentum_to_velocity_xz*ua_z;
		simdv_float ua_yt = a_momentum_to_velocity_xy*ua_x + a_momentum_to_velocity_yy*ua_y + a_momentum_to_velocity_yz*ua_z;
		simdv_float ua_zt = a_momentum_to_velocity_xz*ua_x + a_momentum_to_velocity_yz*ua_y + a_momentum_to_velocity_zz*ua_z;
		
		simdv_float va_xt = a_momentum_to_velocity_xx*va_x + a_momentum_to_velocity_xy*va_y + a_momentum_to_velocity_xz*va_z;
		simdv_float va_yt = a_momentum_to_velocity_xy*va_x + a_momentum_to_velocity_yy*va_y + a_momentum_to_velocity_yz*va_z;
		simdv_float va_zt = a_momentum_to_velocity_xz*va_x + a_momentum_to_velocity_yz*va_y + a_momentum_to_velocity_zz*va_z;
		
		simdv_float ub_xt = b_momentum_to_velocity_xx*ub_x + b_momentum_to_velocity_xy*ub_y + b_momentum_to_velocity_xz*ub_z;
		simdv_float ub_yt = b_momentum_to_velocity_xy*ub_x + b_momentum_to_velocity_yy*ub_y + b_momentum_to_velocity_yz*ub_z;
		simdv_float ub_zt = b_momentum_to_velocity_xz*ub_x + b_momentum_to_velocity_yz*ub_y + b_momentum_to_velocity_zz*ub_z;
		
		simdv_float vb_xt = b_momentum_to_velocity_xx*vb_x + b_momentum_to_velocity_xy*vb_y + b_momentum_to_velocity_xz*vb_z;
		simdv_float vb_yt = b_momentum_to_velocity_xy*vb_x + b_momentum_to_velocity_yy*vb_y + b_momentum_to_velocity_yz*vb_z;
		simdv_float vb_zt = b_momentum_to_velocity_xz*vb_x + b_momentum_to_velocity_yz*vb_y + b_momentum_to_velocity_zz*vb_z;
		
		constraints[i].a[0] = a0; constraints[i].a[1] = a1; constraints[i].a[2] = a2; constraints[i].a[3] = a3;
		constraints[i].b[0] = b0; constraints[i].b[1] = b1; constraints[i].b[2] = b2; constraints[i].b[3] = b3;
		
#if NUDGE_SIMDV_WIDTH == 256
		constraints[i].a[4] = a4; constraints[i].a[5] = a5; constraints[i].a[6] = a6; constraints[i].a[7] = a7;
		constraints[i].b[4] = b4; constraints[i].b[5] = b5; constraints[i].b[6] = b6; constraints[i].b[7] = b7;
#endif
		
		simd_float::storev(constraints[i].n_x, normal_x);
		simd_float::storev(constraints[i].n_y, normal_y);
		simd_float::storev(constraints[i].n_z, normal_z);
		
		simd_float::storev(constraints[i].pa_x, pa_x);
		simd_float::storev(constraints[i].pa_y, pa_y);
		simd_float::storev(constraints[i].pa_z, pa_z);
		
		simd_float::storev(constraints[i].pb_x, pb_x);
		simd_float::storev(constraints[i].pb_y, pb_y);
		simd_float::storev(constraints[i].pb_z, pb_z);
		
		simd_float::storev(constraints[i].normal_velocity_to_normal_impulse, normal_velocity_to_normal_impulse);
		
		simd_float::storev(constraints[i].bias, bias);
		simd_float::storev(constraints[i].friction, friction);
		
		simd_float::storev(constraints[i].u_x, u_x);
		simd_float::storev(constraints[i].u_y, u_y);
		simd_float::storev(constraints[i].u_z, u_z);
		
		simd_float::storev(constraints[i].v_x, v_x);
		simd_float::storev(constraints[i].v_y, v_y);
		simd_float::storev(constraints[i].v_z, v_z);
		
		simd_float::storev(constraints[i].friction_coefficient_x, friction_x);
		simd_float::storev(constraints[i].friction_coefficient_y, friction_y);
		simd_float::storev(constraints[i].friction_coefficient_z, friction_z);
		
		simd_float::storev(constraints[i].ua_x, -ua_xt);
		simd_float::storev(constraints[i].ua_y, -ua_yt);
		simd_float::storev(constraints[i].ua_z, -ua_zt);
		
		simd_float::storev(constraints[i].va_x, -va_xt);
		simd_float::storev(constraints[i].va_y, -va_yt);
		simd_float::storev(constraints[i].va_z, -va_zt);
		
		simd_float::storev(constraints[i].na_x, -na_x);
		simd_float::storev(constraints[i].na_y, -na_y);
		simd_float::storev(constraints[i].na_z, -na_z);
		
		simd_float::storev(constraints[i].ub_x, ub_xt);
		simd_float::storev(constraints[i].ub_y, ub_yt);
		simd_float::storev(constraints[i].ub_z, ub_zt);
		
		simd_float::storev(constraints[i].vb_x, vb_xt);
		simd_float::storev(constraints[i].vb_y, vb_yt);
		simd_float::storev(constraints[i].vb_z, vb_zt);
		
		simd_float::storev(constraints[i].nb_x, nb_x);
		simd_float::storev(constraints[i].nb_y, nb_y);
		simd_float::storev(constraints[i].nb_z, nb_z);
		
		simdv_float cached_impulse_x, cached_impulse_y, cached_impulse_z, unused0;
		load4<sizeof(impulses[0]), 1>((const float*)impulses, slot.indices,
									  cached_impulse_x, cached_impulse_y, cached_impulse_z, unused0);
		
		simdv_float a_velocity_x, a_velocity_y, a_velocity_z;
		simdv_float a_angular_velocity_x, a_angular_velocity_y, a_angular_velocity_z, a_angular_velocity_w;
		load8<sizeof(bodies.momentum[0]), 1>((const float*)bodies.momentum, constraints[i].a,
											 a_velocity_x, a_velocity_y, a_velocity_z, a_mass_inverse,
											 a_angular_velocity_x, a_angular_velocity_y, a_angular_velocity_z, a_angular_velocity_w);
		
		simdv_float b_velocity_x, b_velocity_y, b_velocity_z;
		simdv_float b_angular_velocity_x, b_angular_velocity_y, b_angular_velocity_z, b_angular_velocity_w;
		load8<sizeof(bodies.momentum[0]), 1>((const float*)bodies.momentum, constraints[i].b,
											 b_velocity_x, b_velocity_y, b_velocity_z, b_mass_inverse,
											 b_angular_velocity_x, b_angular_velocity_y, b_angular_velocity_z, b_angular_velocity_w);
		
		simdv_float normal_impulse = simd_float::max(normal_x*cached_impulse_x + normal_y*cached_impulse_y + normal_z*cached_impulse_z, simd_float::zerov());
		simdv_float max_friction_impulse = normal_impulse * friction;
		
		simdv_float friction_impulse_x = u_x*cached_impulse_x + u_y*cached_impulse_y + u_z*cached_impulse_z;
		simdv_float friction_impulse_y = v_x*cached_impulse_x + v_y*cached_impulse_y + v_z*cached_impulse_z;
		
		simdv_float friction_clamp_scale = friction_impulse_x*friction_impulse_x + friction_impulse_y*friction_impulse_y;
		
		friction_clamp_scale = simd_float::rsqrt(friction_clamp_scale);
		friction_clamp_scale = friction_clamp_scale * max_friction_impulse;
		friction_clamp_scale = simd_float::min(simd_float::makev(1.0f), friction_clamp_scale); // Note: First operand is returned on NaN.
		
		friction_impulse_x = friction_impulse_x * friction_clamp_scale;
		friction_impulse_y = friction_impulse_y * friction_clamp_scale;
		
		simdv_float linear_impulse_x = friction_impulse_x*u_x + friction_impulse_y*v_x + normal_x * normal_impulse;
		simdv_float linear_impulse_y = friction_impulse_x*u_y + friction_impulse_y*v_y + normal_y * normal_impulse;
		simdv_float linear_impulse_z = friction_impulse_x*u_z + friction_impulse_y*v_z + normal_z * normal_impulse;
		
		simdv_float a_angular_impulse_x = friction_impulse_x*simd_float::loadv(constraints[i].ua_x) + friction_impulse_y*simd_float::loadv(constraints[i].va_x) + normal_impulse*simd_float::loadv(constraints[i].na_x);
		simdv_float a_angular_impulse_y = friction_impulse_x*simd_float::loadv(constraints[i].ua_y) + friction_impulse_y*simd_float::loadv(constraints[i].va_y) + normal_impulse*simd_float::loadv(constraints[i].na_y);
		simdv_float a_angular_impulse_z = friction_impulse_x*simd_float::loadv(constraints[i].ua_z) + friction_impulse_y*simd_float::loadv(constraints[i].va_z) + normal_impulse*simd_float::loadv(constraints[i].na_z);
		
		simdv_float b_angular_impulse_x = friction_impulse_x*simd_float::loadv(constraints[i].ub_x) + friction_impulse_y*simd_float::loadv(constraints[i].vb_x) + normal_impulse*simd_float::loadv(constraints[i].nb_x);
		simdv_float b_angular_impulse_y = friction_impulse_x*simd_float::loadv(constraints[i].ub_y) + friction_impulse_y*simd_float::loadv(constraints[i].vb_y) + normal_impulse*simd_float::loadv(constraints[i].nb_y);
		simdv_float b_angular_impulse_z = friction_impulse_x*simd_float::loadv(constraints[i].ub_z) + friction_impulse_y*simd_float::loadv(constraints[i].vb_z) + normal_impulse*simd_float::loadv(constraints[i].nb_z);
		
		a_velocity_x -= linear_impulse_x * a_mass_inverse;
		a_velocity_y -= linear_impulse_y * a_mass_inverse;
		a_velocity_z -= linear_impulse_z * a_mass_inverse;
		
		a_angular_velocity_x += a_angular_impulse_x;
		a_angular_velocity_y += a_angular_impulse_y;
		a_angular_velocity_z += a_angular_impulse_z;
		
		b_velocity_x += linear_impulse_x * b_mass_inverse;
		b_velocity_y += linear_impulse_y * b_mass_inverse;
		b_velocity_z += linear_impulse_z * b_mass_inverse;
		
		b_angular_velocity_x += b_angular_impulse_x;
		b_angular_velocity_y += b_angular_impulse_y;
		b_angular_velocity_z += b_angular_impulse_z;
		
		simd_float::storev(constraint_states[i].applied_normal_impulse, normal_impulse);
		simd_float::storev(constraint_states[i].applied_friction_impulse_x, friction_impulse_x);
		simd_float::storev(constraint_states[i].applied_friction_impulse_y, friction_impulse_y);
		
		store8<sizeof(bodies.momentum[0]), 1>((float*)bodies.momentum, constraints[i].a,
											  a_velocity_x, a_velocity_y, a_velocity_z, a_mass_inverse,
											  a_angular_velocity_x, a_angular_velocity_y, a_angular_velocity_z, a_angular_velocity_w);
		
		store8<sizeof(bodies.momentum[0]), 1>((float*)bodies.momentum, constraints[i].b,
											  b_velocity_x, b_velocity_y, b_velocity_z, b_mass_inverse,
											  b_angular_velocity_x, b_angular_velocity_y, b_angular_velocity_z, b_angular_velocity_w);
	}
	
	data->constraint_batches = contact_slot_count;
	
	return data;
}

void apply_impulses(ContactConstraintData* data, BodyData bodies) {
	ContactConstraintV* constraints = data->constraints;
	ContactConstraintStateV* constraint_states = data->constraint_states;
	
	unsigned constraint_batches = data->constraint_batches;
	
	for (unsigned i = 0; i < constraint_batches; ++i) {
		const ContactConstraintV& constraint = constraints[i];
		
		simdv_float a_velocity_x, a_velocity_y, a_velocity_z, a_mass_inverse;
		simdv_float a_angular_velocity_x, a_angular_velocity_y, a_angular_velocity_z, a_angular_velocity_w;
		load8<sizeof(bodies.momentum[0]), 1>((const float*)bodies.momentum, constraint.a,
											 a_velocity_x, a_velocity_y, a_velocity_z, a_mass_inverse,
											 a_angular_velocity_x, a_angular_velocity_y, a_angular_velocity_z, a_angular_velocity_w);
		
		simdv_float pa_z = simd_float::loadv(constraint.pa_z);
		simdv_float pa_x = simd_float::loadv(constraint.pa_x);
		simdv_float pa_y = simd_float::loadv(constraint.pa_y);
		
		simdv_float v_xa = simd_float::madd(a_angular_velocity_y, pa_z, a_velocity_x);
		simdv_float v_ya = simd_float::madd(a_angular_velocity_z, pa_x, a_velocity_y);
		simdv_float v_za = simd_float::madd(a_angular_velocity_x, pa_y, a_velocity_z);
		
		simdv_float b_velocity_x, b_velocity_y, b_velocity_z, b_mass_inverse;
		simdv_float b_angular_velocity_x, b_angular_velocity_y, b_angular_velocity_z, b_angular_velocity_w;
		load8<sizeof(bodies.momentum[0]), 1>((const float*)bodies.momentum, constraint.b,
											 b_velocity_x, b_velocity_y, b_velocity_z, b_mass_inverse,
											 b_angular_velocity_x, b_angular_velocity_y, b_angular_velocity_z, b_angular_velocity_w);
		
		simdv_float pb_z = simd_float::loadv(constraint.pb_z);
		simdv_float pb_x = simd_float::loadv(constraint.pb_x);
		simdv_float pb_y = simd_float::loadv(constraint.pb_y);
		
		simdv_float v_xb = simd_float::madd(b_angular_velocity_y, pb_z, b_velocity_x);
		simdv_float v_yb = simd_float::madd(b_angular_velocity_z, pb_x, b_velocity_y);
		simdv_float v_zb = simd_float::madd(b_angular_velocity_x, pb_y, b_velocity_z);
		
		v_xa = simd_float::madd(b_angular_velocity_z, pb_y, v_xa);
		v_ya = simd_float::madd(b_angular_velocity_x, pb_z, v_ya);
		v_za = simd_float::madd(b_angular_velocity_y, pb_x, v_za);
		
		simdv_float n_x = simd_float::loadv(constraint.n_x);
		simdv_float fu_x = simd_float::loadv(constraint.u_x);
		simdv_float fv_x = simd_float::loadv(constraint.v_x);
		
		v_xb = simd_float::madd(a_angular_velocity_z, pa_y, v_xb);
		v_yb = simd_float::madd(a_angular_velocity_x, pa_z, v_yb);
		v_zb = simd_float::madd(a_angular_velocity_y, pa_x, v_zb);
		
		simdv_float n_y = simd_float::loadv(constraint.n_y);
		simdv_float fu_y = simd_float::loadv(constraint.u_y);
		simdv_float fv_y = simd_float::loadv(constraint.v_y);
		
		simdv_float v_x = v_xb - v_xa;
		simdv_float v_y = v_yb - v_ya;
		simdv_float v_z = v_zb - v_za;
		
		simdv_float t_z = n_x * v_x;
		simdv_float t_x = v_x * fu_x;
		simdv_float t_y = v_x * fv_x;
		
		simdv_float n_z = simd_float::loadv(constraint.n_z);
		simdv_float fu_z = simd_float::loadv(constraint.u_z);
		simdv_float fv_z = simd_float::loadv(constraint.v_z);
		
		simdv_float normal_bias = simd_float::loadv(constraint.bias);
		simdv_float old_normal_impulse = simd_float::loadv(constraint_states[i].applied_normal_impulse);
		simdv_float normal_factor = simd_float::loadv(constraint.normal_velocity_to_normal_impulse);
		
		t_z = simd_float::madd(n_y, v_y, t_z);
		t_x = simd_float::madd(v_y, fu_y, t_x);
		t_y = simd_float::madd(v_y, fv_y, t_y);
		
		normal_bias = normal_bias + old_normal_impulse;
		
		t_z = simd_float::madd(n_z, v_z, t_z);
		t_x = simd_float::madd(v_z, fu_z, t_x);
		t_y = simd_float::madd(v_z, fv_z, t_y);
		
		simdv_float normal_impulse = simd_float::madd(normal_factor, t_z, normal_bias);
		
		simdv_float t_xx = t_x*t_x;
		simdv_float t_yy = t_y*t_y;
		simdv_float t_xy = t_x*t_y;
		simdv_float tl2 = t_xx + t_yy;
		
		normal_impulse = simd_float::max(normal_impulse, simd_float::zerov());
		
		t_x *= tl2;
		t_y *= tl2;
		
		simd_float::storev(constraint_states[i].applied_normal_impulse, normal_impulse);
		
		simdv_float max_friction_impulse = normal_impulse * simd_float::loadv(constraint.friction);
		normal_impulse = normal_impulse - old_normal_impulse;
		
		simdv_float friction_x = simd_float::loadv(constraint.friction_coefficient_x);
		simdv_float friction_factor = t_xx * friction_x;
		simdv_float linear_impulse_x = n_x * normal_impulse;
		
		simdv_float friction_y = simd_float::loadv(constraint.friction_coefficient_y);
		friction_factor = simd_float::madd(t_yy, friction_y, friction_factor);
		simdv_float linear_impulse_y = n_y * normal_impulse;
		
		simdv_float friction_z = simd_float::loadv(constraint.friction_coefficient_z);
		friction_factor = simd_float::madd(t_xy, friction_z, friction_factor);
		simdv_float linear_impulse_z = n_z * normal_impulse;
		
		friction_factor = simd_float::recip(friction_factor);
		
		simdv_float na_x = simd_float::loadv(constraint.na_x);
		simdv_float na_y = simd_float::loadv(constraint.na_y);
		simdv_float na_z = simd_float::loadv(constraint.na_z);
		
		a_angular_velocity_x = simd_float::madd(na_x, normal_impulse, a_angular_velocity_x);
		a_angular_velocity_y = simd_float::madd(na_y, normal_impulse, a_angular_velocity_y);
		a_angular_velocity_z = simd_float::madd(na_z, normal_impulse, a_angular_velocity_z);
		
		simdv_float old_friction_impulse_x = simd_float::loadv(constraint_states[i].applied_friction_impulse_x);
		simdv_float old_friction_impulse_y = simd_float::loadv(constraint_states[i].applied_friction_impulse_y);
		
		friction_factor = simd_float::min(simd_float::makev(1e+6f), friction_factor); // Note: First operand is returned on NaN.
		
		simdv_float friction_impulse_x = t_x*friction_factor;
		simdv_float friction_impulse_y = t_y*friction_factor;
		
		friction_impulse_x = old_friction_impulse_x - friction_impulse_x; // Note: Friction impulse has the wrong sign until this point. This is really an addition.
		friction_impulse_y = old_friction_impulse_y - friction_impulse_y;
		
		simdv_float friction_clamp_scale = friction_impulse_x*friction_impulse_x + friction_impulse_y*friction_impulse_y;
		
		simdv_float nb_x = simd_float::loadv(constraint.nb_x);
		simdv_float nb_y = simd_float::loadv(constraint.nb_y);
		simdv_float nb_z = simd_float::loadv(constraint.nb_z);
		
		friction_clamp_scale = simd_float::rsqrt(friction_clamp_scale);
		
		b_angular_velocity_x = simd_float::madd(nb_x, normal_impulse, b_angular_velocity_x);
		b_angular_velocity_y = simd_float::madd(nb_y, normal_impulse, b_angular_velocity_y);
		b_angular_velocity_z = simd_float::madd(nb_z, normal_impulse, b_angular_velocity_z);
		
		friction_clamp_scale = friction_clamp_scale * max_friction_impulse;
		friction_clamp_scale = simd_float::min(simd_float::makev(1.0f), friction_clamp_scale); // Note: First operand is returned on NaN.
		
		friction_impulse_x = friction_impulse_x * friction_clamp_scale;
		friction_impulse_y = friction_impulse_y * friction_clamp_scale;
		
		simd_float::storev(constraint_states[i].applied_friction_impulse_x, friction_impulse_x);
		simd_float::storev(constraint_states[i].applied_friction_impulse_y, friction_impulse_y);
		
		friction_impulse_x -= old_friction_impulse_x;
		friction_impulse_y -= old_friction_impulse_y;
		
		linear_impulse_x = simd_float::madd(fu_x, friction_impulse_x, linear_impulse_x);
		linear_impulse_y = simd_float::madd(fu_y, friction_impulse_x, linear_impulse_y);
		linear_impulse_z = simd_float::madd(fu_z, friction_impulse_x, linear_impulse_z);
		
		linear_impulse_x = simd_float::madd(fv_x, friction_impulse_y, linear_impulse_x);
		linear_impulse_y = simd_float::madd(fv_y, friction_impulse_y, linear_impulse_y);
		linear_impulse_z = simd_float::madd(fv_z, friction_impulse_y, linear_impulse_z);
		
		simdv_float a_mass_inverse_neg = simd::bitwise_xor(a_mass_inverse, simd_float::makev(-0.0f));
		
		a_velocity_x = simd_float::madd(linear_impulse_x, a_mass_inverse_neg, a_velocity_x);
		a_velocity_y = simd_float::madd(linear_impulse_y, a_mass_inverse_neg, a_velocity_y);
		a_velocity_z = simd_float::madd(linear_impulse_z, a_mass_inverse_neg, a_velocity_z);
		
		simdv_float ua_x = simd_float::loadv(constraint.ua_x);
		simdv_float ua_y = simd_float::loadv(constraint.ua_y);
		simdv_float ua_z = simd_float::loadv(constraint.ua_z);
		
		a_angular_velocity_x = simd_float::madd(ua_x, friction_impulse_x, a_angular_velocity_x);
		a_angular_velocity_y = simd_float::madd(ua_y, friction_impulse_x, a_angular_velocity_y);
		a_angular_velocity_z = simd_float::madd(ua_z, friction_impulse_x, a_angular_velocity_z);
		
		simdv_float va_x = simd_float::loadv(constraint.va_x);
		simdv_float va_y = simd_float::loadv(constraint.va_y);
		simdv_float va_z = simd_float::loadv(constraint.va_z);
		
		a_angular_velocity_x = simd_float::madd(va_x, friction_impulse_y, a_angular_velocity_x);
		a_angular_velocity_y = simd_float::madd(va_y, friction_impulse_y, a_angular_velocity_y);
		a_angular_velocity_z = simd_float::madd(va_z, friction_impulse_y, a_angular_velocity_z);
		
		a_angular_velocity_w = simd_float::zerov(); // Reduces register pressure.
		
		store8<sizeof(bodies.momentum[0]), 1>((float*)bodies.momentum, constraint.a,
											  a_velocity_x, a_velocity_y, a_velocity_z, a_mass_inverse,
											  a_angular_velocity_x, a_angular_velocity_y, a_angular_velocity_z, a_angular_velocity_w);
		
		b_velocity_x = simd_float::madd(linear_impulse_x, b_mass_inverse, b_velocity_x);
		b_velocity_y = simd_float::madd(linear_impulse_y, b_mass_inverse, b_velocity_y);
		b_velocity_z = simd_float::madd(linear_impulse_z, b_mass_inverse, b_velocity_z);
		
		simdv_float ub_x = simd_float::loadv(constraint.ub_x);
		simdv_float ub_y = simd_float::loadv(constraint.ub_y);
		simdv_float ub_z = simd_float::loadv(constraint.ub_z);
		
		b_angular_velocity_x = simd_float::madd(ub_x, friction_impulse_x, b_angular_velocity_x);
		b_angular_velocity_y = simd_float::madd(ub_y, friction_impulse_x, b_angular_velocity_y);
		b_angular_velocity_z = simd_float::madd(ub_z, friction_impulse_x, b_angular_velocity_z);
		
		simdv_float vb_x = simd_float::loadv(constraint.vb_x);
		simdv_float vb_y = simd_float::loadv(constraint.vb_y);
		simdv_float vb_z = simd_float::loadv(constraint.vb_z);
		
		b_angular_velocity_x = simd_float::madd(vb_x, friction_impulse_y, b_angular_velocity_x);
		b_angular_velocity_y = simd_float::madd(vb_y, friction_impulse_y, b_angular_velocity_y);
		b_angular_velocity_z = simd_float::madd(vb_z, friction_impulse_y, b_angular_velocity_z);
		
		b_angular_velocity_w = simd_float::zerov(); // Reduces register pressure.
		
		store8<sizeof(bodies.momentum[0]), 1>((float*)bodies.momentum, constraint.b,
											  b_velocity_x, b_velocity_y, b_velocity_z, b_mass_inverse,
											  b_angular_velocity_x, b_angular_velocity_y, b_angular_velocity_z, b_angular_velocity_w);
	}
}

void update_cached_impulses(ContactConstraintData* data, ContactImpulseData* contact_impulses) {
	uint32_t* constraint_to_contact = data->constraint_to_contact;
	
	ContactConstraintV* constraints = data->constraints;
	ContactConstraintStateV* constraint_states = data->constraint_states;
	unsigned constraint_count = data->constraint_batches * simdv_width32;
	
	for (unsigned i = 0; i < constraint_count; ++i) {
		unsigned contact = constraint_to_contact[i];
		
		unsigned b = i >> simdv_width32_log2;
		unsigned l = i & (simdv_width32-1);
		
		float* impulse = contact_impulses->data[contact].impulse;
		
		impulse[0] = (constraint_states[b].applied_normal_impulse[l] * constraints[b].n_x[l] +
					  constraint_states[b].applied_friction_impulse_x[l] * constraints[b].u_x[l] +
					  constraint_states[b].applied_friction_impulse_y[l] * constraints[b].v_x[l]);
		
		impulse[1] = (constraint_states[b].applied_normal_impulse[l] * constraints[b].n_y[l] +
					  constraint_states[b].applied_friction_impulse_x[l] * constraints[b].u_y[l] +
					  constraint_states[b].applied_friction_impulse_y[l] * constraints[b].v_y[l]);
		
		impulse[2] = (constraint_states[b].applied_normal_impulse[l] * constraints[b].n_z[l] +
					  constraint_states[b].applied_friction_impulse_x[l] * constraints[b].u_z[l] +
					  constraint_states[b].applied_friction_impulse_y[l] * constraints[b].v_z[l]);
	}
}

void advance(ActiveBodies active_bodies, BodyData bodies, float time_step) {
	float half_time_step = 0.5f * time_step;
	
	// TODO: Consider SIMD-optimizing this loop.
	for (unsigned n = 0; n < active_bodies.count; ++n) {
		unsigned i = active_bodies.indices[n];
		
		float3 velocity = make_float3(bodies.momentum[i].velocity);
		float3 angular_velocity = make_float3(bodies.momentum[i].angular_velocity);
		
		if (length2(velocity) < 1e-2f && length2(angular_velocity) < 1e-1f) {
			if (bodies.idle_counters[i] < 0xff)
				++bodies.idle_counters[i];
		}
		else {
			bodies.idle_counters[i] = 0;
		}
		
		Rotation dr = { angular_velocity };
		
		dr = dr * make_rotation(bodies.transforms[i].rotation);
		dr.v *= half_time_step;
		dr.s *= half_time_step;
		
		bodies.transforms[i].position[0] += velocity.x * time_step;
		bodies.transforms[i].position[1] += velocity.y * time_step;
		bodies.transforms[i].position[2] += velocity.z * time_step;
		
		bodies.transforms[i].rotation[0] += dr.v.x;
		bodies.transforms[i].rotation[1] += dr.v.y;
		bodies.transforms[i].rotation[2] += dr.v.z;
		bodies.transforms[i].rotation[3] += dr.s;
		
		Rotation rotation = normalize(make_rotation(bodies.transforms[i].rotation));
		
		bodies.transforms[i].rotation[0] = rotation.v.x;
		bodies.transforms[i].rotation[1] = rotation.v.y;
		bodies.transforms[i].rotation[2] = rotation.v.z;
		bodies.transforms[i].rotation[3] = rotation.s;
	}
}

}

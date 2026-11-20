// include/strata/kernels/cpu/tq_cpu.hpp - the trellis expert formats (TQ2_T / TQK6 / TQK7 / TQP5 / TQP6 / TQP7,
// ggml types 144-149 of the agention llama.cpp fork) on the CPU: decode-and-dot against Q8_0 activations (ggml's
// block_q8_0), one or several tokens per call, on AVX-512, AVX2 or plain C++.
//
// THE FORMAT (third_party/ggml/ggml-common-tq.h): a 128-weight block is {fp16 d; trellis steps of K bits}.
// TQ2_T / TQK6 / TQK7 (K = 8 / 6 / 7): 32 steps; step t has a 15-bit state whose hash picks two points (pairs) of a
// 2048-point fp16 codebook, which are weights 4t..4t+3, times d.  TQP5 / TQP6 / TQP7 (the pair trellis, K = 5 / 6
// / 7): 64 steps; step t's hash picks one point (the first of the two), weights 2t and 2t+1, times d.
//
// THE DECODE: a 32768 x 4 int16 table maps a state straight to its four weights at scale 2^13 (256 KiB, the
// approach of the fork's ggml-cpu vec_dot): one 8-byte lookup per four weights, no hash.  The TQP types read one
// point per state, so they hash the state (one multiply) and read the point from a 2048 x 2 int16 table (8 KiB,
// L1-resident) instead, as the fork's ggml-cpu TQP kernels do.  The int16 rounding is exact for codebook values
// >= 2^-3 in magnitude and within 2^-14 below that; both tables are rounded the same way as the fork's ggml-cpu
// tables (so the TQP point of state s is the first pair of the state table's row s).
//
// THE ARITHMETIC, identical on every ISA so every path gives the same bits (and so does any token-group size):
//   S[i][k] = sum_j w16[i][32k + j] * q8[i][k][j]        exact in int32 (|w16| < 2^15, |q8| <= 127)
//   acc[k]  = fma((float) S[i][k], d_w[i] * d_a[i][k], acc[k])      for blocks i in order, k = 0..3
//   dot     = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * 2^-13
// i.e. four float accumulators, one per Q8_0 sub-block position, each a fixed sequential chain.  It differs from
// ggml-cpu's vec_dot (same integers, other float summation order) by float rounding only.
//
// THE KERNELS: `tq_isa()` picks AVX-512 (F/BW/VL/VBMI/VNNI, cpu_avx512_ok) or AVX2 (cpu_avx2_ok) at run time; the
// scalar kernel is the reference (on a CPU without AVX2 the engine keeps ggml-cpu's vec_dot, see tq_kernels_enabled).
// STRATA_TQ_ISA=scalar|avx2|avx512 caps the choice (tests, A/B); STRATA_NO_TQ_KERNELS=1 makes native_expert.cpp use
// ggml-cpu's vec_dot instead (the previous path).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

enum class TqIsa : int { scalar = 0, avx2 = 1, avx512 = 2 };

/// EXPERIMENTAL lower-rate versions of the TQK rule (tools/trellis/tqenc.hpp, version 1; Strata-internal ids, not ggml
/// types, not in any GGUF): TQK5X = 22 bytes / 128 weights (1.375 bpw), TQK4X = 18 bytes (1.125 bpw).  Same hash,
/// codebook, 15-bit mask, LSB-first circular stream and fp16 scale as TQK6/7 with K = 5 / 4.
inline constexpr int kTqTypeK5X = 224;   ///< version 2: K = 5 with the computed codebook (tq_computed.hpp)
inline constexpr int kTqTypeK4X = 225;   ///< version 2: K = 4, computed codebook
inline constexpr int kTqTypeK5T = 226;   ///< version 1: K = 5 with the fork's point table (the first pilots)
inline constexpr int kTqTypeK4T = 227;   ///< version 1: K = 4, point table
/// The agentionai fork's ggml types TQK5 / TQK4 (in its GGUFs): the same bytes and table codebook as TQK5T / TQK4T.
inline constexpr int kTqTypeK5 = 150;
inline constexpr int kTqTypeK4 = 151;
/// TQ2_T (144), TQK6 (145), TQK7 (146), TQP5 / TQP6 / TQP7 (147-149), TQK5 / TQK4 (150 / 151) and the experimental
/// TQK5X / TQK4X / TQK5T / TQK4T.
bool tq_supported(int ggml_type) noexcept;
/// Bits per step: 8, 6, 7, 5, 4; TQP5 / TQP6 / TQP7: 5, 6, 7 (0 for any other type).
int tq_kbits(int ggml_type) noexcept;
/// Whether the type is a pair trellis (TQP: 64 steps of 2 weights per block) rather than 32 steps of 4.
inline constexpr bool tq_is_pair(int ggml_type) noexcept { return ggml_type >= 147 && ggml_type <= 149; }
/// Values per block of the trellis types, and bytes per block (2 + 4 * bits: 34, 26, 30, 22, 18; TQP: 2 + 8 * bits:
/// 42, 50, 58).
constexpr int kTqBlock = 128;
inline size_t tq_block_bytes(int ggml_type) noexcept { return (size_t) (2 + (tq_is_pair(ggml_type) ? 8 : 4) * tq_kbits(ggml_type)); }
/// The best ISA this CPU runs (capped by STRATA_TQ_ISA).
TqIsa tq_isa();
/// Whether this CPU can run `isa` at all (scalar: always).
bool tq_isa_ok(TqIsa isa);
/// Whether STRATA_NO_TQ_KERNELS=1 is absent.  The engine's native expert rows take these kernels for the trellis types
/// on every CPU (AVX-512 or AVX2 where the CPU has them, else the scalar kernel: a ggml from upstream has no trellis
/// type to fall back to, and the AVX code is only reached behind tq_isa_ok).  STRATA_NO_TQ_KERNELS=1 asks for
/// ggml-cpu's vec_dot instead, honoured only by a ggml that has the types (the agentionai fork; tests, A/B).
bool tq_kernels_enabled();
const char* tq_isa_name(TqIsa isa);

/// The 32768 x 4 int16 state table of the fork's types (filled on first use; thread-safe).
const int16_t* tq_state_table();
/// The same for the computed codebook of TQK5X / TQK4X (tq_computed.hpp).
const int16_t* tq_state_table_computed();
/// The table a type decodes with.
const int16_t* tq_state_table_for(int ggml_type);
/// The 2048 x 2 int16 codebook points at the same scale and rounding (filled on first use; thread-safe).
const int16_t* tq_pair_table();
/// One row of n weights of a trellis type decoded to float through the kernels' own state walk and the fp16
/// codebook: d * codebook value, the expression of ggml's to_float (so equal to it bit for bit).  For tests.
void tq_dequant_row(int ggml_type, const uint8_t* w, int n, float* out);

/// Rows [r0, r1) of a trellis matrix (`row_bytes` per row, `n` weights per row) against `nt` (<= 8) Q8_0
/// activations of n values: out[t][r] = w_r . a[t].
void tq_rows(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1);
/// Gate and up rows [r0, r1) (gate at blob, up at blob + up_off): ff[t][r] = silu(g) * u, the SwiGLU expression
/// of native_gu_rows.
void tq_gu_rows(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                const void* const* act, int nt, float* const* ff, int r0, int r1);

/// Largest token group per call and widest row the kernels take.
inline constexpr int kTqMaxTokens = 8;
inline constexpr int kTqMaxN = 5120;   // (DeepSeek-V4.1 hidden width; was 4096)

namespace tq_detail {
/// One token's Q8_0 activation unpacked for the kernels: the int8 values widened to int16 and the scales as floats.
struct TqAct {
    alignas(64) int16_t q[kTqMaxN];
    alignas(64) float d[kTqMaxN / 32];
};
void unpack_act(const void* q8_0, int n, TqAct& a);
// the per-ISA row kernels (dot products only; `out[t][r]` for rows [r0, r1)); the TQP ones (tq_rows_pair_*) take
// K = 5 / 6 / 7 and read tq_pair_table()
void tq_rows_scalar(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
void tq_rows_avx2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1);
void tq_rows_avx512(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
void tq_rows_pair_scalar(int k, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                         float* const* out, int r0, int r1);
void tq_rows_pair_avx2(int k, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                       float* const* out, int r0, int r1);
void tq_rows_pair_avx512(int k, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
}  // namespace tq_detail

}  // namespace strata::kernels::cpu

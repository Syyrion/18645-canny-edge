#include <opencv2/opencv.hpp>
#include <iostream>
#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>

static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

typedef std::size_t size_t;

const int TG22 = 13573; // tan(22.5 deg) * 2^15

#define SIMD_WIDTH 32
#define CACHE_LINE_WIDTH 64

// #define SIMD_ENABLED

// n must be a power of 2
inline size_t align_size(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

// Target CPU: L1d cache 32Kb 8-way associativity

/**
 * @param boundary_top top row boundary (inclusive)
 * @param boundary_bottom bottom row boundary (non-inclusive)
 * @param src_cols actual source image width (always less than pmap_step)
 * @param src_rows actual source image height
 * @param pmap_step source row stride (must be 32b aligned). Use when indexing dx, dy, and pmap.
 * @param dx Input x gradient. Base pointer must be 32b aligned. Rows must be 32b aligned with padding.
 * @param dy Input y gradient. Base pointer must be 32b aligned. Rows must be 32b aligned with padding.
 * @param pmap Output pmap. Pointer must be 32b aligned. Rows must be 32b aligned with padding.
 * @param low_thresh
 * @param high_thresh
 */
void nonmax(const int boundary_top,
            const int boundary_bottom,
            const int src_cols,
            const int src_rows,
            const int pmap_step,
            const short *const dx,
            const short *const dy,
            uchar *const pmap,
            const short low_thresh,
            const short high_thresh)
{
    int j;

    const int row_start = std::max(0, boundary_top - 1);         // starts one row above the top boundary (unless at src limits)
    const int row_end = std::min(src_rows, boundary_bottom + 1); // ends one row below the bottom boundary (unless at src limits)

    // ? opencv's implementation seems to allocate more space than is actually necessary.

    // size of a single magnitude buffer in bytes
    const size_t mag_size = align_size(SIMD_WIDTH + (src_cols * sizeof(short)) + sizeof(short), SIMD_WIDTH);
    const size_t mag_step = mag_size / sizeof(short);

    // 64 byte aligned
    short *buffer = (short *)_mm_malloc(3 * mag_size, CACHE_LINE_WIDTH);
    short *mag_prev, *mag_act, *mag_next;

    // ring buffer pointers
    mag_prev = buffer + SIMD_WIDTH / sizeof(short); // pointer starts 32 bytes in since we want to index with -1
    mag_act = mag_prev + mag_step;
    mag_next = mag_act + mag_step;

    // initial magnitude values
    if (row_start == boundary_top)
        // zero the magnitudes if we're at the top boundary
        memset(mag_next - 1, 0, (src_cols + 2) * sizeof(short));
    else
        mag_next[-1] = mag_next[src_cols] = 0;
    mag_act[-1] = mag_act[src_cols] = 0;
    mag_prev[-1] = mag_prev[src_cols] = 0;

    // constants in registers
    const __m256i v_low = _mm256_set1_epi16(low_thresh); // ymm0
    const __m256i v_high = _mm256_set1_epi16(high_thresh);
    const __m256i v_zero = _mm256_setzero_si256();
    const __m256i v_ones = _mm256_set1_epi16(-1);
    const __m256i v_tg22 = _mm256_set1_epi32(TG22); // madd operand: (TG22, 0) per 32-bit lane

    for (int i = row_start; i <= boundary_bottom; ++i)
    {
        // scroll ring buffer
        std::swap(mag_next, mag_act);
        std::swap(mag_next, mag_prev);

        if (i < row_end)
        {
            // calculate next row magnitudes

            const short *row_dx = dx + i * pmap_step;
            const short *row_dy = dy + i * pmap_step;

            j = 0;

            // simd part ymm8-ymm15
            __m256i v_dx0, v_dy0;
            __m256i v_dx1, v_dy1;
            __m256i v_dx2, v_dy2;
            __m256i v_dx3, v_dy3;

#ifdef SIMD_ENABLED

            // simd by 64 shorts
            for (; j <= src_cols - 64; j += 64)
            {
                v_dx0 = _mm256_load_si256((const __m256i *)(row_dx + j + 00));
                v_dy0 = _mm256_load_si256((const __m256i *)(row_dy + j + 00));
                v_dx1 = _mm256_load_si256((const __m256i *)(row_dx + j + 16));
                v_dy1 = _mm256_load_si256((const __m256i *)(row_dy + j + 16));
                v_dx2 = _mm256_load_si256((const __m256i *)(row_dx + j + 32));
                v_dy2 = _mm256_load_si256((const __m256i *)(row_dy + j + 32));
                v_dx3 = _mm256_load_si256((const __m256i *)(row_dx + j + 48));
                v_dy3 = _mm256_load_si256((const __m256i *)(row_dy + j + 48));
                v_dx0 = _mm256_abs_epi16(v_dx0);
                v_dy0 = _mm256_abs_epi16(v_dy0);
                v_dx1 = _mm256_abs_epi16(v_dx1);
                v_dy1 = _mm256_abs_epi16(v_dy1);
                v_dx2 = _mm256_abs_epi16(v_dx2);
                v_dy2 = _mm256_abs_epi16(v_dy2);
                v_dx3 = _mm256_abs_epi16(v_dx3);
                v_dy3 = _mm256_abs_epi16(v_dy3);
                _mm256_store_si256((__m256i *)(mag_next + j + 00), _mm256_add_epi16(v_dx0, v_dy0));
                _mm256_store_si256((__m256i *)(mag_next + j + 16), _mm256_add_epi16(v_dx1, v_dy1));
                _mm256_store_si256((__m256i *)(mag_next + j + 32), _mm256_add_epi16(v_dx2, v_dy2));
                _mm256_store_si256((__m256i *)(mag_next + j + 48), _mm256_add_epi16(v_dx3, v_dy3));
            }

            // simd by 32 shorts
            for (; j <= src_cols - 32; j += 32)
            {
                v_dx0 = _mm256_load_si256((const __m256i *)(row_dx + j + 00));
                v_dy0 = _mm256_load_si256((const __m256i *)(row_dy + j + 00));
                v_dx1 = _mm256_load_si256((const __m256i *)(row_dx + j + 16));
                v_dy1 = _mm256_load_si256((const __m256i *)(row_dy + j + 16));
                v_dx0 = _mm256_abs_epi16(v_dx0);
                v_dy0 = _mm256_abs_epi16(v_dy0);
                v_dx1 = _mm256_abs_epi16(v_dx1);
                v_dy1 = _mm256_abs_epi16(v_dy1);
                _mm256_store_si256((__m256i *)(mag_next + j + 00), _mm256_add_epi16(v_dx0, v_dy0));
                _mm256_store_si256((__m256i *)(mag_next + j + 16), _mm256_add_epi16(v_dx1, v_dy1));
            }

            // simd by 16 shorts
            for (; j <= src_cols - 16; j += 16)
            {
                v_dx0 = _mm256_load_si256((const __m256i *)(row_dx + j));
                v_dy0 = _mm256_load_si256((const __m256i *)(row_dy + j));
                v_dx0 = _mm256_abs_epi16(v_dx0);
                v_dy0 = _mm256_abs_epi16(v_dy0);
                _mm256_store_si256((__m256i *)(mag_next + j), _mm256_add_epi16(v_dx0, v_dy0));
            }

#endif
            // scalar for remaining columns
            for (; j < src_cols; ++j)
                mag_next[j] = std::abs((int)row_dx[j]) + std::abs((int)row_dy[j]);

            // at the very beginning we do not have a complete ring
            // buffer of 3 magnitude rows for non-maxima suppression
            if (i <= boundary_top)
                continue;
        }
        else
        {
            // zero the magnitudes if we're at the bottom boundary
            // we don't need to re-zero the value at -1 or src_cols since they never change
            memset(mag_next, 0, src_cols * sizeof(short));
        }

        uchar *pmap_row = pmap + i * pmap_step;

        const short *curr_dx = dx + (i - 1) * pmap_step;
        const short *curr_dy = dy + (i - 1) * pmap_step;

        j = 0;

#ifdef SIMD_ENABLED

        for (; j <= src_cols - 16; j += 16)
        {
            const __m256i v_m = _mm256_loadu_si256((const __m256i *)(mag_act + j));
            const __m256i v_cand = _mm256_cmpgt_epi16(v_m, v_low);

            // sparse fast path: most 16-pixel blocks have nothing above the low threshold
            if (_mm256_testz_si256(v_cand, v_cand))
                continue;

            // ---- direction classification (16 lanes) ----
            const __m256i v_dx = _mm256_load_si256((const __m256i *)(curr_dx + j));
            const __m256i v_dy = _mm256_load_si256((const __m256i *)(curr_dy + j));
            const __m256i v_ax = _mm256_abs_epi16(v_dx);
            const __m256i v_ay = _mm256_abs_epi16(v_dy);

            // Widen |dx|, |dy| to 32 bit (the fixed-point products need it).
            // unpacklo/hi interleave within each 128-bit lane; packs_epi32 below undoes that
            // exactly, so lane order is preserved without any cross-lane permute.
            const __m256i ax_lo = _mm256_unpacklo_epi16(v_ax, v_zero);
            const __m256i ax_hi = _mm256_unpackhi_epi16(v_ax, v_zero);
            const __m256i ay_lo = _mm256_unpacklo_epi16(v_ay, v_zero);
            const __m256i ay_hi = _mm256_unpackhi_epi16(v_ay, v_zero);

            // x * TG22 (madd: upper 16 bits of each 32-bit lane are zero, so it is a plain multiply)
            const __m256i tg22x_lo = _mm256_madd_epi16(ax_lo, v_tg22);
            const __m256i tg22x_hi = _mm256_madd_epi16(ax_hi, v_tg22);
            // y << 15
            const __m256i y_lo = _mm256_slli_epi32(ay_lo, 15);
            const __m256i y_hi = _mm256_slli_epi32(ay_hi, 15);
            // tg67x = tg22x + (x << 16)
            const __m256i tg67x_lo = _mm256_add_epi32(tg22x_lo, _mm256_slli_epi32(ax_lo, 16));
            const __m256i tg67x_hi = _mm256_add_epi32(tg22x_hi, _mm256_slli_epi32(ax_hi, 16));

            // horizontal: y < tg22x      vertical: y > tg67x      (otherwise diagonal)
            const __m256i is_h = _mm256_packs_epi32(_mm256_cmpgt_epi32(tg22x_lo, y_lo),
                                                    _mm256_cmpgt_epi32(tg22x_hi, y_hi));
            const __m256i is_v = _mm256_packs_epi32(_mm256_cmpgt_epi32(y_lo, tg67x_lo),
                                                    _mm256_cmpgt_epi32(y_hi, tg67x_hi));
            // diagonal sign: all-ones where (dx ^ dy) < 0, i.e. s == -1
            const __m256i neg = _mm256_srai_epi16(_mm256_xor_si256(v_dx, v_dy), 15);

            // ---- neighbour loads (unaligned, +-1 element) ----
            const __m256i act_l = _mm256_loadu_si256((const __m256i *)(mag_act + j - 1));
            const __m256i act_r = _mm256_loadu_si256((const __m256i *)(mag_act + j + 1));
            const __m256i prev_c = _mm256_loadu_si256((const __m256i *)(mag_prev + j));
            const __m256i prev_l = _mm256_loadu_si256((const __m256i *)(mag_prev + j - 1));
            const __m256i prev_r = _mm256_loadu_si256((const __m256i *)(mag_prev + j + 1));
            const __m256i next_c = _mm256_loadu_si256((const __m256i *)(mag_next + j));
            const __m256i next_l = _mm256_loadu_si256((const __m256i *)(mag_next + j - 1));
            const __m256i next_r = _mm256_loadu_si256((const __m256i *)(mag_next + j + 1));

            // horizontal: m > act[k-1] && m >= act[k+1]   (m >= r  <=>  !(r > m))
            const __m256i c_h = _mm256_andnot_si256(_mm256_cmpgt_epi16(act_r, v_m),
                                                    _mm256_cmpgt_epi16(v_m, act_l));
            // vertical: m > prev[k] && m >= next[k]
            const __m256i c_v = _mm256_andnot_si256(_mm256_cmpgt_epi16(next_c, v_m),
                                                    _mm256_cmpgt_epi16(v_m, prev_c));
            // diagonal: m > prev[k-s] && m > next[k+s]
            //   s = +1: prev_l, next_r      s = -1: prev_r, next_l
            const __m256i p_d = _mm256_blendv_epi8(prev_l, prev_r, neg);
            const __m256i n_d = _mm256_blendv_epi8(next_r, next_l, neg);
            const __m256i c_d = _mm256_and_si256(_mm256_cmpgt_epi16(v_m, p_d),
                                                 _mm256_cmpgt_epi16(v_m, n_d));

            // select the condition matching each lane's direction
            const __m256i is_d = _mm256_xor_si256(_mm256_or_si256(is_h, is_v), v_ones);
            __m256i is_max = _mm256_or_si256(_mm256_and_si256(is_h, c_h),
                                             _mm256_or_si256(_mm256_and_si256(is_v, c_v),
                                                             _mm256_and_si256(is_d, c_d)));
            is_max = _mm256_and_si256(is_max, v_cand); // m > low

            const __m256i is_strong = _mm256_and_si256(is_max, _mm256_cmpgt_epi16(v_m, v_high));

            // ---- write pmap: 0 / 1 (weak) / 2 (strong) ----
            // 0xFFFF >> 15 == 1, so r = is_max?1:0 + is_strong?1:0
            const __m256i r16 = _mm256_add_epi16(_mm256_srli_epi16(is_max, 15),
                                                 _mm256_srli_epi16(is_strong, 15));
            // 16 x u16 -> 16 x u8 (packus is per-128-bit-lane, so gather qwords 0 and 2)
            const __m256i r8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(r16, r16), 0x08);
            _mm_storeu_si128((__m128i *)(pmap_row + j), _mm256_castsi256_si128(r8));

            // ---- bit-scan only the strong pixels (sparse) to seed hysteresis ----
            unsigned strong_bits = _pext_u32((unsigned)_mm256_movemask_epi8(is_strong), 0x55555555u);
            while (strong_bits != 0)
            {
                int l = __builtin_ctz(strong_bits);
                strong_bits &= strong_bits - 1;
                // CANNY_PUSH_STACK(pmap_row + j + l, stack);
                (void)l;
            }
        }

#endif

        // scalar loop for remaining elements
        for (; j < src_cols; j++)
        {
            int m = mag_act[j];

            if (m > low_thresh)
            {
                short xs = curr_dx[j];
                short ys = curr_dy[j];
                int x = (int)std::abs(xs);
                int y = (int)std::abs(ys) << 15;
                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > mag_act[j - 1] && m >= mag_act[j + 1])
                    {
                        if (m > high_thresh)
                        {
                            pmap_row[j] = 2;
                            // CANNY_PUSH_STACK(_pmap_row + j, stack);
                        }
                        else
                        {
                            pmap_row[j] = 1;
                        }
                        continue;
                    }
                }
                else
                {
                    int tg67x = tg22x + (x << 16);
                    if (y > tg67x)
                    {
                        if (m > mag_prev[j] && m >= mag_next[j])
                        {
                            if (m > high_thresh)
                            {
                                pmap_row[j] = 2;
                                // CANNY_PUSH_STACK(_pmap_row + j, stack);
                            }
                            else
                            {
                                pmap_row[j] = 1;
                            }
                            continue;
                        }
                    }
                    else
                    {
                        int s = ((xs ^ ys) < 0) ? -1 : 1;
                        if (m > mag_prev[j - s] && m > mag_next[j + s])
                        {
                            if (m > high_thresh)
                            {
                                pmap_row[j] = 2;
                                // CANNY_PUSH_STACK(_pmap_row + j, stack);
                            }
                            else
                            {
                                pmap_row[j] = 1;
                            }
                            continue;
                        }
                    }
                }
            }
        }
    }

    _mm_free(buffer);
}

bool is_aligned_64(const void *ptr)
{
    return ptr != nullptr &&
           (reinterpret_cast<std::uintptr_t>(ptr) % 64) == 0;
}

int main(int argc, char *argv[])
{
    unsigned long long t0, t1;

    if (argc < 3)
    {
        std::cerr << "Usage: canny <low_thresh> <high_thresh> <image-path> <output-path>\n";
        return 1;
    }

    int low_thresh, high_thresh;

    try
    {
        low_thresh = std::stoi(argv[1]);
    }
    catch (const std::exception &)
    {
        std::cerr << "invalid low threshold\n";
        return 1;
    }
    try
    {
        high_thresh = std::stoi(argv[2]);
    }
    catch (const std::exception &)
    {
        std::cerr << "invalid high threshold\n";
        return 1;
    }

    if (high_thresh < low_thresh)
        std::swap(low_thresh, high_thresh);

    cv::Mat image = cv::imread(argv[3], cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Could not load image: " << argv[1] << '\n';
        return 1;
    }

    cv::Mat gray, blurred, dx, dy, grad_mag, grad_phase, edges;
    int src_rows = image.size[0];
    int src_cols = image.size[1];

    // setup
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);      // gray is CV_8U
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1); // blurred is CV_8U

    t0 = rdtsc();
    cv::Canny(gray, edges, low_thresh, high_thresh);
    t1 = rdtsc();

    std::cerr << "baseline pixels/cycle: " << (src_cols * src_rows) / ((double)(t1 - t0)) << "\n";

    cv::imwrite(argv[4], edges);

    cv::Sobel(blurred, dx, CV_16S, 1, 0, 3);
    cv::Sobel(blurred, dy, CV_16S, 0, 1, 3);

    // Pad src_cols by 33 to 64 uchars. This makes room for the border pmap values.
    // This value is also used as the stride for dx, dy and pmap
    // [32 uchars PAD] [src_cols uchars] [1 to 32 uchars PAD (32 if src_cols is multiple of 32, less if not)]
    const size_t pmap_step = align_size(SIMD_WIDTH + (src_cols * sizeof(uchar)) + sizeof(uchar), SIMD_WIDTH);

    cv::copyMakeBorder(dx, dx, 0, 0, 0, pmap_step - src_cols, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(dy, dy, 0, 0, 0, pmap_step - src_cols, cv::BORDER_CONSTANT, 0);
    cv::Mat pmap = cv::Mat::zeros(cv::Size(pmap_step, src_rows + 2), CV_8U); // +2 for borders

    // make sure opencv hasn't added any padding without our knowledge
    if (!dx.isContinuous())
    {
        std::cerr << "error: dx contains padding" << "\n";
        return 1;
    }
    if (!dy.isContinuous())
    {
        std::cerr << "error: dx contains padding" << "\n";
        return 1;
    }
    if (!pmap.isContinuous())
    {
        std::cerr << "error: pmap contains padding" << "\n";
        return 1;
    }

    // get pointers to matrix data
    short *dx_p = dx.ptr<short>(0);
    short *dy_p = dy.ptr<short>(0);
    uchar *pmap_p = pmap.ptr<uchar>(0);

    // check aligment
    if (!is_aligned_64(dx_p))
    {
        std::cerr << "error: dx is not aligned to 64b" << "\n";
        return 1;
    }
    if (!is_aligned_64(dy_p))
    {
        std::cerr << "error: dy is not aligned to 64b" << "\n";
        return 1;
    }
    if (!is_aligned_64(pmap_p))
    {
        std::cerr << "error: pmap is not aligned to 64b" << "\n";
        return 1;
    }

    std::cerr << "src_cols: " << src_cols << "\n";
    std::cerr << "src_rows: " << src_rows << "\n";
    std::cerr << "pmap_step: " << pmap_step << "\n";

    t0 = rdtsc();
    nonmax(0, src_rows, src_cols, src_rows, pmap_step, dx_p, dy_p, pmap_p + SIMD_WIDTH, low_thresh, high_thresh);
    t1 = rdtsc();

    std::cerr << "pixels/cycle: " << (src_cols * src_rows) / ((double)(t1 - t0)) << "\n";

    cv::Mat result;
    pmap.convertTo(result, CV_8U, 127.0); // amplify pmap so we can see the edges
    cv::imwrite(argv[4], result);

    return 0;
}

---
title: "Optimizing mod11 checksum verification with SIMD"
date: 2023-12-03
---

I've recently read [a great article about optimizing Luhn algorithm with SWAR and SIMD](https://nullprogram.com/blog/2022/04/30/),
and I am a big fan of unecessary optimization, so I tried to optimize the mod 11
checksum algorithm.

This algorithm is most known for beign used in the [ISBN-10 checksum](https://en.wikipedia.org/wiki/ISBN#ISBN-10_check_digits),
but I am mostly interested in Brazil's (my home country) individual taxpayer
registration number, the CPF, it's equivalent to the social security number in
the US, it serves to identify every resident in Brazil.

The CPF is formed by 11 digits, 9 significant digits and 2 checksum digits at
the end, it's often written with punctuation like: "000.000.000-00", but in
this article I will ignore the punctuation. The checksum used in it is slightly
different from the one used in the ISBN-10 (but the code presented here can be easily
adapted) and it must be performed 2 times, one for each checksum digit, the
first checking the first nine digits agains the first checksum digit, and the
second checking the 2nd to 9th digit and the first checksum digit against the
second one. Each check goes like this:

 1. The 1st digit is multiplied by 1, the 2nd by 2... So it goes until de 9th is multiplied by 9.
 2. The result of the multiplications are added and divided by 11, the checksum will be the rest of the division.
 3. If the rest is 10 we consider it as 0, then we check if it's the same as the checksum digit.

Example: 24685571070 (randomly generated)

<pre style="font-family:monospace,monospace;border:none;">
2 4 6  8  5  5  7  1 0
x x x  x  x  x  x  x x
1 2 3  4  5  6  7  8 9
----------------------
2+8+18+32+25+30+49+8+0 = 127
</pre>

172 mod 11 = 7 (first checksum checks out)

<pre style="font-family:monospace,monospace;border:none;">
4 6  8  5  5  7  1 0 7
x x  x  x  x  x  x x x
1 2  3  4  5  6  7 8 9
-----------------------
4+12+24+20+25+42+7+0+63 = 197
</pre>

197 mod 11 = 10 (second checksum checks out because 10 becomes 0)

This algorithm implemented in C:

```
int
mod11(const char *s)
{
    int final = 0;
    for (int i = 0; i < 9; i++)
        final += (s[i-1] - '0') * i;

    final %= 11;

    //return (final == 10 ? 0 : final)  == s[9] - '0';
    return (final != 10) * final == s[9] - '0';
}

int
check_cpf(const char *s)
{
    return mod11(s) && mod11(s+1);
}
```

But what if, hypothetically speaking, this implementation is too slow? We can do
better rewritting the mod11 function with SSE2 SIMD instructions, which
parallelizes the algorithm, thus, making it more efficient:

```
#include <emmintrin.h>

int
mod11(const char *s)
{
    // load the string into a register
    __m128i r = _mm_loadu_si128((void *)s);

    // convert ascii to decimal
    r = _mm_xor_si128(r, _mm_set1_epi8(0x30));

    // multipy the first 9 numbers by 1, 2, 3...
    __m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
    __m128i even = _mm_mullo_epi16(r, m);
    __m128i odd = _mm_mullo_epi16(
        _mm_srli_epi16(r, 8),
        _mm_srli_epi16(m, 8)
    );

    r = _mm_or_si128(
        _mm_slli_epi16(odd, 8),
        _mm_srli_epi16(_mm_slli_epi16(even, 8), 8)
    );

    // sum the results into one int
    r = _mm_sad_epu8(r, _mm_setzero_si128());
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
    int final = _mm_cvtsi128_si32(r);

    final %= 11;

    //return (final == 10 ? 0 : final)  == s[9] - '0';
    return (final != 10) * final == s[9] - '0';
}
```

In the code above, the multiplication part is kinda tricky because there is no
simple way to multiply 8 bit integers with SIMD, so we have to use a trick that
cost us some precious instructions, But, if we dont constrain ourselves to SSE2
we can simplify the multiplication with the SSSE3 instruction `_mm_addubs_epi16`
wich multiplies unsigned 8 bit numbers and add them in pairs resulting in 8
signed 16 bit numbers, the result, in this case, will never surpass the 8 bits,
so we can perform a horizontal sum as if they where 8 bit numbers like before:

```
#include <tmmintrin.h>

int
mod11(const char *s)
{
    // load the string into a register
    __m128i r = _mm_loadu_si128((void *)s);

    // convert ascii to decimal
    r = _mm_xor_si128(r, _mm_set1_epi8(0x30));

    // multipy the first 9 numbers by 1, 2, 3...
    __m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
    r = _mm_maddubs_epi16(r, m);

    // sum the results into one int
    r = _mm_sad_epu8(r, _mm_setzero_si128());
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
    int final = _mm_cvtsi128_si32(r);

    final %= 11;

    //return (final == 10 ? 0 : final)  == s[9] - '0';
    return (final != 10) * final == s[9] - '0';
}
```

A little bit neater.

## Benchmark

Remembering that all implementations here assume you are passing valid digits
only.

All the code was compiled with gcc 13.2.1 on linux with the `-O3` flag and ran
on my notebook with an AMD Ryzen 5 5500U @ 4.0GHz.

 - Iterative: ~118 checksums per second
 - SSE2: ~218 checksums per second
 - SSSE3: ~237 checksums per second.

In the end the SSSE3 version got a 100% speed improvement over the iterative
version.

If you know how to further optimize the code shown, let me know!

## Full source code

all implementations in a `#ifdef` soup.

```
#ifdef __SSSE3__
#  include <tmmintrin.h>
#elif defined __SSE2__
#  include <emmintrin.h>
#endif

int
mod11(const char *s)
{

#ifdef __SSE2__

    __m128i r = _mm_loadu_si128((void *)s);
    r = _mm_xor_si128(r, _mm_set1_epi8(0x30));

    __m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);

#ifdef __SSSE3__
    r = _mm_maddubs_epi16(r, m);
#else

    __m128i even = _mm_mullo_epi16(r, m);
    __m128i odd = _mm_mullo_epi16(
    _mm_srli_epi16(r, 8),
        _mm_srli_epi16(m, 8)
    );

    r = _mm_or_si128(
        _mm_slli_epi16(odd, 8),
        _mm_srli_epi16(_mm_slli_epi16(even, 8), 8)
    );
#endif

    r = _mm_sad_epu8(r, _mm_setzero_si128());
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
    int final = _mm_cvtsi128_si32(r);

#else
    int final = 0;
    for (int i = 1; i <= 9; i++)
        final += (s[i-1] - '0') * i;

#endif
    final %= 11;

    return (final != 10) * final == s[9] - '0';
}

int
check_cpf(const char *s)
{
    return mod11(s) && mod11(s+1);
}

#ifdef BENCH
#include <stdio.h>
#include <time.h>
#include <stdint.h>

static double
now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec/1e9;
}

int
main(void)
{
    static const char cpfs[][11] = {
        "51386130222", "54535728097",
        "36709354105", "31797721232",
    };

    long n = 1L << 28;
    double start = now();
    unsigned r = 1;
    for (long i = 0; i < n; i++) {
        uint32_t n = i;
        r += check_cpf(cpfs[(n*0x1c5bf891U)>>30]);  // random draw
    }

    volatile unsigned sink = r;
    (void)sink;
    printf("%.3f M-ops/s\n", n / 1e6 / (now() - start));
}

#elif defined TEST
#include <stdio.h>
#include <assert.h>

int
main(void)
{
    assert(check_cpf("51386130222"));
    assert(check_cpf("36709354105"));
    assert(check_cpf("38104725168") == 0);
    assert(check_cpf("56709354105") == 0);
    printf("working!\n");
    return 0;
}

#endif
```


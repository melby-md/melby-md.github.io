---
title: "Optimizing mod11 checksum verification with SIMD"
date: 2023-12-03
---

I've recently read [a great article about optimizing Luhn algorithm with SWAR and SIMD](https://nullprogram.com/blog/2022/04/30/),
and I am a big fan of unecessary optimization, so I tried to optimize the mod 11
checksum algorithm. This algorithm is most known for beign used in the [ISBN-10 checksum](https://en.wikipedia.org/wiki/ISBN#ISBN-10_check_digits),
but I am mostly interested in Brazil's (my home country) individual taxpayer
registration number, the [CPF](https://en.wikipedia.org/wiki/CPF_number), it's
equivalent to the social security number in the US, it serves to identify every
resident in Brazil.

The CPF is formed by 11 digits, 9 significant digits and 2 checksum digits at
the end, it's often written with punctuation like: "000.000.000-00", but in
this article I will ignore the punctuation. The checksum used in it is slightly
different from the one used in the ISBN-10 (but the code presented here can be
easily adapted) and it must be performed 2 times, one for each checksum digit,
the first checking the first nine digits agains the first checksum digit, and the
second checking the 2nd to 9th digit and the first checksum digit against the
second one. Each check goes like this:

 1. The 1st digit is multiplied by 1, the 2nd by 2... So it goes until de 9th is multiplied by 9.
 2. The result of the multiplications are added and divided by 11, the checksum will be the rest of the division.
 3. If the rest is 10 consider it as 0, then check if it's the same as the checksum digit.

Example: 24685571070 (randomly generated)

<pre class="noborder">
2 4 6  8  5  5  7  1 0
x x x  x  x  x  x  x x
1 2 3  4  5  6  7  8 9
----------------------
2+8+18+32+25+30+49+8+0 = 127
</pre>

172 mod 11 = 7 (first checksum checks out)

<pre class="noborder">
4 6  8  5  5  7  1 0 7
x x  x  x  x  x  x x x
1 2  3  4  5  6  7 8 9
-----------------------
4+12+24+20+25+42+7+0+63 = 197
</pre>

197 mod 11 = 10 (second checksum checks out because 10 becomes 0)

This algorithm implemented in C (assuming you're feeding the function with
strings with the right length and only containing digits):

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

But what if, hypothetically speaking, this implementation is too slow for your
high performance software&trade;? We can do better by rewritting the mod11
function with SSE2 SIMD instructions:

```
#include <emmintrin.h>

int
mod11(const char *s)
{
    // load the string into a vector
    __m128i r = _mm_loadu_si128((void *)s);

    // convert ascii to decimal
    r = _mm_xor_si128(r, _mm_set1_epi8(0x30));

    // multipy the first 9 numbers by 1, 2, 3...
    __m128i m = _mm_set_epi32(0, 0x00000900, 0x07080506, 0x03040102);
    r = _mm_mullo_epi16(r, m);
    r = _mm_srli_epi16(r, 8);

    // sum the results into one int
    r = _mm_sad_epu8(r, _mm_setzero_si128());
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
    int final = _mm_cvtsi128_si32(r);

    final %= 11;

    //return (final == 10 ? 0 : final)  == s[9] - '0';
    return (final != 10) * final == s[9] - '0';
}
```

Lets go line by line. First load the array into a integer vector:

```
__m128i r = _mm_loadu_si128((void *)s);
```

This will load memory out of bounds, but it will deal with it later. Then,
convert the ASCII digits into its decimal values by XORing every byte with 0x30:

```
r = _mm_xor_si128(r, _mm_set1_epi8(0x30));
```

Then comes the tricky part, there is no strightfoward way to multiply 8 bit
integers with x86 SIMD, but, because I don't need the result of each
multiplication in a separate byte, I will use a trick:

```
__m128i m = _mm_set_epi32(0, 0x00000900, 0x07080506, 0x03040102);
```

First, the numbers that will multiply the CPF digits are loaded into `m`,
the zeored bytes will multiply the bytes loaded out of bounds, zeroing them,
then comes the multiplication:

```
r = _mm_mullo_epi16(r, m);
```
In the end this multiplication will result in each high byte of every 16 bit
number to contain the sum of every 2 consecutive multiplications, sounds weird?
I will illustrate:


<pre class="noborder">
     a  b
   x 2  1
   -------
     1a 1b
+ 2a 2b
-------------
  2a 1a+2b 1b
</pre>

That's why the numbers are switched in `m`. after the multiplication, comes a
shift right to eliminate the low byte:

```
r = _mm_srli_epi16(r, 8);
```

Ta-dah, a vector with the sum of every 2 consecutive results of the
multiplication of each byte. Now we need to sum it all into one integer. In this
case, the result of the addition will never surpass 8 bits, so the high byte
will be 0, so the function can perform a horizontal sum as 8 bit numbers, which
is faster than a 16 bit sum. So, first use `_mm_sad_epu8` which subtracts 8 bit
numbers, then add each consecutive 8 numbers into a 16 bit number, I used a
zeroed vector beacuse I am only interested in the addition in the end (weirdly
enough).

```
r = _mm_sad_epu8(r, _mm_setzero_si128());
```

To add the 2 numbers, shuffle the vector and add it to
itself:

```
r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
```

To convert to `int`:

```
int final = _mm_cvtsi128_si32(r);
```

The rest is the same as the iterative version.

If we don't constrain ourselves to SSE2 we can simplify the multiplication with
the SSSE3 instruction `_mm_addubs_epi16` which does almost the same as the
instructions used in the multiplication before, so just change the order of the
numbers in `m` and multiply it with `r`:

```
__m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
r = _mm_maddubs_epi16(r, m);
```

the sum part stays the same:

```
#include <tmmintrin.h>

int
mod11(const char *s)
{
    // load the string into a vector
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

A little bit neater I think.

## Benchmark

All the code was compiled with gcc 13.2.1 on linux with the `-O3` flag and ran
on my notebook with an AMD Ryzen 5 5500U @ 4.0GHz.

By using the `-march=native` the compiler can optimize the code even further,
but that depends on your processor model.

Results ranked by speed:

 1. SSE2 with `-march=native`: ~246 million checksums per second
 2. SSSE3 with `-march=native`: ~242 million checksums per second.
 3. SSSE3: ~239 million checksums per second.
 4. SSE2: ~237 million checksums per second
 5. Iterative: ~118 million checksums per second

Surprisingly enough, even though the SSSE3 version was faster than the SSE2
version, with `-march=native`, the latter version was faster.

In the end, the vectorized implementations had a 100% or more speed improvement.

If you know how to further optimize the code shown or use a different aproach
(SWAR, other architeture, etc) let me know!

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

#ifdef __SSSE3__
    __m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
    r = _mm_maddubs_epi16(r, m);
#else
    __m128i m = _mm_set_epi32(0, 0x00000900, 0x07080506, 0x03040102);
    r = _mm_mullo_epi16(r, m);
    r = _mm_srli_epi16(r, 8);
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
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
static double
now(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / f.QuadPart;
}

#else
#include <time.h>
static double
now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec/1e9;
}
#endif

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


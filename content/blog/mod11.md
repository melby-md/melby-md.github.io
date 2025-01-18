---
title: "Optimizing mod11 checksum verification with SIMD"
date: 2023-12-03
lastmod: 2025-01-17
---

I've recently read [a great article about optimizing the Luhn algorithm with SWAR and SIMD](https://nullprogram.com/blog/2022/04/30/),
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

 1. The 1st digit is multiplied by 1, the 2nd by 2... So it goes until the 9th is multiplied by 9.
 2. The result of the multiplications are added and divided by 11, the checksum will be the rest of the division.
 3. If the rest is 10 consider it as 0, then check if it's the same as the checksum digit.

Example: 24685571070 (randomly generated)

<pre>
2 4 6  8  5  5  7  1 0
x x x  x  x  x  x  x x
1 2 3  4  5  6  7  8 9
----------------------
2+8+18+32+25+30+49+8+0 = 127
</pre>

172 mod 11 = 7 (first checksum checks out)

<pre>
4 6  8  5  5  7  1 0 7
x x  x  x  x  x  x x x
1 2  3  4  5  6  7 8 9
-----------------------
4+12+24+20+25+42+7+0+63 = 197
</pre>

197 mod 11 = 10 (second checksum checks out because 10 becomes 0)

This algorithm implemented in C (assuming you're feeding the function with
ascii strings with 11 digits):

``` c
bool
mod11(char *s)
{
    int sum = 0;
    for (int i = 1; i <= 9; i++)
        sum += (s[i-1] & 0x0f) * i;

    sum %= 11;

    return (sum == 10 ? 0 : sum) == (s[9] & 0x0f);
}

bool
check_cpf(char *s)
{
	return mod11(s) & mod11(s+1);
}
```

But, what if, hypothetically speaking, this implementation is too slow for your
high performance software&trade;? We can do better by rewritting the mod11
function with SSE2 SIMD instructions:

``` c
#include <emmintrin.h>

bool
mod11(char *s)
{
	// load the string into a vector
	__m128i r = _mm_loadu_si128((const __m128i *)s);

	// convert ascii to decimal
	r = _mm_and_si128(r, _mm_set1_epi8(0x0f));

	// multipy the first 9 numbers by 1, 2, 3..., and sum adjacent pairs
	__m128i m = _mm_set_epi32(0, 0x00000900, 0x07080506, 0x03040102);
	r = _mm_mullo_epi16(r, m);
	r = _mm_srli_epi16(r, 8);

	// sum the results into one int
	r = _mm_sad_epu8(r, _mm_setzero_si128());
	r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
	int sum = _mm_cvtsi128_si32(r);

	sum %= 11;

	return (sum == 10 ? 0 : sum) == (s[9] & 0x0f);
}
```

Lets go line by line. First load the array into a integer vector:

``` c
__m128i r = _mm_loadu_si128((const __m128i *)s);
```

This will load memory out of bounds, but it will deal with it later. Then,
convert the ASCII digits into its decimal values by zeoring the upper 4 bits:

``` c
r = _mm_and_si128(r, _mm_set1_epi8(0x0f));
```

Then comes the tricky part, there is no straightfoward way to multiply 8 bit
integers with x86 SIMD, but I don't need the result of each multiplication in
a separate byte because the function will sum it all in the end, so I will use a
trick:

``` c
__m128i m = _mm_set_epi32(0, 0x00000900, 0x07080506, 0x03040102);
```

The numbers that will multiply the CPF digits are loaded into `m`,
the 0x00 bytes will multiply the bytes loaded out of bounds, zeroing them,
then comes the multiplication:

``` c
r = _mm_mullo_epi16(r, m);
```

In the end, this multiplication will result in each high byte of every 16 bit
number to contain the sum of every 2 consecutive multiplications, sounds weird?
I will illustrate:

<pre>
     a  b
   x 2  1
   -------
     1a 1b
+ 2a 2b
-------------
  2a <b>1a+2b</b> 1b
</pre>

That's why the numbers are switched in `m`. after the multiplication, comes a
right shift to remove the low byte:

``` c
r = _mm_srli_epi16(r, 8);
```

Ta-dah, a vector with the sum of every 2 consecutive results of the
multiplication of each byte. Now we need to sum it all into one integer. In this
case, the result of the addition will never surpass 8 bits, so the high byte
will be 0, the function will perform a horizontal sum as 8 bit numbers, which
is faster than a 16 bit sum. So, first use `_mm_sad_epu8`, which subtracts 8 bit
numbers, then add each consecutive 8 numbers into a 16 bit number, I used a
zeroed vector beacuse I am only interested in the addition in the end (weirdly
enough).

``` c
r = _mm_sad_epu8(r, _mm_setzero_si128());
```

To add the 2 numbers, shuffle the vector and add it to
itself:

``` c
r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
```

To convert to `int`:

``` c
int final = _mm_cvtsi128_si32(r);
```

The rest is the same as the iterative version.

If we don't constrain ourselves to SSE2, we can simplify the multiplication with
the SSSE3 instruction `_mm_addubs_epi16` which does almost the same as the
instructions used in the multiplication before, so just change the order of the
numbers in `m` and multiply it with `r`:

``` c
__m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
r = _mm_maddubs_epi16(r, m);
```

the sum part stays the same:

``` c
#include <tmmintrin.h>

bool
mod11_ssse3(char *s)
{
	// load the string into a vector
	__m128i r = _mm_loadu_si128((const __m128i *)s);

	// convert ascii to decimal
	r = _mm_and_si128(r, _mm_set1_epi8(0x0f));

	// multipy the first 9 numbers by 1, 2, 3..., and sum adjacent pairs
	__m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
	r = _mm_maddubs_epi16(r, m);

	// sum the results into one int
	r = _mm_sad_epu8(r, _mm_setzero_si128());
	r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
	int sum = _mm_cvtsi128_si32(r);

	sum %= 11;

	return (sum == 10 ? 0 : sum) == (s[9] & 0x0f);
}
```

A little bit neater I think.

## Benchmark

All the code was compiled with gcc 13.2.1 with `-O3 -mssse3` flags and ran on my
notebook with an AMD Ryzen 5 5500U @ 4.0GHz.

Results ranked by speed:

 1. SSSE3: ~81 million checksums per second.
 2. SSE2: ~80 million checksums per second.
 3. Iterative: ~72 million checksums per second.

Well... I was expecting more, but it is a speed up none the less.

If you know how to further optimize the code shown let me know!

## Full source code

all implementations in a `#ifdef` soup.

``` c
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#elif defined __SSE2__
#include <emmintrin.h>
#endif

bool
mod11_iterative(char *s)
{
    int sum = 0;
    for (int i = 1; i <= 9; i++)
        sum += (s[i-1] & 0x0f) * i;

    sum %= 11;

    return (sum == 10 ? 0 : sum) == (s[9] & 0x0f);
}

bool
check_cpf_iterative(char *s)
{
	return mod11_iterative(s) & mod11_iterative(s+1);
}

#ifdef __SSE2__
bool
mod11_sse2(char *s)
{
	// load the string into a vector
	__m128i r = _mm_loadu_si128((const __m128i *)s);

	// convert ascii to decimal
	r = _mm_and_si128(r, _mm_set1_epi8(0x0f));

	// multipy the first 9 numbers by 1, 2, 3..., and sum adjacent pairs
	__m128i m = _mm_set_epi32(0, 0x00000900, 0x07080506, 0x03040102);
	r = _mm_mullo_epi16(r, m);
	r = _mm_srli_epi16(r, 8);

	// sum the results into one int
	r = _mm_sad_epu8(r, _mm_setzero_si128());
	r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
	int sum = _mm_cvtsi128_si32(r);

	sum %= 11;

	return (sum == 10 ? 0 : sum) == (s[9] & 0x0f);
}

bool
check_cpf_sse2(char *s)
{
	return mod11_sse2(s) & mod11_sse2(s+1);
}
#endif

#ifdef __SSSE3__
bool
mod11_ssse3(char *s)
{
	// load the string into a vector
	__m128i r = _mm_loadu_si128((const __m128i *)s);

	// convert ascii to decimal
	r = _mm_and_si128(r, _mm_set1_epi8(0x0f));

	// multipy the first 9 numbers by 1, 2, 3..., and sum adjacent pairs
	__m128i m = _mm_set_epi32(0, 0x00000009, 0x08070605, 0x04030201);
	r = _mm_maddubs_epi16(r, m);

	// sum the results into one int
	r = _mm_sad_epu8(r, _mm_setzero_si128());
	r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 2));
	int sum = _mm_cvtsi128_si32(r);

	sum %= 11;

	return (sum == 10 ? 0 : sum) == (s[9] & 0x0f);
}

bool
check_cpf_ssse3(char *s)
{
	return mod11_ssse3(s) & mod11_ssse3(s+1);
}
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

double
now(void)
{
	LARGE_INTEGER f, t;
	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&t);
	return (double)t.QuadPart / f.QuadPart;
}
#else
double
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
	char cpfs[][12] = {
		"84490986025", "11111111111",
		"82269940040", "12312312312",
		"23799146059", "00000000000",
		"05321024014", "42424242424",
	};
	long n = 1L << 28;
	volatile unsigned sink = 0;
	double start;

	srand(time(NULL));

	start = now();
	for (long i = 0; i < n; i++)
		sink += check_cpf_iterative(cpfs[rand() & 7]);
	printf("Iterative: %.3f M-ops/s\n", n / 1e6 / (now() - start));

#ifdef __SSE2__
	start = now();
	for (long i = 0; i < n; i++)
		sink += check_cpf_sse2(cpfs[rand() & 7]);
	printf("SSE2: %.3f M-ops/s\n", n / 1e6 / (now() - start));
#endif

#ifdef __SSSE3__
	start = now();
	for (long i = 0; i < n; i++)
		sink += check_cpf_ssse3(cpfs[rand() & 7]);
	printf("SSSE3: %.3f M-ops/s\n", n / 1e6 / (now() - start));
#endif

	(void)sink;

	return 0;
}
```


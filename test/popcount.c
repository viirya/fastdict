#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned long long u64;
typedef unsigned int       u32;

inline u32 popcount64_1(u64 x) { return __builtin_popcountll(x); }

inline u32 popcount64_2(u64 x)
{
    u32 y;

    x = (x & 0x5555555555555555ULL) + ((x >>  1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >>  2) & 0x3333333333333333ULL);
    x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >>  4) & 0x0F0F0F0F0F0F0F0FULL);
    x = (x & 0x000F000F000F000FULL) + ((x >>  8) & 0x000F000F000F000FULL);
    x = (x & 0x0000001F0000001FULL) + ((x >> 16) & 0x0000001F0000001FULL);
    y = (x & 0x000000000000003F   ) + ((x >> 32) & 0x000000000000003F   );
    return y;
}

inline u32 popcount64_3(u64 x)
{
    x = (x & 0x5555555555555555ULL) + ((x >>  1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >>  2) & 0x3333333333333333ULL);
    x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >>  4) & 0x0F0F0F0F0F0F0F0FULL);
    return (x * 0x0101010101010101ULL) >> 56;
}

inline u32 popcount64_4(u64 x)
{
    x = (x & 0x5555555555555555ULL) + ((x >>  1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >>  2) & 0x3333333333333333ULL);
    x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >>  4) & 0x0F0F0F0F0F0F0F0FULL);
    return (((u32)(x >> 32)) * 0x01010101 >> 24) + 
           (((u32)(x      )) * 0x01010101 >> 24);
}

u64 data[1024];

u64 lrand64(void)
{
    u32 lo = lrand48();
    u32 hi = lrand48();

    return ((u64)hi << 32) | lo;
}

volatile u32 pt;

main()
{
    int i, j;
    u32 p, pp;
    time_t t1, t2;

    srand48(time(0));

    for (i = 0; i < 1024; i++)
        data[i] = lrand64();

    for (i = 0; i < 1024; i++)
    {
        p = popcount64_1(data[i]);

        if ((pp = popcount64_2(data[i])) != p)
        {
            printf("FAIL 2: %llx %d %d\n", data[i], p, pp);
            exit(1);
        }

        if ((pp = popcount64_3(data[i])) != p)
        {
            printf("FAIL 3: %llx %d %d\n", data[i], p, pp);
            exit(1);
        }

        if ((pp = popcount64_4(data[i])) != p)
        {
            printf("FAIL 4: %llx %d %d\n", data[i], p, pp);
            exit(1);
        }
    }

    t1 = clock();
    for (j = 0; j < 1000000; j++)
        for (i = 0; i < 1024; i++)
            pt = popcount64_1(data[i]);
    t2 = clock();

    printf("popcount64_1 = %d clocks\n", t2 - t1);
    printf("popcount64_1 = %f seconds \n", ((float)t2 - t1) / CLOCKS_PER_SEC);
 
    t1 = clock();
    for (j = 0; j < 1000000; j++)
        for (i = 0; i < 1024; i++)
            pt = popcount64_2(data[i]);
    t2 = clock();

    printf("popcount64_2 = %d clocks\n", t2 - t1);
    printf("popcount64_2 = %f seconds\n", ((float)t2 - t1) / CLOCKS_PER_SEC);
 
    t1 = clock();
    for (j = 0; j < 1000000; j++)
        for (i = 0; i < 1024; i++)
            pt = popcount64_3(data[i]);
    t2 = clock();

    printf("popcount64_3 = %d clocks\n", t2 - t1);
    printf("popcount64_3 = %f seconds\n", ((float)(t2 - t1)) / CLOCKS_PER_SEC);
 
    t1 = clock();
    for (j = 0; j < 1000000; j++)
        for (i = 0; i < 1024; i++)
            pt = popcount64_4(data[i]);
    t2 = clock();

    printf("popcount64_4 = %d clocks\n", t2 - t1);
    printf("popcount64_4 = %f clocks\n", ((float)(t2 - t1)) / CLOCKS_PER_SEC);
 
    return 0;
}


# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <pthread.h>
# include <immintrin.h>
# include "simdxorshift128plus.h"

volatile long long int total_cycle = 0;
pthread_mutex_t lock;

// use simdxorshift128plus start

static void xorshift128plus_jump_onkeys(uint64_t in1, uint64_t in2, uint64_t * output1, uint64_t * output2) {
    /* used by avx_xorshift128plus_init */
    static const uint64_t JUMP[] = { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
        for (int b = 0; b < 64; b++) {
            if (JUMP[i] & 1ULL << b) {
                s0 ^= in1;
                s1 ^= in2;
            }
            //xorshift128plus_onkeys(&in1, &in2);
            uint64_t ps1 = in1;
            const uint64_t ps0 = in2;
            in1 = ps0;
            ps1 ^= ps1 << 23; // a
            in2 = ps1 ^ ps0 ^ (ps1 >> 18) ^ (ps0 >> 5); // b, c
        }
    output1[0] = s0;
    output2[0] = s1;
}

void avx_xorshift128plus_init(uint64_t key1, uint64_t key2, avx_xorshift128plus_key_t *key) {
    uint64_t S0[4];
    uint64_t S1[4];
    S0[0] = key1;
    S1[0] = key2;
    xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
    xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
    xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
    key->part1 = _mm256_loadu_si256((const __m256i *) S0);
    key->part2 = _mm256_loadu_si256((const __m256i *) S1);
}


__m256i avx_xorshift128plus(avx_xorshift128plus_key_t *key) {
    __m256i s1 = key->part1;
    const __m256i s0 = key->part2;
    key->part1 = key->part2;
    s1 = _mm256_xor_si256(key->part2, _mm256_slli_epi64(key->part2, 23));
    key->part2 = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_xor_si256(s1, s0),
                    _mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
    return _mm256_add_epi64(key->part2, s0);
}

// use simdxorshift128plus end


// function to calculate pi
void* cal(void* task){
    long long int tasks = *(int*)task;
    long long int cycles = 0;
    
    avx_xorshift128plus_key_t key;
    avx_xorshift128plus_init(324, 4444, &key);
    __m256 full = _mm256_set_ps(INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX);

    for (int i = 0; i < tasks; i+=8){

        float sum[8];

    	__m256i xi = avx_xorshift128plus(&key);
    	__m256 xf = _mm256_cvtepi32_ps(xi);
    	__m256 x = _mm256_div_ps(xf, full);

    	__m256i yi = avx_xorshift128plus(&key);
    	__m256 yf = _mm256_cvtepi32_ps(yi);
    	__m256 y = _mm256_div_ps(yf, full);

    	__m256 xx = _mm256_mul_ps(x, x);
    	__m256 yy = _mm256_mul_ps(y, y);
    	__m256 zz = _mm256_add_ps(xx, yy);

    	_mm256_store_ps(sum, zz);

    	for (int i = 0; i < 8; i++){
    	    if (sum[i] <= 1.f) {
                cycles++;
            }
    	}
    }

    pthread_mutex_lock(&lock);
    total_cycle += cycles;
    pthread_mutex_unlock(&lock);
}

int main(int argc, char** argv){
    int threads = atoi(argv[1]);
    long long int tosses = atoll(argv[2]);

    pthread_t* pt;
    pt = (pthread_t*)malloc(threads * sizeof(pthread_t));
    pthread_mutex_init(&lock, NULL);
    long long int tasks = tosses / threads;    

    for (int i = 0; i < threads; i++){
    	if (i == threads - 1){
    	    tasks += tosses % threads;
    	    pthread_create(&pt[i], NULL, cal, (void*)&tasks);
    	}
    	else{
    	    pthread_create(&pt[i], NULL, cal, (void*)&tasks);
    	}
    }

    for (int i = 0; i < threads; i++){
	   pthread_join(pt[i], NULL);
    }

    free(pt);
    pthread_mutex_destroy(&lock);
    float pi = 4 * total_cycle / ((float) tosses);
    printf("%lf\n", pi);

    return 0;
}

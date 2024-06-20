extern "C" {
    __device__ unsigned long long xoroshiro128plus(unsigned long long *s)
    {
        unsigned long long s0 = s[0];
        unsigned long long s1 = s[1];
        unsigned long long result = s0 + s1;

        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14); // a, b
        s[1] = (s1 << 36) | (s1 >> 28); // c

        return result;
    }

    __global__ void generate_random_bytes(unsigned char *output, unsigned long long *state, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            unsigned long long random_value = xoroshiro128plus(state);
            output[idx] = random_value & 0xFF;
        }
    }
}

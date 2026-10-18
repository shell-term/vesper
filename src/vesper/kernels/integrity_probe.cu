// SHA-256, FIPS 180-4 sections 4.1.2, 4.2.2, 5.1.1 and 6.2.
// https://doi.org/10.6028/NIST.FIPS.180-4
// Independent implementation. One thread hashes one consecutive 4 KiB chunk.
// Integer integrity checks only; no model arithmetic or weight transformation.

__device__ __constant__ unsigned int sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

__device__ __forceinline__ unsigned int sha_rotr(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}

extern "C" __global__ void sha256_chunks(const unsigned char *data, unsigned char *digests, int bytes) {
    int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    int start = chunk * 4096;
    if (start >= bytes) return;
    int length = min(4096, bytes - start);
    int padded = ((length + 9 + 63) / 64) * 64;
    unsigned int h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    for (int block = 0; block < padded; block += 64) {
        unsigned int w[64];
        #pragma unroll
        for (int word = 0; word < 16; ++word) {
            unsigned int value = 0;
            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                int position = block + word * 4 + byte;
                unsigned int octet = 0;
                if (position < length) octet = data[start + position];
                else if (position == length) octet = 0x80;
                else if (position >= padded - 8) {
                    unsigned long long bits = (unsigned long long)length * 8;
                    octet = (bits >> (8 * (padded - 1 - position))) & 255;
                }
                value = (value << 8) | octet;
            }
            w[word] = value;
        }
        #pragma unroll
        for (int i = 16; i < 64; ++i) {
            unsigned int x = w[i - 15], y = w[i - 2];
            unsigned int s0 = sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3);
            unsigned int s1 = sha_rotr(y, 17) ^ sha_rotr(y, 19) ^ (y >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        unsigned int a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], hh=h[7];
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            unsigned int s1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
            unsigned int ch = (e & f) ^ (~e & g);
            unsigned int t1 = hh + s1 + ch + sha256_k[i] + w[i];
            unsigned int s0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
            unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
            unsigned int t2 = s0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            digests[chunk * 32 + i * 4 + byte] = (h[i] >> (24 - byte * 8)) & 255;
        }
    }
}

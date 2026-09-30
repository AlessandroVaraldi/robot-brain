/* See host.h. */
#define _POSIX_C_SOURCE 199309L       /* clock_gettime, with -std=c11 on glibc */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "host.h"

esp_err_t usb_serial_jtag_driver_install(usb_serial_jtag_driver_config_t *cfg)
{
    (void)cfg;
    return ESP_OK;
}

int usb_serial_jtag_read_bytes(void *buf, uint32_t length, uint32_t ticks)
{
    (void)ticks;
    size_t n = fread(buf, 1, length, stdin);
    if (n == 0) exit(0);                        /* the other end hung up */
    return (int)n;
}

int usb_serial_jtag_write_bytes(const void *src, size_t size, uint32_t ticks)
{
    (void)ticks;
    fwrite(src, 1, size, stdout);
    fflush(stdout);
    return (int)size;
}

int esp_efuse_find_purpose(esp_efuse_purpose_t purpose, esp_efuse_block_t *block)
{
    (void)purpose;
    *block = EFUSE_BLK_KEY0;
    return getenv("CHIP_HOST_NOKEY") == NULL;
}

int64_t esp_timer_get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/* SHA-256 (FIPS 180-4) and HMAC (RFC 2104), small and slow. */
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR(x, n) ((x) >> (n) | (x) << (32 - (n)))

static void block(uint32_t h[8], const uint8_t *p)
{
    uint32_t w[64], a[8];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | p[4 * i + 1] << 16 | p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ w[i - 15] >> 3;
        uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ w[i - 2] >> 10;
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    memcpy(a, h, sizeof a);
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = a[7] + (ROR(a[4], 6) ^ ROR(a[4], 11) ^ ROR(a[4], 25))
                      + ((a[4] & a[5]) ^ (~a[4] & a[6])) + K[i] + w[i];
        uint32_t t2 = (ROR(a[0], 2) ^ ROR(a[0], 13) ^ ROR(a[0], 22))
                      + ((a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]));
        memmove(a + 1, a, 7 * sizeof *a);
        a[4] += t1;
        a[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++) h[i] += a[i];
}

static void sha256(const uint8_t *m, size_t n, uint8_t out[32])
{
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    size_t i = 0;
    for (; i + 64 <= n; i += 64) block(h, m + i);
    uint8_t tail[128] = {0};
    size_t r = n - i, len = r + 1 + 8 <= 64 ? 64 : 128;
    memcpy(tail, m + i, r);
    tail[r] = 0x80;
    for (int j = 0; j < 8; j++) tail[len - 1 - j] = (uint8_t)((uint64_t)n * 8 >> 8 * j);
    for (size_t j = 0; j < len; j += 64) block(h, tail + j);
    for (int j = 0; j < 32; j++) out[j] = (uint8_t)(h[j / 4] >> (24 - 8 * (j % 4)));
}

esp_err_t esp_hmac_calculate(hmac_key_id_t key_id, const void *message, size_t len, uint8_t *mac)
{
    (void)key_id;
    uint8_t pad[64 + 256], inner[32];
    if (len > 256) return ESP_FAIL;
    for (int i = 0; i < 64; i++) pad[i] = (uint8_t)(i < 32 ? i : 0) ^ 0x36;
    memcpy(pad + 64, message, len);
    sha256(pad, 64 + len, inner);
    for (int i = 0; i < 64; i++) pad[i] = (uint8_t)(i < 32 ? i : 0) ^ 0x5c;
    memcpy(pad + 64, inner, 32);
    sha256(pad, 64 + 32, mac);
    return ESP_OK;
}

void app_main(void);

int main(void)
{
    app_main();                                 /* on the chip, ESP-IDF calls it */
    return 0;
}

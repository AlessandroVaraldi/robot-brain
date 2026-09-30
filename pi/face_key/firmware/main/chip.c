/* John's key chip: HMAC-SHA256 under a key burnt into an eFuse of this
 * ESP32-C3. The key is read-protected: the HMAC peripheral uses it, nothing
 * can read it, not even this program.
 *
 * Over the USB serial port, one request per line:
 *
 *   HELLO              -> JOHN-CHIP 1 KEY   (or NOKEY: no key burnt yet)
 *   MAC <label> <hex>  -> OK <64 hex digits> | BUSY | ERR <why>
 *
 * The MAC is of "<label>:<data>", the label one of locks, check, memory,
 * names, and the data at most 64 bytes. Every MAC costs a token; tokens come
 * back at one a second, up to 20, so the chip cannot be used to try keys
 * quickly. This is SoftChip in pi/face_key/chip.py, in hardware.
 */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_efuse.h"
#include "esp_hmac.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

#define LINE_MAX_LEN 200
#define DATA_MAX 64
#define PER_SECOND 1.0
#define BURST 20.0

static const char *LABELS[] = {"locks", "check", "memory", "names"};

static int key_id = -1;         /* the HMAC key block, or -1 if none is burnt */
static double tokens = BURST;
static int64_t last_us;

static void reply(const char *s)
{
    usb_serial_jtag_write_bytes(s, strlen(s), portMAX_DELAY);
    usb_serial_jtag_write_bytes("\n", 1, portMAX_DELAY);
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* The bytes in `hex`, or -1 if it is not hex or longer than `max` bytes. */
static int unhex(const char *hex, uint8_t *out, size_t max)
{
    size_t n = strlen(hex);
    if (n % 2 || n / 2 > max) return -1;
    for (size_t i = 0; i < n / 2; i++) {
        int hi = hex_digit(hex[2 * i]), lo = hex_digit(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (uint8_t)(hi << 4 | lo);
    }
    return (int)(n / 2);
}

static bool spend(void)
{
    int64_t now = esp_timer_get_time();
    tokens += (double)(now - last_us) / 1e6 * PER_SECOND;
    if (tokens > BURST) tokens = BURST;
    last_us = now;
    if (tokens < 1) return false;
    tokens -= 1;
    return true;
}

static void handle(char *line)
{
    if (strcmp(line, "HELLO") == 0) {
        reply(key_id >= 0 ? "JOHN-CHIP 1 KEY" : "JOHN-CHIP 1 NOKEY");
        return;
    }
    if (strncmp(line, "MAC ", 4) != 0) {
        reply("ERR unknown request");
        return;
    }
    char *label = line + 4;
    char *hex = strchr(label, ' ');
    if (hex) *hex++ = '\0';
    else hex = "";

    bool known = false;
    for (size_t i = 0; i < sizeof LABELS / sizeof *LABELS; i++)
        known |= strcmp(label, LABELS[i]) == 0;
    if (!known) {
        reply("ERR unknown label");
        return;
    }
    uint8_t msg[8 + DATA_MAX];
    size_t n = strlen(label);
    memcpy(msg, label, n);
    msg[n++] = ':';
    int len = unhex(hex, msg + n, DATA_MAX);
    if (len < 0) {
        reply("ERR bad data");
        return;
    }
    if (key_id < 0) {
        reply("ERR no key");
        return;
    }
    if (!spend()) {
        reply("BUSY");
        return;
    }
    uint8_t mac[32];
    if (esp_hmac_calculate((hmac_key_id_t)key_id, msg, n + (size_t)len, mac) != ESP_OK) {
        reply("ERR hmac");
        return;
    }
    char out[3 + 64 + 1] = "OK ";
    for (int i = 0; i < 32; i++)
        snprintf(out + 3 + 2 * i, 3, "%02x", mac[i]);
    reply(out);
}

void app_main(void)
{
    esp_efuse_block_t block;
    if (esp_efuse_find_purpose(ESP_EFUSE_KEY_PURPOSE_HMAC_UP, &block))
        key_id = HMAC_KEY0 + (block - EFUSE_BLK_KEY0);

    usb_serial_jtag_driver_config_t cfg = {.tx_buffer_size = 256, .rx_buffer_size = 256};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
    last_us = esp_timer_get_time();

    char line[LINE_MAX_LEN + 1];
    size_t len = 0;
    bool too_long = false;
    for (;;) {
        uint8_t c;
        if (usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY) != 1) continue;
        if (c == '\r') continue;
        if (c != '\n') {
            if (len < LINE_MAX_LEN) line[len++] = (char)c;
            else too_long = true;
            continue;
        }
        line[len] = '\0';
        if (too_long) reply("ERR too long");
        else if (len) handle(line);
        len = 0;
        too_long = false;
    }
}

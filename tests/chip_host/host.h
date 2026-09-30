/* Just enough of ESP-IDF to build firmware/main/chip.c on a computer, for
 * tests/test_chip_firmware.py: the USB port is stdin and stdout, the eFuse
 * key is bytes 0..31 (or none, with CHIP_HOST_NOKEY set). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL -1
#define ESP_ERROR_CHECK(x) ((void)(x))
#define portMAX_DELAY 0xffffffffu

typedef struct { uint32_t tx_buffer_size, rx_buffer_size; } usb_serial_jtag_driver_config_t;
esp_err_t usb_serial_jtag_driver_install(usb_serial_jtag_driver_config_t *cfg);
int usb_serial_jtag_read_bytes(void *buf, uint32_t length, uint32_t ticks_to_wait);
int usb_serial_jtag_write_bytes(const void *src, size_t size, uint32_t ticks_to_wait);

typedef enum { EFUSE_BLK_KEY0 = 4 } esp_efuse_block_t;
typedef enum { ESP_EFUSE_KEY_PURPOSE_HMAC_UP = 8 } esp_efuse_purpose_t;
int esp_efuse_find_purpose(esp_efuse_purpose_t purpose, esp_efuse_block_t *block);

typedef enum { HMAC_KEY0 = 0 } hmac_key_id_t;
esp_err_t esp_hmac_calculate(hmac_key_id_t key_id, const void *message, size_t message_len,
                             uint8_t *hmac);

int64_t esp_timer_get_time(void);

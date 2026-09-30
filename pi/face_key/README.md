# face_key

How John recognises people by face without keeping anything that links a
face to a name. The brain uses it with `one_mind.py --face-check`
(`john/encounters.py`); for now it runs in the brain's own process, so the
chip goes on the USB of the machine running the brain.

- **Consent first.** Only someone who has told John their name and agreed to
  have their face remembered is enrolled. They can ask to be forgotten.
- **The face is the identity, the name is an attribute.** A person is whoever
  their face opens; the name is something John knows about them, which they
  can change ("UPDATE MY NAME TO ...", confirmed with a YES).
- **No faces are stored.** Each person has up to three *locks* (`lock.py`).
  Each lock holds a random key that only a similar enough face gets back. The
  key has 120 bits, protected by a code that corrects 56 of 510 face bits.
- **A key that stays on the robot.** The locks are encrypted, and every key
  they give is checked, with an HMAC key held by a small chip (`chip.py`),
  which answers slowly on purpose. A copy of the stored data opens nothing
  without the chip. `SoftChip` stands in for it during development.
- **Two frames, one person.** Each of two frames must open the same
  person's lock on its own. With a single frame, a father opened his
  daughter's lock on LFW.
- **Admin.** Names are also sealed with the admin's public key. The robot can
  write them but not read them. With the private key, kept off the robot,
  `admin.py` lists the names and deletes people by name, so a misread name
  does not stay attached to a face.

```
python3 -m face_key.admin keygen ~/.john/admin.key            # prints the public key: put it in memory/admin.pub
python3 -m face_key.admin list --key - < ~/.john/admin.key
python3 -m face_key.admin delete Luca --key - < ~/.john/admin.key    # --yes to delete
```

Run these from `pi/`. Requirements: `numpy`, `bchlib`, `cryptography`, and
`pyserial` for the chip. `bchlib` has no build for the Pi's architecture and
compiles on install.

## The chip

`firmware/` is the program for an ESP32-C3 on the Pi's USB (ESP-IDF 6.0).
`chip.c` describes the protocol. `tests/test_chip_firmware.py` builds the same
code for a computer and checks it against `SoftChip`.

```
cd pi/face_key/firmware
idf.py build
idf.py -p /dev/ttyACM0 flash        # reversible: HELLO answers NOKEY until a key is burnt
```

Putting the key in cannot be undone, so it is done by hand, with the chip in
front of you. On the Pi:

```
head -c 32 /dev/urandom > key.bin
espefuse --chip esp32c3 -p /dev/ttyACM0 burn-key BLOCK_KEY0 key.bin HMAC_UP
shred -u key.bin
```

The key block becomes read-protected: from then on the key exists only inside
the chip. If the chip breaks, everyone has to be enrolled again.

## Measured

On LFW (press photos, SFace, frontal faces only), with `eval/face_key_lfw.py`:
- 328 people enrolled from two photos each, another 328 never enrolled.
- 90.2% of pairs of further photos recognised, and 94.6% of people with
  several further photos recognised from some consecutive pair.
- Nobody was taken for someone else, and no pair of a stranger's photos was
  recognised (2,207 pairs).
- Two frames checked against everyone take 39 ms on a laptop.

The robot's camera, lighting and distance are not LFW: the numbers that count
will be the ones measured there.

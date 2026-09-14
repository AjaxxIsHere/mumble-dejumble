#!/usr/bin/env python3
"""Patch Qwen3.5 GGUF metadata: phantom MTP layer (block_count=25) -> 24.

The base checkpoint has no MTP/nextn weights, but the converter inherited
mtp_num_hidden_layers=1 from the config template and wrote block_count=25
into the header while the tensor table only contains blk.0..blk.23.
Runtimes trust the header and fail with "missing tensor 'blk.24...'".
(Canonical upstream fix is convert_hf_to_gguf.py --no-mtp.)

Rewrites the GGUF v3 header:
  - <arch>.block_count: 25 -> 24
  - <arch>.nextn_predict_layers: key dropped entirely
  - <arch>.attention.recurrent_layers: truncated to 24 entries

File layout per GGUF spec: header+KV, then tensor infos (unaligned),
then zero padding to general.alignment (default 32), then tensor data.
Tensor data is copied verbatim; the infos/pad/data boundaries are
recomputed so tensor offsets (relative to data start) stay valid.
"""
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b'GGUF'

# GGUF metadata value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARRAY, U64, I64, F64 = range(13)
FMT = {U8: '<B', I8: '<b', U16: '<H', I16: '<h', U32: '<I', I32: '<i',
       F32: '<f', BOOL: '<B', U64: '<Q', I64: '<q', F64: '<d'}


class Reader:
    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def scalar(self, fmt):
        (v,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.scalar('<Q')
        s = self.buf[self.pos:self.pos + n].decode('utf-8')
        self.pos += n
        return s

    def value(self, t):
        if t == STR:
            return self.string()
        if t == ARRAY:
            et = self.scalar('<I')
            n = self.scalar('<Q')
            return (et, [self.value(et) for _ in range(n)])
        return self.scalar(FMT[t])


def enc_str(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def enc_value(t, v):
    if t == STR:
        return enc_str(v)
    if t == ARRAY:
        et, items = v
        return (struct.pack('<I', et) + struct.pack('<Q', len(items))
                + b''.join(enc_value(et, x) for x in items))
    return struct.pack(FMT[t], v)


def align_up(n, a):
    return (n + a - 1) // a * a


def tensor_infos_size(raw, pos, n_tensors):
    """Byte size of the tensor info array starting at pos."""
    start = pos
    for _ in range(n_tensors):
        n = struct.unpack_from('<Q', raw, pos)[0]
        pos += 8 + n                       # name
        ndims = struct.unpack_from('<I', raw, pos)[0]
        pos += 4 + 8 * ndims               # dims
        pos += 4                           # ggml type
        pos += 8                           # data offset
    return pos - start


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.fixed.gguf')

    raw = src.read_bytes()
    r = Reader(raw)
    if raw[:4] != GGUF_MAGIC:
        sys.exit('not a GGUF file')
    r.pos = 4
    version = r.scalar('<I')
    if version != 3:
        sys.exit(f'expected GGUF v3, got v{version}')
    n_tensors = r.scalar('<Q')
    n_kv = r.scalar('<Q')
    print(f'src: v{version}, {n_tensors} tensors, {n_kv} kv pairs')

    kvs = {}
    for _ in range(n_kv):
        key = r.string()
        t = r.scalar('<I')
        kvs[key] = (t, r.value(t))
    infos_start = r.pos  # tensor info array starts immediately after KV

    arch = kvs['general.architecture'][1]
    bc_key = f'{arch}.block_count'
    ntp_key = f'{arch}.nextn_predict_layers'
    rl_key = f'{arch}.attention.recurrent_layers'
    if bc_key not in kvs:
        sys.exit(f'{bc_key} not found')

    old_bc = kvs[bc_key][1]
    if old_bc == 24 and ntp_key not in kvs:
        print('already patched, nothing to do')
        return

    # --- metadata mutations ---
    kvs[bc_key] = (U32, 24)
    dropped_ntp = kvs.pop(ntp_key, None) is not None
    if rl_key in kvs:
        t, (et, items) = kvs[rl_key]
        assert t == ARRAY
        kvs[rl_key] = (ARRAY, (et, items[:24]))

    print(f'block_count: {old_bc} -> 24')
    print(f'nextn_predict_layers: {"dropped" if dropped_ntp else "not present"}')
    if rl_key in kvs:
        print(f'recurrent_layers: truncated to 24 entries')

    # --- layout of source ---
    alignment = kvs.get('general.alignment', (U32, 32))[1]
    infos_sz = tensor_infos_size(raw, infos_start, n_tensors)
    src_data_start = align_up(infos_start + infos_sz, alignment)
    assert src_data_start <= len(raw), 'data start beyond EOF'
    print(f'src: infos@{infos_start} (+{infos_sz}B), data@{src_data_start} '
          f'(alignment {alignment})')

    # --- serialize: header + KV, infos verbatim, re-aligned pad, data verbatim ---
    out = bytearray()
    out += GGUF_MAGIC
    out += struct.pack('<I', version)
    out += struct.pack('<Q', n_tensors)
    out += struct.pack('<Q', len(kvs))
    for key, (t, v) in kvs.items():
        out += enc_str(key)
        out += struct.pack('<I', t)
        out += enc_value(t, v)

    out += raw[infos_start:infos_start + infos_sz]          # tensor infos
    out += b'\x00' * (align_up(len(out), alignment) - len(out))  # data pad
    out += raw[src_data_start:]                             # tensor data

    dst.write_bytes(out)
    print(f'wrote {dst} ({len(out)} bytes; src {len(raw)})')


if __name__ == '__main__':
    main()

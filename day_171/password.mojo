from gpu.id import thread_idx, block_idx, block_dim
from memory import UnsafePointer, atomic_store
from string import ascii_lowercase
from sync import volatile_load

const BASE: UInt32 = 0x811c9dc5
const PRIME: UInt32 = 0x01000193
const ALPHABET = 26
const MAX_PWD_LEN = 8
const BLOCK_SIZE = 256

fn fnv1a_rounds(input_str: UnsafePointer[UInt8], length: Int32, rounds: Int32) -> UInt32:
    var hash: UInt32 = 0
    for r in range(rounds):
        hash = BASE
        for i in range(length):
            hash = hash ^ input_str[i]
            hash = hash * PRIME
    return hash

# Converts number to base-26 password of given length
fn index_to_password(index: Int64, length: Int32, buffer: UnsafePointer[UInt8]):
    var value = index
    for i in range(length - 1, -1, -1):
        let char_code = value % ALPHABET
        buffer[i] = ('a'.ord + char_code).to(UInt8)
        value = value // ALPHABET

@kernel
fn password_crack_kernel(
    target_hash: UInt32,
    password_length: Int32,
    R: Int32,
    found: UnsafePointer[Int32],
    output_password: UnsafePointer[UInt8]
):
    let idx = block_idx.x * block_dim.x + thread_idx.x
    let stride = block_dim.x * grid_dim.x
    let total = pow(ALPHABET, password_length).to(Int64)

    var local_pwd: UInt8[MAX_PWD_LEN]
    var i = idx.to(Int64)
    while i < total and volatile_load(found) == 0:
        index_to_password(i, password_length, &local_pwd)
        let hash = fnv1a_rounds(&local_pwd, password_length, R)
        if hash == target_hash:
            for j in range(password_length):
                output_password[j] = local_pwd[j]
            output_password[password_length] = 0  # null terminator
            atomic_store(found, 1)
        i += stride

@export
def solve(
    target_hash: UInt32,
    password_length: Int32,
    R: Int32,
    output_password: UnsafePointer[UInt8]
):
    from gpu.host import DeviceContext
    var ctx = DeviceContext()

    var d_found = ctx.alloc_device_memory 
    ctx.memset(d_found, 0, 1)

    let grid_size = 1024
    ctx.enqueue_function[password_crack_kernel](
        target_hash, password_length, R, d_found, output_password,
        grid_dim=(grid_size,), block_dim=(BLOCK_SIZE,)
    )
    ctx.synchronize()

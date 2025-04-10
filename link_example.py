from numba import cuda
import numpy as np

cu_code = cuda.CUSource("""
extern "C" __device__
int bar(int *out, int a)
{
  *out = a * 2;
  return 0;
}
""")

bar = cuda.declare_device('bar', 'int32(int32)', link=cu_code)


@cuda.jit("void(int32[::1], int32[::1])")
def foo(r, x):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = bar(x[i])


x = np.arange(10, dtype=np.int32)
r = np.empty_like(x)

foo[1, 32](r, x)
print(x)
print(r)

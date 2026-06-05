#!/usr/bin/env python3
"""探测 MUJOCO_EGL_DEVICE_ID (= EGL 设备枚举下标) <-> nvidia-smi GPU 的物理映射。

纯查询:只调 eglQueryDevicesEXT / eglQueryDeviceStringEXT / eglQueryDeviceAttribEXT,
不创建渲染 context、不 import mujoco、几乎不占显存,因此与任何正在跑的训练进程完全隔离。

MuJoCo 的 EGL 后端正是用 eglQueryDevicesEXT 拿到设备数组、再用 MUJOCO_EGL_DEVICE_ID
做下标,所以本脚本枚举出的下标语义 == MUJOCO_EGL_DEVICE_ID 的语义。
"""
import ctypes, os, subprocess


def load_egl():
    for name in ("libEGL.so.1", "libEGL.so", "libEGL_nvidia.so.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit("找不到 libEGL")


egl = load_egl()
egl.eglGetProcAddress.restype = ctypes.c_void_p
egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]


def proc(name, restype, argtypes):
    addr = egl.eglGetProcAddress(name.encode())
    if not addr:
        return None
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)


EGLDevice = ctypes.c_void_p
EGLAttrib = ctypes.c_ssize_t
qDevices = proc("eglQueryDevicesEXT", ctypes.c_uint,
                [ctypes.c_int, ctypes.POINTER(EGLDevice), ctypes.POINTER(ctypes.c_int)])
qString = proc("eglQueryDeviceStringEXT", ctypes.c_char_p, [EGLDevice, ctypes.c_int])
qAttrib = proc("eglQueryDeviceAttribEXT", ctypes.c_uint,
               [EGLDevice, ctypes.c_int, ctypes.POINTER(EGLAttrib)])
if not qDevices:
    raise SystemExit("驱动不支持 eglQueryDevicesEXT 扩展")

EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377
EGL_CUDA_DEVICE_NV = 0x323A

MAX = 32
devs = (EGLDevice * MAX)()
n = ctypes.c_int(0)
qDevices(MAX, devs, ctypes.byref(n))


def norm(pci):
    if not pci:
        return None
    p = pci.lower()
    return p.split(":", 1)[1] if p.count(":") >= 2 else p  # 去掉 domain,留 bus:dev.func


def node_to_pci(path):
    if not path:
        return None
    base = os.path.basename(path)
    try:
        real = os.path.realpath(f"/sys/class/drm/{base}/device")
        cand = os.path.basename(real)
        return cand if (":" in cand and "." in cand) else os.path.basename(os.path.dirname(real))
    except Exception:
        return None


# nvidia-smi 真值表: pci -> {index, uuid}
smi = {}
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"]).decode()
for line in out.strip().splitlines():
    idx, uuid, pci = [x.strip() for x in line.split(",")]
    smi[norm(pci)] = {"index": idx, "uuid": uuid}

print(f"EGL 枚举到 {n.value} 个设备\n")
hdr = f"{'MUJOCO_EGL_DEVICE_ID':>20} | {'render_node':>12} | {'PCI':>11} | {'CUDA_NV':>7} | {'nvidia-smi':>10} | UUID"
print(hdr)
print("-" * len(hdr))
gpu0_egl = None
for i in range(n.value):
    d = devs[i]
    card = qString(d, EGL_DRM_DEVICE_FILE_EXT) if qString else None
    rnode = qString(d, EGL_DRM_RENDER_NODE_FILE_EXT) if qString else None
    card = card.decode() if card else None
    rnode = rnode.decode() if rnode else None
    pci = node_to_pci(rnode) or node_to_pci(card)
    cuda = ctypes.c_ssize_t(-1)
    if qAttrib:
        qAttrib(d, EGL_CUDA_DEVICE_NV, ctypes.byref(cuda))
    rec = smi.get(norm(pci))
    smi_idx = rec["index"] if rec else "?"
    uuid = rec["uuid"] if rec else "?"
    if smi_idx == "0":
        gpu0_egl = i
    node = os.path.basename(rnode or card or "?")
    print(f"{i:>20} | {node:>12} | {str(norm(pci)):>11} | {cuda.value:>7} | {smi_idx:>10} | {uuid}")

print()
if gpu0_egl is not None:
    print(f">>> 要把渲染钉到 nvidia-smi gpu0,应设 MUJOCO_EGL_DEVICE_ID = {gpu0_egl}")
else:
    print(">>> 没匹配到 nvidia-smi gpu0(可能 DRM 节点查询失败),请看上表 CUDA_NV 列人工对齐")

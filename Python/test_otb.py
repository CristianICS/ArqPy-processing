import sys, os, pathlib, ctypes

OTB = r"C:\OTB-9.1.1-Win64"
py  = os.path.join(OTB, "lib", "otb", "python")
bin = os.path.join(OTB, "bin")
lib = os.path.join(OTB, "lib")

print("Python exe:", sys.executable)
print("64-bit?   :", sys.maxsize > 2**32)
try:
    import numpy as np
    print("NumPy     :", np.__version__)
except Exception as e:
    print("NumPy     : ERROR ->", e)

print("PYTHONPATH contains OTB python? ", py in os.environ.get("PYTHONPATH",""))
print("PATH head :", os.environ["PATH"].split(";")[:5])

# Help the loader (even if PATH is correct)
os.add_dll_directory(bin)
os.add_dll_directory(lib)

# Try importing OTB normally
try:
    import otbApplication as otb
    print("OTB import OK. #apps:", len(otb.Registry.GetAvailableApplications()))
except Exception as e:
    print("Import failed:", type(e).__name__, e)
    # Load the extension directly to get the exact missing DLL
    pyd = pathlib.Path(py) / "_otbApplication.pyd"
    try:
        ctypes.WinDLL(str(pyd))
        print("WinDLL load OK (unexpected).")
    except OSError as e2:
        print("WinDLL error for _otbApplication.pyd:", e2)

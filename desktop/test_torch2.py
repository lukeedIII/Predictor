import sys, traceback
try:
    import torch
    sys.stdout.write('OK: ' + torch.__version__ + '\n')
    sys.stdout.flush()
except Exception as e:
    sys.stdout.write('FAIL: ' + str(e)[:300] + '\n')
    sys.stdout.flush()
    traceback.print_exc()

import sys
try:
    import torch
    sys.stdout.write('torch ' + torch.__version__ + ' OK, CUDA: ' + str(torch.cuda.is_available()) + '\n')
    sys.stdout.flush()
    import xgboost
    sys.stdout.write('xgboost OK\n')
    sys.stdout.flush()
    import pandas
    sys.stdout.write('pandas OK\n')
    sys.stdout.flush()
    import fastapi
    sys.stdout.write('fastapi OK\n')
    sys.stdout.flush()
    import sklearn
    sys.stdout.write('sklearn OK\n')
    sys.stdout.flush()
    sys.stdout.write('ALL PASS\n')
    sys.stdout.flush()
except Exception as e:
    sys.stdout.write('FAIL: ' + str(e)[:300] + '\n')
    sys.stdout.flush()

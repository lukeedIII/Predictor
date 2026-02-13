import sys
try:
    import torch
    sys.stdout.write('TORCH OK: ' + torch.__version__ + ', CUDA: ' + str(torch.cuda.is_available()) + '\n')
    import xgboost
    sys.stdout.write('XGBoost OK\n')
    import pandas
    sys.stdout.write('Pandas OK\n')
    import fastapi
    sys.stdout.write('FastAPI OK\n')
    sys.stdout.write('ALL IMPORTS PASS\n')
except Exception as e:
    sys.stdout.write('FAIL: ' + str(e)[:300] + '\n')
    import traceback
    traceback.print_exc()

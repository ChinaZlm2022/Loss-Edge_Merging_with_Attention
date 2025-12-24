import torch
import random
from train import main

if __name__ == '__main__':
    # 设置随机种子
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 开始训练
    main()
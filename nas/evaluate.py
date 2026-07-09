import torch
import time
from torch import nn

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

# evaluate needs the actual model (nn.Module, actual dataset)
def evaluate(model, dataset, device, skip_test=False):
    model.to(device)
    model.eval()
    metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        ## Warmup starts
        for batch_idx, (x, target) in enumerate(dataset):
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            if batch_idx == 10:
                break

        ## Warmup ends
        time_list = []
        for batch_idx, (x, target) in enumerate(dataset):
            if batch_idx >= 10:
                break
            x = x.to(device)
            target = target.to(device)

            start = cuda_time()
            pred = model(x)
            end = cuda_time()

            time_list.append((end - start) * 1000)
            if not skip_test:
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        if not skip_test:
            accuracy = metrics["test_correct"] / metrics["test_total"]
        else:
            accuracy = 0
        latency = sum(time_list) / len(time_list)

    return accuracy, latency

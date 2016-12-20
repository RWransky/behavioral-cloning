def accuracy(targets, predicted, tol=1e-2):
    rate = 0
    total = targets.shape[0]
    for i in range(total):
        if abs(targets[i] - predicted[i]) <= tol:
            rate += 1
    return rate/float(total)

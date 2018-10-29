import numpy as np


# copy from [monodepth](https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py)
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    log10_error = np.abs(np.log10(gt)-np.log10(pred))
    log10_mean = np.mean(log10_error)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, log10_mean

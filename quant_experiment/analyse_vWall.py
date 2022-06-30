import copy
import os
import pickle
from glob import glob

import numpy as np
import pandas as pd


def main():
    folder = "output/experiments_vWall"
    df = pd.DataFrame()
    cut_failed = True
    savefiles = []
    experiments = [x for x in os.walk(folder)]
    for exp in experiments[1:]:
        if "vipmp_final_save" in exp[-1]:
            savefiles.append(exp[0] + "/vipmp_final_save")

    for sf in savefiles:
        with open(sf, mode="rb") as f:
            save_vipmp = pickle.load(f)
        df = df.append(
            {
                "n_v_walls": len(save_vipmp.env.v_walls),
                "n_via_points": len(save_vipmp.env.via_points),
                "nt": save_vipmp.nt,
                "nbasis_pmp": save_vipmp.nbasis_pmp,
                "n_dim": save_vipmp.n_dim,
                "n_iter_cpmp": save_vipmp.n_iter_cpmp,
                "n_sub_iter_cpmp": save_vipmp.n_sub_iter_cpmp,
                "lagrange_learning_rate": save_vipmp.lagrange_learning_rate,
                "alpha": save_vipmp.alpha,
                "vipmp_runtime": save_vipmp.runtime,
                "vipmp_violations": save_vipmp.violations,
                "vipmp_kl": save_vipmp.kl_2_prior.numpy() / save_vipmp.nbasis_pmp,
                "vipmp_loss": save_vipmp.loss_hist,
                "vipmp_loss_size": len(save_vipmp.loss_hist),
            },
            ignore_index=True,
        )

    loss_hist = df
    print(
        "vWalls\tn\tfail\t\tviolations\tkl\t\t\tviolations(S)\tkl(S)\t\truntime\t\truntime(S)"
    )
    for i in range(1, 4):
        df_o = df[df.n_v_walls == i]

        failed_vipmp = df_o.vipmp_violations > 0.3
        df_cut_vipmp = df_o[np.logical_not(failed_vipmp)]

        n = df_o.shape[0]
        print(
            f"{i}\t"
            f"{n}\t"
            f"{failed_vipmp.sum():>2}({failed_vipmp.sum()/n*100:>4.1f}%)  "
            f"{df_o.vipmp_violations.mean()*100:>4.1f}% +/- {df_o.vipmp_violations.std()*100:>4.1f}%\t"
            f"{df_o.vipmp_kl.mean():>5.2f} +/- {df_o.vipmp_kl.std():>6.2f}\t"
            f"{df_cut_vipmp.vipmp_violations.mean()*100:.2f}% +/- {df_cut_vipmp.vipmp_violations.std()*100:.2f}%\t"
            f"{df_cut_vipmp.vipmp_kl.mean():>5.2f} +/- {df_cut_vipmp.vipmp_kl.std():>5.2f}\t"
            f"{df_o.vipmp_runtime.mean():>5.1f} +/- {df_o.vipmp_runtime.std():>5.1f}\t"
            f"{df_cut_vipmp.vipmp_runtime.mean():>5.1f} +/- {df_cut_vipmp.vipmp_runtime.std():>5.1f}\t"
        )


if __name__ == "__main__":
    main()

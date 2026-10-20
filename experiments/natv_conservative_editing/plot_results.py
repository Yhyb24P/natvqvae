"""Plot frozen generation statistics; error bars are the observed three-seed range."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=Path(__file__).parent


def main():
    frame=pd.read_csv(RUN/'evaluation/per_seed_summary.csv')
    colors={'profile':'#888888','ESM2':'#0072B2','SE':'#009E73','QVAE':'#D55E00'}
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for row,mode in enumerate(['substitution','sourced_indel_repair']):
        subset=frame[frame['mode']==mode]
        for column,metric in enumerate(['constraint_unique_fraction','within_mask_diversity']):
            ax=axes[row,column]
            for arm,color in colors.items():
                grouped=subset[subset.arm==arm].groupby('edit_budget')[metric]
                mean=grouped.mean();low=grouped.min();high=grouped.max()
                ax.errorbar(mean.index,mean,yerr=np.vstack([mean-low,high-mean]),label=arm,color=color,marker='o',capsize=2,lw=1.3)
            ax.set_ylim(0,1.05);ax.set_xlabel('Total edit budget');ax.grid(axis='y',alpha=.2)
            ax.set_ylabel('Distinct constrained combinations / attempts' if column==0 else 'Mean Hamming fraction at shared edited sites')
            ax.set_title('Substitutions' if row==0 else 'Sourced short InDel + conditional repair')
    axes[0,0].legend(frameon=False,ncol=2)
    out=RUN/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'conservative_comparison.png',dpi=180)
    fig.savefig(out/'conservative_comparison.svg')
    plt.close(fig)
    path=out/'conservative_comparison.svg';path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':main()

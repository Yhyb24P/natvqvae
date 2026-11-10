"""Publication figures from frozen NatV scores, without model execution or fitting."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import load_yaml, sha256_file, write_json
from .evaluate import _bootstrap
from .fragment_comparison import _require

COLORS = {'esm': '#59717D', 'task': '#CC793B', 'single_mixture': '#187B78'}
REGIONS = ('NMNAT', 'Nudix')


def component_means(rows, value):
    groups = sorted({r['component'] for r in rows})
    return np.array([np.mean([float(r[value]) for r in rows if r['component'] == g]) for g in groups])


def paired_geometry(esm, task, mixture):
    x, y = task - esm, mixture - esm
    _require(np.isfinite([x, y]).all(), 'Nonfinite paired scores')
    return x, y, {'task_worse_mix_better': int(((x > 0) & (y < 0)).sum()),
                  'both_better': int(((x < 0) & (y < 0)).sum()),
                  'both_worse': int(((x > 0) & (y > 0)).sum()),
                  'task_better_mix_worse': int(((x < 0) & (y > 0)).sum()),
                  'mix_better_than_esm': int((y < 0).sum()),
                  'mix_better_than_both': int((mixture < np.minimum(esm, task)).sum())}


def _study_design(output):
    from html import escape
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig=plt.figure(figsize=(170/25.4,162.5/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,680),ylim=(650,0));ax.axis('off')
    point_scale=(170/25.4*72)/680
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="170mm" height="162.5mm" viewBox="0 0 680 650" data-generator="fireworks-tech-graph" data-style-id="1" data-diagram-type="architecture">',
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#59717D"/></marker></defs>',
           '<rect width="680" height="650" fill="white"/>']
    svg[0]=svg[0].replace('>', ' role="img" aria-labelledby="study-design-title study-design-desc">',1)
    svg.insert(1,'<title id="study-design-title">NatV study and probability mixing</title><desc id="study-design-desc">Annotated targets, controlled training conditions, and arithmetic mixing of frozen pretrained and trained task-head probabilities.</desc>')
    def text(x,y,label,size=12.5,weight='normal',anchor='start'):
        ax.text(x,y,label,fontsize=size*point_scale,fontweight=weight,ha='center' if anchor=='middle' else 'left',va='baseline',color='#203744')
        svg.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#203744">{escape(label)}</text>')
    def box(x,y,w,h,lines,color='#EAF1F4'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=5',facecolor=color,edgecolor='#59717D',linewidth=.7))
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{color}" stroke="#59717D"/>')
        for i,line in enumerate(lines):text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*17+4,line,anchor='middle')
    def edge(points):
        coords=np.array([[float(v) for v in pair.split(',')] for pair in points.split()])
        ax.plot(coords[:-1,0],coords[:-1,1],color='#59717D',lw=1.6*point_scale)
        ax.annotate('',xy=coords[-1],xytext=coords[-2],arrowprops={'arrowstyle':'-|>','color':'#59717D','lw':1.6*point_scale})
        svg.append(f'<polyline points="{points}" fill="none" stroke="#59717D" stroke-width="1.6" marker-end="url(#arrow)"/>')
    text(20,24,'A  NatV masking task',15,'bold')
    text(20,48,'KVP40 reference: 341 residues')
    box(35,65,324,30,['NMNAT 4–181'],'#E0EEEE');box(359,65,40,30,[])
    box(399,65,246,30,['Nudix 204–338'],'#FAEAD8')
    text(379,56,'182–203',anchor='middle')
    for x in (80,150,240,480,590):
        ax.add_patch(FancyBboxPatch((x,100),8,8,boxstyle='square,pad=0',facecolor='#203744',edgecolor='none'))
        svg.append(f'<rect x="{x}" y="100" width="8" height="8" fill="#203744"/>')
    text(30,118,'Dark marks: illustrative targets; other residues remain visible')
    text(30,139,'27 panels; homolog-specific coordinates; all panels is weighted')
    text(20,179,'B  Original controlled comparison',15,'bold')
    text(20,204,'1,437 external/self fragment pairs: matched target columns and budgets')
    text(325,231,'QVAE',13,'bold',anchor='middle');text(535,231,'Deterministic',13,'bold',anchor='middle')
    for y,name in [(245,'No auxiliary stage'),(283,'Self fragments'),(321,'External fragments')]:
        text(25,y+21,name);box(245,y,160,30,['3 pipeline replicates']);box(455,y,160,30,['3 pipeline replicates'])
    text(20,376,'C  Exploratory mixed prediction: existing no-auxiliary task weights',14,'bold')
    box(20,395,175,32,['Same masked sequence'])
    box(20,465,175,56,['Frozen ESM-2','encoder'])
    box(255,425,180,48,['Original ESM head','Normalized 20AA pE'])
    box(255,530,180,66,['Trained deterministic','task model (one seed)','Normalized 20AA pD'],'#FAEAD8')
    box(490,470,170,70,['Arithmetic pool','pM = 0.5 pE + 0.5 pD'],'#E0EEEE')
    edge('107,427 107,465');edge('195,484 223,484 223,449 255,449')
    edge('195,504 223,504 223,563 255,563')
    edge('435,449 465,449 465,489 490,489');edge('435,563 465,563 465,521 490,521')
    text(20,624,'Mix and score each seed separately; then average scores')
    text(20,643,'Prior six-cell generation is separate from the mixed predictor')
    svg.append('</svg>');data=''.join(svg)
    (output/'study_design.svg').write_text(data)
    for ext in ('pdf','png'): fig.savefig(output/f'study_design.{ext}',dpi=220)
    plt.close(fig)

    fig=plt.figure(figsize=(170/25.4,130/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,680),ylim=(520,0));ax.axis('off')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="170mm" height="130mm" viewBox="0 0 680 520" data-generator="fireworks-tech-graph" data-style-id="1" data-diagram-type="architecture">',
         '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#59717D"/></marker></defs>']
    svg[0]=svg[0].replace('>', ' role="img" aria-labelledby="qvae-paths-title qvae-paths-desc">',1)
    svg.insert(1,'<title id="qvae-paths-title">QVAE information paths</title><desc id="qvae-paths-desc">The full-sequence posterior supports training and importance scoring; generation samples the context-conditioned prior.</desc>')
    text(20,28,'QVAE: posterior inference and prior generation',15,'bold')
    box(20,70,170,64,['Full-sequence','ESM features'])
    box(230,70,180,64,['Posterior','q(z | full, context, M)'])
    box(460,70,180,64,['Posterior z','Train / importance'])
    box(20,215,170,64,['Masked ESM features','Trainable context','encoder (256)'])
    box(230,215,180,64,['Conditional prior','p(z | context, M)'])
    box(460,215,180,64,['Prior z','Generation / scoring'])
    box(460,365,180,64,['Context + sampled z','FiLM decoder','20AA distributions'])
    box(230,365,180,64,['Masked target residues','Joint: shared z','Marginals: independent z'])
    edge('190,102 230,102');edge('410,102 460,102')
    edge('105,215 105,165 320,165 320,134')
    edge('190,247 230,247');edge('410,247 460,247')
    edge('640,102 665,102 665,397 640,397');edge('550,279 550,365')
    edge('105,279 105,340 480,340 480,365');edge('460,397 410,397')
    text(20,475,'16 categorical factors × 8 states; full-sequence features enter only the posterior')
    text(20,500,'Deterministic: context-derived modulation, without posterior or sampled z')
    svg.append('</svg>');(output/'qvae_paths.svg').write_text(''.join(svg))
    for ext in ('pdf','png'):fig.savefig(output/f'qvae_paths.{ext}',dpi=220)
    plt.close(fig)


def render_manuscript_figures(config_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    config = load_yaml(config_path)
    output = Path(output)
    _require(not output.exists() or not any(output.iterdir()), 'Use an empty figure output directory')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {str(config_path), config['six_cell'], config['calibration'], *config['scores'].values()}
    six = json.loads(Path(config['six_cell']).read_text())
    calibration = json.loads(Path(config['calibration']).read_text())
    _require(six['status'] == calibration['status'] == 'COMPLETED', 'Incomplete source results')
    plotted = {'paired': {}, 'accuracy': {}, 'nearest': {}}
    _study_design(output)

    def table(path):
        inputs.add(str(path))
        with Path(path).open() as f:
            return list(csv.DictReader(f))

    def finish(fig, name):
        _require(np.isclose(fig.get_figwidth() * 25.4, 170), 'Figure is not 170 mm wide')
        fig.canvas.draw()
        bounds=fig.get_tightbbox(fig.canvas.get_renderer())
        _require(bounds.x0 >= -.02 and bounds.y0 >= -.02 and bounds.x1 <= fig.get_figwidth()+.02 and bounds.y1 <= fig.get_figheight()+.02, f'Clipped figure labels: {name}')
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(output / f'{name}.{extension}', dpi=220)
        plt.close(fig)

    def panel(ax, title):
        ax.set_title(title, loc='left', fontsize=9.5, fontweight='bold', pad=9)
        ax.spines[['top', 'right']].set_visible(False)

    def forest(ax, rows, labels, colors=None):
        ax.axvline(0, color='.7', lw=.8, ls='--')
        for i, row in enumerate(rows):
            color = colors[i] if colors else '#244B62'
            if row['ci95'] is not None:
                ax.plot(row['ci95'], [i, i], color=color, lw=1.4)
            ax.plot(row['mean'], i, 'o', ms=4, color=color)
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - .5, -.7)
        ax.set_xlabel('ΔNLL (nats / target residue)')
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useOffset=False)

    def estimate(values):
        return _bootstrap(values, config['bootstrap_repeats'], config['bootstrap_seed'])

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        # Equal-width panels keep text at its authored final-size point value.
        fig = plt.figure(figsize=(170 / 25.4, 180 / 25.4), layout='constrained')
        grid = fig.add_gridspec(2, 2, wspace=.12, hspace=.25)
        a, b, c, d = [fig.add_subplot(cell) for cell in grid]
        selected = [r for r in six['comparisons'] if r['scope'] == 'mean_of_pipeline_seeds' and
                    r['identity_stratum'] == 'all_identities' and r['target_sites'] == 'all_sizes']
        lookup = {(r['comparison'], r['region']): r for r in selected}
        specs = [(f'{arm}:{contrast}', region) for region in REGIONS for arm in ('QVAE', 'deterministic')
                 for contrast in ('external_minus_self', 'external_minus_random')]
        forest(a, [lookup[k] for k in specs],
               [f'{"N" if r=="NMNAT" else "Nu"}/{"Q" if m.startswith("QVAE") else "D"} ext−{"self" if m.endswith("self") else "rnd"}' for m, r in specs])
        panel(a, 'A  Initialization contrasts')
        specs = [(f'{arm}:{cond}_minus_frozen_esm2', region) for region in REGIONS
                 for arm in ('QVAE', 'deterministic') for cond in ('random', 'self', 'external')]
        forest(b, [lookup[k] for k in specs],
               [f'{r[0]} / {m.split(":")[0][0]} / {m.split(":")[1].split("_")[0]}' for m, r in specs])
        panel(b, 'B  Task models − ESM-2')
        b.set_yticklabels([f'{"N" if r=="NMNAT" else "Nu"} / {"Q" if m.startswith("QVAE") else "D"} / {m.split(":")[1].split("_")[0]}' for m,r in specs])
        for name, label in [('full_length', 'Full length'), ('nmnat', 'NMNAT'), ('nudix', 'Nudix')]:
            rows = table(Path(config['neighbors']) / f'{name}_both_nearest.csv')
            rows = [r for r in rows if r['role'] == 'test' and r['target_role'] == 'training']
            _require(len(rows) == len({r['sequence_id'] for r in rows}) == 231, 'Nearest-match denominator differs')
            values = np.sort([float(r['fident']) for r in rows])
            if values.max() <= 1: values *= 100
            c.step(np.r_[20,values,100], np.r_[0,np.arange(1,len(values)+1)/len(values),1], where='post', label=label)
            plotted['nearest'][name] = {'sequences':len(values), 'at_least_70':int((values>=70).sum()), 'maximum':float(values.max())}
        c.axvline(70, color='.6', ls='--', lw=.8); c.set(xlim=(20,100), ylim=(0,1.03),
            xlabel='Nearest-training identity (%)', ylabel='Cumulative fraction of proteins')
        c.legend(frameon=False, fontsize=8.5); panel(c, 'C  Held-out homology context')
        rows = [r for r in six['comparisons'] if r['scope']=='mean_of_pipeline_seeds' and
                r['target_sites']=='all_sizes' and r['identity_stratum']!='all_identities' and
                r['comparison']=='deterministic:random_minus_frozen_esm2' and r['region'] in REGIONS]
        rows.sort(key=lambda r:(REGIONS.index(r['region']), ['below_0.5','0.5_to_0.7','0.7_to_0.9','at_least_0.9','no_qualifying_hit'].index(r['identity_stratum'])))
        labels=[f'{r["region"]}\n{r["identity_stratum"].replace("below_0.5","<50%").replace("0.5_to_0.7","50–70%").replace("0.7_to_0.9","70–90%")} ({r["sequences"]}p, {r["groups"]}c)' for r in rows]
        forest(d, rows, labels);panel(d, 'D  Task − ESM by identity')
        finish(fig, 'pretraining_and_identity')

        supplemental_specs = [
            ('initialization', [('QVAE:external_minus_self','QVAE'),('deterministic:external_minus_self','Deterministic')]),
            ('model_arms', [(f'{x}:QVAE_minus_deterministic',x) for x in ('random','self','external')]),
            ('latent', [(f'{x}:joint_minus_site_marginal',x) for x in ('random','self','external')]),
            ('esm_reference', [(f'{arm}:{x}_minus_frozen_esm2',f'{"Q" if arm=="QVAE" else "D"}/{x}') for arm in ('QVAE','deterministic') for x in ('random','self','external')])]
        for name,specs in supplemental_specs:
            fig,axes=plt.subplots(1,3,figsize=(170/25.4,75/25.4),layout='constrained',sharey=True)
            for ax,region in zip(axes,('NMNAT','Nudix','global')):
                forest(ax,[lookup[m,region] for m,_ in specs],[label for _,label in specs])
                ax.set_xlabel('ΔNLL (nats / target)');panel(ax,region)
                for i,(m,_) in enumerate(specs):
                    for seed in ('9201','9202','9203'):
                        row=next(r for r in six['comparisons'] if r['comparison']==m and r['region']==region and r['scope']==seed and r['identity_stratum']=='all_identities' and r['target_sites']=='all_sizes')
                        ax.plot(row['mean'],i-.18+int(seed[-1])*.06,'^',color='.65',ms=3)
            finish(fig,f'six_cell_{name}')
        gen=table(config['generation_effects'])
        gen=[r for r in gen if r['seed']=='pooled' and r['analysis_unit']=='heldout_component']
        for contrast in ('external_minus_self','within_condition'):
            entries=[r for r in gen if r['contrast_kind']==contrast]
            if not entries: continue
            fig,axes=plt.subplots(1,3,figsize=(170/25.4,85/25.4),layout='constrained',sharey=True)
            for ax,metric in zip(axes,('mean_Hamming','unique_sequences','pair_covariance')):
                chosen=[r for r in entries if r['metric']==metric]
                if not chosen:
                    ax.axis('off');continue
                forest(ax,[{'mean':float(r['mean']),'ci95':[float(r['ci95_low']),float(r['ci95_high'])]} for r in chosen],[(r['condition'] if contrast=='within_condition' else ('QVAE' if r['comparison'].startswith('QVAE') else 'Deterministic')) for r in chosen])
                ax.set_xlabel('Difference');panel(ax,metric.replace('mean_Hamming','Substitutions').replace('unique_sequences','Unique / 384').replace('pair_covariance','Edit covariance'))
            finish(fig,f'generation_{contrast}')

        generation=table(config['generation_runs'])
        generation=[r for r in generation if r['component']!='KVP40']
        absolute=[]
        fig,axes=plt.subplots(1,2,figsize=(170/25.4,110/25.4),layout='constrained',sharey=True)
        for i,(arm,distribution,condition) in enumerate((a,d,c) for a,d in [('QVAE','joint'),('QVAE','site_marginal'),('deterministic','factorized')] for c in ('random','self','external')):
            selected=[r for r in generation if (r['arm'],r['distribution'],r['condition'])==(arm,distribution,condition)]
            _require(len(selected)==132 and len({r['component'] for r in selected})==44 and all(int(r['attempts'])==384 for r in selected),'Generation absolute denominator differs')
            record={'arm':arm,'distribution':distribution,'condition':condition}
            for ax,metric in zip(axes,('mean_Hamming','unique_sequences')):
                values=component_means(selected,metric);record[metric]=float(values.mean())
                ax.plot(values.mean(),i,'o',color='#244B62',ms=4)
                for seed in ('9201','9202','9203'):
                    ax.plot(np.mean([float(r[metric]) for r in selected if r['seed']==seed]),i,'.',color='.6',ms=3)
            absolute.append(record)
        labels=[f'{"Q joint" if r["distribution"]=="joint" else "Q marginal" if r["arm"]=="QVAE" else "D"} / {r["condition"]}' for r in absolute]
        for ax,title in zip(axes,('Substitutions','Unique sequences / 384')):
            ax.set_yticks(range(9),labels);ax.set_ylim(8.5,-.5);panel(ax,title)
        axes[1].axvline(384,color='.6',ls='--',lw=.8);axes[1].set_xlim(380,384.3)
        finish(fig,'generation_absolute');plotted['generation_absolute']=absolute

        for split in ('test', 'validation'):
            scores = table(config['scores'][split])
            prefix = Path(config['complementarity']) / split
            strata = table(prefix / 'stratum_contributions.csv')
            group_metrics = table(prefix / 'group_scores.csv')
            fig = plt.figure(figsize=(170/25.4, 165/25.4), layout='constrained')
            grid=fig.add_gridspec(3,2,wspace=.10,hspace=.2)
            axes=[fig.add_subplot(cell) for cell in grid]
            for j, region in enumerate(REGIONS):
                arrays = {m:component_means([r for r in scores if r['method']==m and r['region']==region], 'nll') for m in COLORS}
                a=axes[j]
                for i,(method,values) in enumerate(arrays.items()):
                    a.plot([i-.18,i+.18],[values.mean()]*2,color=COLORS[method],lw=2.5)
                    for seed in (9201,9202,9203):
                        v=np.mean([float(r['nll']) for r in scores if r['method']==method and r['region']==region and int(r['seed'])==seed])
                        a.plot(i,v,'.',color=COLORS[method],alpha=.65,ms=4)
                    a.text(i,values.mean()+.015,f'{values.mean():.4f}',ha='center')
                a.set_xticks(range(3),['ESM','Task','50:50']);a.set_ylabel('NLL (nats / target)');panel(a,f'{chr(65+j)}  {region}: absolute NLL')
                a.margins(y=.3)
                x,y,counts=paired_geometry(arrays['esm'],arrays['task'],arrays['single_mixture'])
                plotted['paired'][f'{split}:{region}']=counts
                a=axes[2+j]; a.scatter(x,y,c=COLORS['single_mixture'],s=14,alpha=.8)
                lim=[min(x.min(),y.min())-.03,max(x.max(),y.max())+.03]
                a.axhline(0,color='.65',lw=.8);a.axvline(0,color='.65',lw=.8)
                a.plot(lim,lim,ls='--',color='.65',lw=.8);a.set(xlim=lim,ylim=lim,
                    xlabel='Task − ESM NLL',ylabel='Mixture − ESM NLL')
                panel(a,f'{chr(67+j)}  {region}: 44 components')
                for xs,ys,key in [(.98,.03,'task_worse_mix_better'),(.03,.03,'both_better'),(.98,.9,'both_worse'),(.03,.9,'task_better_mix_worse')]:
                    a.text(xs,ys,str(counts[key]),transform=a.transAxes,ha='right' if xs>.5 else 'left',va='bottom' if ys<.5 else 'top',color='.3',bbox={'facecolor':'white','edgecolor':'none','pad':1.2},zorder=5)
                effects=[]
                for method,reference in [('single_calibrated_task','task'),('single_calibrated_mixture','single_mixture')]:
                    v=[component_means([r for r in scores if r['method']==m and r['region']==region],'nll') for m in (method,reference)]
                    effects.append(estimate(v[0]-v[1]))
                a=axes[4+j];forest(a,effects,['Task: scaled\n− raw','Mixture: scaled\n− raw']);panel(a,f'{chr(69+j)}  {region}: scaling')
            finish(fig,f'head_mixture_{split}')

            rows=[r for r in strata if r['method']=='single_mixture' and r['region']=='NMNAT']
            def subset(axis,category):
                return next(r for r in rows if r['partition']==axis and r['category']==category)
            overall=subset('all','all')
            accuracy=estimate(component_means([r for r in group_metrics if r['method']=='single_mixture' and r['region']=='NMNAT'],'accuracy_change')*100)
            plotted['accuracy'][split]=accuracy
            fig=plt.figure(figsize=(170/25.4,170/25.4), layout='constrained')
            grid=fig.add_gridspec(3,2,wspace=.12,hspace=.25)
            a,b,c,d,e=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1]),fig.add_subplot(grid[2,:])]
            matrix=np.array([[float(subset('esm_to_mixture_top1',f'esm_{ec}_mixture_{mc}')['mass'])*100 for mc in ('correct','wrong')] for ec in ('correct','wrong')])
            _require(np.isclose(matrix.sum(),100), 'Correctness transitions do not share the full denominator')
            a.imshow(matrix,cmap='Blues',vmin=0,vmax=55)
            for (i,j),v in np.ndenumerate(matrix): a.text(j,i,f'{v:.2f}%',ha='center',va='center',color='white' if v>30 else 'black')
            a.set_xticks([0,1],['Correct','Wrong']);a.set_yticks([0,1],['Correct','Wrong']);a.set_xlabel('Mixture top-1');a.set_ylabel('ESM top-1');panel(a,'A  Top-1 transitions')
            r=[subset('head_top1',cat) for cat in ('same','different')]
            forest(b,[{'mean':float(x['mixture_minus_esm']),'ci95':[float(x['delta_ci_low']),float(x['delta_ci_high'])]} for x in r],
                   [f'{label}\n{float(x["mass"])*100:.2f}% mass' for x,label in zip(r,['Agree','Differ'])]);panel(b,'B  Top-1 agreement')
            probability=[subset('esm_true_probability',cat) for cat in ('below_0.01','0.01_to_0.05','at_least_0.05')]
            labels=[f'{label}\n{float(x["mass"])*100:.2f}% mass' for x,label in zip(probability,['<0.01','0.01–<0.05','≥0.05'])]
            forest(c,[{'mean':float(x['mixture_minus_esm']),'ci95':[float(x['delta_ci_low']),float(x['delta_ci_high'])]} for x in probability],labels);panel(c,'C  ESM true-label probability')
            d.axvline(0,color='.7',ls='--',lw=.8)
            d.plot([float(x['accuracy_change'])*100 for x in probability],range(3),'o',color=COLORS['single_mixture'])
            d.set_yticks(range(3),labels);d.set_ylim(2.5,-.7);d.set_xlabel('Accuracy contribution (pp)\nDescriptive; no stratum intervals');panel(d,'D  Same strata: accuracy')
            cost=float(overall['weighted_head_cost']);gap=float(overall['jensen_gap']);delta=float(overall['mixture_minus_esm'])
            _require(np.isclose(cost-gap,delta), 'Jensen reconstruction differs')
            e.bar([0,1,2],[cost,-gap,delta],bottom=[0,cost,0],color=[COLORS['task'],COLORS['esm'],COLORS['single_mixture']],width=.55)
            for x,v in enumerate([cost,-gap,delta]):e.text(x,([cost,cost-gap,delta][x]),f'{v:+.5f}',ha='center',va='bottom' if v>0 else 'top')
            e.axhline(0,color='.7',lw=.8);e.set_xticks(range(3),['Weighted head\ndifference','Subtract\nJensen gap','Net mixture\n− ESM']);e.set_ylabel('NLL contribution');e.margins(y=.35);panel(e,'E  Log-score algebraic decomposition')
            fig.suptitle(f'NMNAT {split}: top-1 Δ {accuracy["mean"]:+.2f} pp [{accuracy["ci95"][0]:+.2f}, {accuracy["ci95"][1]:+.2f}]',fontsize=9.5)
            finish(fig,f'head_decomposition_{split}')
    write_json(output/'figure_identity.json',{'status':'COMPLETED','input_sha256':{p:sha256_file(p) for p in sorted(inputs)},
        'implementation_sha256':sha256_file(__file__),'output_sha256':{p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.suffix in ('.pdf','.svg','.png')},'width_mm':170,'minimum_font_pt':8.5,'figures':plotted,
        'bootstrap':{'repeats':config['bootstrap_repeats'],'seed':config['bootstrap_seed'],'unit':'component after seed averaging'},
        'limits':'Historically evaluated samples; conditional intervals. Probability strata use the true label and are not inference gates. Overlapping partitions are not additive to each other.'})
    return {'status':'COMPLETED','output':str(output)}

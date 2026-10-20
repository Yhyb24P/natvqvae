"""Local profile compatibility annotation; this does not measure enzyme function."""
from pathlib import Path
import pandas as pd
import pyhmmer
from .intake import load_manifest,verify_entry,sha256_file


def screen_sequences(frame:pd.DataFrame,source:Path,evalue=.001,coverage=.5):
    alphabet=pyhmmer.easel.Alphabet.amino()
    block=pyhmmer.easel.DigitalSequenceBlock(alphabet,[
        pyhmmer.easel.TextSequence(name=row.candidate_id,sequence=row.sequence).digitize(alphabet)
        for row in frame.itertuples() if row.sequence])
    records=[];inputs={}
    for name,domain in [('hmm_pf01467','NMNAT'),('hmm_pf00293','Nudix')]:
        entry=next(e for e in load_manifest()['entries'] if e['name']==name)
        result=verify_entry(entry,source)
        if not result.ok:raise ValueError(result.detail)
        inputs[name]={'path':str(result.path),'sha256':sha256_file(result.path),'pyhmmer':pyhmmer.__version__}
        with pyhmmer.plan7.HMMFile(str(result.path)) as handle:profile=handle.read()
        hits=next(iter(pyhmmer.hmmsearch([profile],block,cpus=4,E=evalue,domE=evalue)))
        for hit in hits:
            for dom in hit.domains:
                alignment=dom.alignment;span=(alignment.hmm_to-alignment.hmm_from+1)/profile.M
                identifier=hit.name.decode() if isinstance(hit.name,bytes) else hit.name
                records.append({'candidate_id':identifier,'domain':domain,'independent_evalue':float(dom.i_evalue),
                                'HMM_coverage':float(span),'from':int(alignment.target_from),'to':int(alignment.target_to),
                                'qualifies':dom.i_evalue<=evalue and span>=coverage})
    table=pd.DataFrame(records,columns=['candidate_id','domain','independent_evalue','HMM_coverage','from','to','qualifies'])
    table['qualifies']=table.qualifies.astype(bool)
    ordered=set()
    for candidate,group in table[table.qualifies].groupby('candidate_id'):
        first=group[group.domain=='NMNAT'];second=group[group.domain=='Nudix']
        if any(a.to<b[0] for a in first.itertuples() for b in second[['from']].itertuples(index=False,name=None)):
            ordered.add(candidate)
    return table,ordered,inputs

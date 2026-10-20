import pytest
np=pytest.importorskip("numpy")
pytest.importorskip("Bio")
from natvqvae.conservative_editing import scaffold,current_positions,edit_distance,wt_identity,constrained_letters,AA
from natvqvae.sequence import wt_sequence


def test_global_distance_includes_indels_and_terminal_truncation():
    wt=wt_sequence()
    assert edit_distance('AB','BA')==2
    assert wt_identity(wt)==1
    assert wt_identity(wt[:300])<.9
    assert edit_distance(wt,'AAA'+wt)==3


def test_deleted_and_inserted_coordinates_stay_explicit():
    wt=wt_sequence()
    deleted=scaffold({'type':'DELETE','anchor':23,'length':3,'sequence':wt[22:25]})
    assert deleted.sequence==wt[:22]+wt[25:]
    assert deleted.wt_to_current[22:25]==(None,None,None)
    assert current_positions(deleted,[21,25])==[21,22]
    with pytest.raises(ValueError):current_positions(deleted,[22])
    inserted=scaffold({'type':'INSERT','anchor':30,'length':2,'sequence':'GA'})
    assert inserted.sequence[30:32]=='GA'
    assert current_positions(inserted,[29,30])==[29,32]
    assert edit_distance(inserted.sequence,wt)==2


def test_sampling_preserves_requested_substitution_count():
    reference='ACDE'
    letters=constrained_letters(np.zeros((4,20)),reference,[0.,.25,.5,.999999])
    assert all(a!=b for a,b in zip(reference,letters))
    assert set(letters)<=set(AA)

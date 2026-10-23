load /home/yhshy/git_files/natvqvae/data/results/evaluation/kvp40_epcr_nmnat_environment/coordinates/reference_WT0.pdb, wt
load /home/yhshy/git_files/natvqvae/data/results/evaluation/kvp40_epcr_nmnat_environment/coordinates/ligands_1F9A.pdb, lig_1F9A
load /home/yhshy/git_files/natvqvae/data/results/evaluation/kvp40_epcr_nmnat_environment/coordinates/ligands_1NUS.pdb, lig_1NUS
load /home/yhshy/git_files/natvqvae/data/results/evaluation/kvp40_epcr_nmnat_environment/coordinates/ligands_2QJO.pdb, lig_2QJO
hide everything
show cartoon, wt
set cartoon_transparency, 0.15, wt
color grey80, wt
select nmnat, wt and resi 4-181
color palecyan, nmnat
select mut10, wt and (resi 83 or resi 162)
show spheres, mut10
color orange, mut10
select mut89, wt and (resi 21 or resi 137 or resi 156)
show spheres, mut89
color magenta, mut89
set sphere_scale, 0.45
set stick_radius, 0.15
bg_color white
orient (wt and resi 4-181) or lig_1F9A or lig_1NUS or lig_2QJO
disable lig_1F9A
disable lig_1NUS
disable lig_2QJO
enable lig_1F9A
color yellow, lig_1F9A
show sticks, lig_1F9A
png /tmp/tmp_81u3igf/state_1F9A.png
disable lig_1F9A
enable lig_1NUS
color green, lig_1NUS
show sticks, lig_1NUS
png /tmp/tmp_81u3igf/state_1NUS.png
disable lig_1NUS
enable lig_2QJO
color blue, lig_2QJO
show sticks, lig_2QJO
png /tmp/tmp_81u3igf/state_2QJO.png
disable lig_2QJO
quit

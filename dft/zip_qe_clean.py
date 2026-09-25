import os
import shutil

src = 'results_analysis/qe_proper_dft_runs'
dst = 'clean_qe_inputs'

if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst)

shutil.copy(os.path.join(src, 'README_Instructions.md'), dst)
shutil.copytree(os.path.join(src, 'pseudo'), os.path.join(dst, 'pseudo'))

for r in range(1, 6):
    rank_name = f'crystal_rank0{r}'
    s1 = 'step1_pbe_u_relax'
    s2 = 'step2_hse_static'
    
    os.makedirs(os.path.join(dst, rank_name, s1))
    os.makedirs(os.path.join(dst, rank_name, s2))
    
    shutil.copy(os.path.join(src, rank_name, s1, 'pw_relax.in'), 
                os.path.join(dst, rank_name, s1))
    shutil.copy(os.path.join(src, rank_name, s2, 'pw_hse_static.in'), 
                os.path.join(dst, rank_name, s2))

shutil.make_archive('QGAN_QE_Inputs_Clean', 'zip', dst)
print("Done creating QGAN_QE_Inputs_Clean.zip safely!")

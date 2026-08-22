# Low-Frequency-Stochastic-Gravitational-Wave-Background-in-Gaia-DR3-catalog
## License

Copyright (c) 2026 Volodymyr Akhmetov and co-authors.
All rights reserved.

The source code is provided for inspection and research
reproducibility purposes only.

No permission is granted to use, copy, modify, distribute,
publish, sublicense, or commercially exploit this software
without prior written permission from the copyright holders.

For permission to use the software, please contact the authors.

This repository contains the  Python software used for the analysis
presented in:

Akhmetov et al. (2026),
"Low-Frequency Stochastic Gravitational-Wave Background in Gaia DR3 catalog".

The repository includes:
- VSH analysis;
- HDC analysis;
- GW signal simulations;
- Monte Carlo injection-and-recovery simulations.

The programs use Gaia DR3 quasar proper motions as input.

**GW signal simulations**

python gw_simulator.py \
    --input Gaia_QSO.csv \
    --output gw_plane.csv \
    --mode plane \
    --hc 1e-11 \
    --frequency 5e-9 \
    --ra-gw 45 \
    --dec-gw 45 \
    --psi 0 \
    --phase 0 \
    --seed 99999

**VSH analysis**

    python vsh_decomposition.py \
    --input gw_simulated.csv \
    --lmax 5 \
    --coefficients vsh_coefficients.csv \
    --power vsh_power.csv

**HDC analysis**

python hdc_gamma.py \
    --input Gaia_DR3_QSO.csv \
    --output gamma_DR3.csv \
    --theta-min 0 \
    --theta-max 180 \
    --bin-width 5 \
    --healpix \
    --nside 50 \
    --pair-block 100000

**Monte Carlo injection-and-recovery simulations**

python mc_gw_simulator.py \
    --input Gaia_QSO.csv \
    --hc 1e-11 \
    --frequency 5e-9 \
    --n-mc 1000 \
    --seed 99999 \
    --equal-polarization \
    --save-catalogues \
    --output-dir MC_1e-11


    

    

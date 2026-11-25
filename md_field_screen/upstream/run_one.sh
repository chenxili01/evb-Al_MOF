#!/usr/bin/env bash
# One step of the ps_runs pipeline on a GH200 GPU node.
#
#   run_one.sh prep <run_dir>          solvate -> genion -> EM          (writes <run_dir>/em.gro)
#   run_one.sh rep  <run_dir> <N>      NVT -> NPT -> MD for replica N   (writes <run_dir>/repN/md.*)
#
# Every step is skipped if its output exists, and MD restarts from md.cpt,
# so a job that hits the wall time can simply be resubmitted.
# Env overrides: MD_NSTEPS (cap production steps, e.g. for a benchmark; the
# tpr keeps 20 ns so a later resubmission continues to the full length).
#SBATCH --account=<your-allocation>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --requeue                 # node failure -> Slurm requeues; the script resumes from md.cpt

set -euo pipefail

MODE=$1; RUN=$(readlink -f "$2"); REP=${3:-}
ROOT=$(readlink -f "$(dirname "$RUN")/../..")      # morphologies/
MDP=$ROOT/mdp
NT=${SLURM_CPUS_PER_TASK:-16}

module load GPU/GROMACS/2026.3-gcccuda-2026.03-cu13.0-es
export OMP_NUM_THREADS=$NT
G() { srun --ntasks=1 --cpus-per-task="$NT" gmx_mpi "$@"; }
MDRUN() { G mdrun -ntomp "$NT" -nb gpu -maxh 23.5 "$@"; }
MD_GPU=(-pme gpu -bonded gpu -update gpu)   # dynamics only; EM supports -nb gpu only

echo "mode=$MODE run=$RUN rep=$REP host=$(hostname) job=${SLURM_JOB_ID:-}"
cd "$RUN"

if [[ $MODE == prep ]]; then
    if [[ -s em.gro ]]; then echo "em.gro exists, nothing to do"; exit 0; fi
    # solvate/genion edit topol.top in place: always restart from the dry copy
    [[ -f topol.dry.top ]] || cp topol.top topol.dry.top
    cp topol.dry.top topol.top
    G solvate -cp start.gro -cs spc216.gro -p topol.top -o solv.gro
    G grompp -f "$MDP/em.mdp" -c solv.gro -p topol.top -o ions.tpr -po ions_mdout.mdp -maxwarn 1  # net charge
    echo SOL | G genion -s ions.tpr -p topol.top -pname K -nname CL -neutral -o ions.gro
    # K+ count must equal the number of linkers (-1 e per linker)
    nlink=$(awk '/^\[ *molecules/{f=1;next} f&&($1=="MOL"||$1=="TCC"){s+=$2} END{print s}' topol.dry.top)
    nk=$(awk '$1=="K"{print $2}' topol.top)
    [[ $nk == "$nlink" ]] || { echo "ERROR: $nk K+ but $nlink linkers" >&2; exit 1; }
    G grompp -f "$MDP/em.mdp" -c ions.gro -p topol.top -o em.tpr -po em_mdout.mdp   # no -maxwarn: must be 0 warnings
    MDRUN -deffnm em
    rm -f \#*\#
    exit 0
fi

[[ $MODE == rep && -n $REP ]] || { echo "usage: $0 prep|rep <run_dir> [N]" >&2; exit 2; }
[[ -s em.gro ]] || { echo "ERROR: $RUN/em.gro missing; run prep first" >&2; exit 1; }
mkdir -p "rep$REP"; cd "rep$REP"
TOP=../topol.top

# reproducible, distinct velocity seed per (model, run, replica)
SEED=$(printf '%s' "$(basename "$(dirname "$(dirname "$RUN")")")/$(basename "$RUN")/rep$REP" | cksum | awk '{print $1 % 2147483647}')
echo "$SEED" > gen_seed.txt

if [[ ! -s nvt.gro ]]; then
    sed "s/GENSEED/$SEED/" "$MDP/nvt.mdp" > nvt.mdp
    G grompp -f nvt.mdp -c ../em.gro -r ../em.gro -p $TOP -o nvt.tpr -po nvt_mdout.mdp
    MDRUN -deffnm nvt "${MD_GPU[@]}"
fi
if [[ ! -s npt.gro ]]; then
    G grompp -f "$MDP/npt.mdp" -c nvt.gro -r ../em.gro -t nvt.cpt -p $TOP -o npt.tpr -po npt_mdout.mdp
    MDRUN -deffnm npt "${MD_GPU[@]}"
fi
if [[ ! -f md.done ]]; then
    [[ -s md.tpr ]] || G grompp -f "$MDP/md.mdp" -c npt.gro -r ../em.gro -t npt.cpt -p $TOP -o md.tpr -po md_mdout.mdp
    extra=(); [[ -n ${MD_NSTEPS:-} ]] && extra=(-nsteps "$MD_NSTEPS")
    MDRUN -deffnm md -cpi md.cpt "${MD_GPU[@]}" "${extra[@]}"
    # finished only if md.log reached the tpr length (not a -maxh or capped stop).
    # Parsed from files, no extra srun step: a failing srun after a finished mdrun
    # killed 55 jobs on 2026-11-09 (set -e + hidden stderr).
    want=$(awk '$1=="nsteps"{print $3; exit}' md_mdout.mdp)
    got=$(awk '/^ +Step +Time/{getline; s=$1} END{print s}' md.log)
    echo "md steps: log=$got mdp=$want"
    if [[ -n $want && $got == "$want" ]] && tail -n 50 md.log | grep -q "Finished mdrun"; then
        touch md.done
    else
        echo "md not finished: resubmit to continue"
    fi
fi
rm -f \#*\#
echo done

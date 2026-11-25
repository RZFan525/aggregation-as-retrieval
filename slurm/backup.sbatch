#!/bin/bash
#SBATCH --job-name=aggmath-backup
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze
#SBATCH --output=AggMath/logs/slurm/%x-%j.out

# Snapshot the whole AggMath directory (code including uncommitted files, outputs,
# logs) into /project/pi_hzamani_umass_edu/runze/backups/aggmath_<stamp>.tar.zst,
# with a SHA-256 checksum and a file listing. The source is only read.
# Set EXCLUDE_RUNNING=1 to skip the BM25 top-m sweep outputs that may still be
# written (m = 4, 8, 32, 64, 128); back them up with a later run.

set -euo pipefail
DEST=/project/pi_hzamani_umass_edu/runze/backups
STAMP="$(date +%Y%m%d_%H%M)"
ARCHIVE="${DEST}/aggmath_${STAMP}.tar.zst"
mkdir -p "${DEST}"
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none} archive=${ARCHIVE}"

excludes=()
if [[ "${EXCLUDE_RUNNING:-0}" == "1" ]]; then
  for m in 4 8 32 64 128; do
    excludes+=("--exclude=AggMath/outputs/SingleShot_RAG/bm25_m${m}")
  done
fi

# Logs of running jobs can change while tar reads them; GNU tar then returns 1,
# which is fine. Anything above 1 is a real error.
set +e
tar "${excludes[@]}" --warning=no-file-changed \
  --use-compress-program="zstd -T${SLURM_CPUS_PER_TASK:-1} -10" -cf "${ARCHIVE}" AggMath
rc=$?
set -e
if ((rc > 1)); then
  echo "tar failed with exit code ${rc}" >&2
  exit "${rc}"
fi

(cd "${DEST}" && sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
tar --use-compress-program="zstd -d" -tf "${ARCHIVE}" > "${ARCHIVE%.tar.zst}.filelist.txt"
ls -la "${ARCHIVE%.tar.zst}"*
echo "files in archive: $(wc -l < "${ARCHIVE%.tar.zst}.filelist.txt")"
echo "done $(date -Is)"

# we use 80% CPU to run this script
# please change the number of threads as needed.

ncpus=$(grep -c ^processor /proc/cpuinfo) # get number of cpus
threads=$(($ncpus*4/5)) # number of threads
if [ ! $threads -ge 1 ]; then threads=1; fi

if command -v tee &> /dev/null
then
  if ! command -v unbuffer &> /dev/null
  then
    # echo "unbuffer is not installed"
    julia --project=../ -t $threads generator-1d-gaussian.jl | tee output1.log
    julia --project=../ -t $threads generator-poisson-neumann.jl | tee output2.log
    julia --project=../ -t $threads poisson_neumann.jl | tee output3.log
    julia --project=../ -t $threads poisson_torus.jl | tee output4.log
  else
    # echo "unbuffer is installed"
    unbuffer julia --project=../ -t $threads generator-1d-gaussian.jl | tee output1.log
    unbuffer julia --project=../ -t $threads generator-poisson-neumann.jl | tee output2.log
    unbuffer julia --project=../ -t $threads poisson_neumann.jl | tee output3.log
    unbuffer julia --project=../ -t $threads poisson_torus.jl | tee output4.log
  fi
else
    # echo "tee is not installed"
    julia --project=../ -t $threads generator-1d-gaussian.jl > output1.log
    julia --project=../ -t $threads generator-poisson-neumann.jl > output2.log
    julia --project=../ -t $threads poisson_neumann.jl > output3.log
    julia --project=../ -t $threads poisson_torus.jl > output4.log
fi

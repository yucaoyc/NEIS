# we use 75% CPU to run this script
# please change the number of threads as needed.

ncpus=$(grep -c ^processor /proc/cpuinfo) # get number of cpus
threads=$(($ncpus*3/4)) # number of threads
if [ ! $threads -ge 1 ]; then threads=1; fi

unbuffer julia --project=../ -t $threads eg1.jl | tee output_eg1.log
unbuffer julia --project=../ -t $threads eg2.jl | tee output_eg2.log
unbuffer julia --project=../ -t $threads eg3.jl | tee output_eg3.log

import numpy as np

# default
nlim = 20  # max number of cycles for sim
refinement = "none"  # static, uniform mesh
min_nodes = 1  # start with  this number of nodes
max_nodes = 128  # scale up to this number of nodes
pack_size = -1


input_str = " -i inputs/linear_wave3d.in "

param_str = (
    lambda: f" parthenon/meshblock/nx1={mb_nx[0]:d} parthenon/meshblock/nx2={mb_nx[1]:d} parthenon/meshblock/nx3={mb_nx[2]:d} parthenon/time/nlim={nlim:d} parthenon/mesh/nx1={m_nx[0]:d} parthenon/mesh/nx2={m_nx[1]:d} parthenon/mesh/nx3={m_nx[2]:d} parthenon/mesh/refinement={refinement:s} parthenon/mesh/pack_size={pack_size:d} "
)

machine = "CrusherGPU"

if machine == "CrusherGPU":
    print("# Configuration for Crusher  using 8 GPUs per node")
    print("############## ENVIRONMENT ###############")
    print(
        "module load PrgEnv-amd craype-accel-amd-gfx90a cmake hdf5 cray-python rocm/5.2.0 cray-mpich/8.1.16"
    )
    print("export MPICH_GPU_SUPPORT_ENABLED=1")
    tasks_per_gpu = 1
    # Mesh size (per GPU)
    init_m_nx = np.array([512, 512, 512], dtype=int)
    # adjust for 8 GPUs per node
    init_m_nx[0] *= 2
    init_m_nx[1] *= 2
    init_m_nx[2] *= 2
    # MeshBlock size
    mb_nx_arr = [np.array([128, 512, 512], dtype=int)]
    gpus_per_node = 8
    tasks_per_node = tasks_per_gpu * gpus_per_node
    cmd_str = "./build/bin/parthenon-hydro"
    launch_str = (
        lambda: f"srun --nodes {nodes} --ntasks {nodes*tasks_per_node} --ntasks-per-node {tasks_per_node} "
    )


for mb_nx in mb_nx_arr:
    m_nx = np.copy(init_m_nx)
    # spin up mesh size to target size
    nodes = 1
    i = 0
    while True:
        if nodes < min_nodes:
            m_nx[i] *= 2
            nodes *= 2
            i = (i + 1) % 3
        else:
            break

    print("##########################################")
    log_str = (
        lambda: f" |tee weak_static.out.cfg={machine}_nodes={nodes}_taskspernode={tasks_per_node}_mesh={m_nx[0]}x{m_nx[1]}x{m_nx[2]}_block={mb_nx[0]}x{mb_nx[1]}x{mb_nx[2]}_pack={pack_size}\n"
    )
    i = 0
    while True:
        if nodes == 1 or True:
            print(launch_str() + cmd_str + input_str + param_str() + log_str())
        m_nx[i] *= 2
        nodes *= 2
        i = (i + 1) % 3
        if nodes > max_nodes:
            break

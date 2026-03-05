import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import amep
import numpy as np

import time

def main(n_particle: str, phi: str, epsilon: str, speed_u: str, D: str, D_r: str, nskip_frames:int = 0,njobs:int = 4 ,  
                nxbins: int = 150, nybins: int = 150, rmax: float = 5.0, safe_frames: bool = False, info_end: str = "", seed: int = None, use_kdTree: bool = False):

    # info_string = f"nparticle_{n_particle}_den_{phi}_e_{epsilon}_U_{speed_u}_D_{D}_Dr_{D_r}"
    info_string = f"atom_{n_particle}_den_{phi}_e_{epsilon}_U_{speed_u}_D_{D}_Dr_{D_r}_seed_{seed}" + info_end
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(script_dir, "low_density_ABP", f"den_{phi}", f"e_{epsilon}", f"U_{speed_u}", f"D_{D}", f"Dr_{D_r}", "do00", "snap") 
    save_dir = os.path.join(script_dir, "main_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_dir = os.path.join(script_dir, "Plot_PCF")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print("AAAAA")

    traj = amep.load.traj( 
        save_dir, 
        mode='lammps', 
        dumps='dump*.txt', 
        savedir=save_dir, 
        trajfile=f'lammps_{info_string}.h5amep' 
        )


    print(f"Njobs = {njobs}")
    pcf2d = amep.evaluate.PCF2d(
        traj, 
        nav=traj.nframes, 
        nxbins=nxbins, 
        nybins=nybins,
        max_workers=njobs, 
        rmax=rmax, 
        skip=0.0,
        mode="orientations",
        use_kDTree = use_kdTree,
    )
    # just to use pcf 2d
    pcfangle = pcf2d
    # pcfangle = amep.evaluate.PCFangle(
    #     traj, 
    #     nav=traj.nframes, 
    #     nabins = 90,
    #     ndbins=200,
    #     max_workers=njobs, 
    #     rmax=rmax, 
    #     skip=0.0,
    #     mode="orientations",
    #     use_kDTree = use_kdTree,
    # )

    PCF_path = os.path.join(plot_dir, f"PCF2d_{info_string}_{use_kdTree}.h5")
    PCF_plot = os.path.join(plot_dir, f"PCF2d_{info_string}_{use_kdTree}.png")
    PCF_path_a = os.path.join(plot_dir, f"PCFangle_{info_string}_{use_kdTree}.h5")
    PCF_plot_a = os.path.join(plot_dir, f"PCFangle_{info_string}_{use_kdTree}_!!!!!!!!!!.png")
    pcf2d.save(PCF_path)
    pcfangle.save(PCF_path_a)
    fig, axs = amep.plot.new(figsize=(3.6,3))

    mp = amep.plot.field(axs, pcf2d.avg, pcf2d.x, pcf2d.y)

    cax = amep.plot.add_colorbar(

        fig, axs, mp, label=r"$g(\Delta x, \Delta y)$"

    )

    # axs.set_xlim(-5,5)

    # axs.set_ylim(-5,5)

    axs.set_xlabel(r"$\Delta x$")

    axs.set_ylabel(r"$\Delta y$")   

    fig.savefig(PCF_plot)

    r = pcfangle.r
    theta = pcfangle.theta
    X = r*np.cos(theta)
    Y = r*np.sin(theta)
    fig, axs = amep.plot.new(figsize=(3.6,3))
    mp = amep.plot.field(
        axs, pcfangle.avg, X, Y
    )
    cax = amep.plot.add_colorbar(
    fig, axs, mp, label=r"$g(\Delta x, \Delta y)$"
    )
    axs.set_xlabel(r"$\Delta x$")
    axs.set_ylabel(r"$\Delta y$")
    fig.savefig(PCF_plot_a)


if __name__ == "__main__": 
    start_time = time.time()

    n_cores_to_use = 10 
    main(
        n_particle="5000", 
        phi="0.05", 
        epsilon="50", 
        speed_u="10", D="0.1", 
        D_r="0", 
        njobs=n_cores_to_use, 
        nskip_frames=21, 
        safe_frames=False, 
        info_end="_100%", 
        seed=2525,
        rmax=15.0,
        use_kdTree = True
        )
    end_time = time.time()
    timetook = end_time - start_time
    print(f"\nTotal time: {timetook:.2f} seconds")
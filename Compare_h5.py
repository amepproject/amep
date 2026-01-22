import h5py
import numpy as np
import amep

def compare_h5_files(file1, file2):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        return compare_groups(f1, f2, path="/")

def compare_groups(g1, g2, path):
    if set(g1.keys()) != set(g2.keys()):
        print(f"Different keys at {path}")
        print("File1:", g1.keys())
        print("File2:", g2.keys())
        return False

    for key in g1.keys():
        obj1 = g1[key]
        obj2 = g2[key]
        current_path = f"{path}{key}/"

        # Both datasets
        if isinstance(obj1, h5py.Dataset):
            if not isinstance(obj2, h5py.Dataset):
                print(f"Type mismatch at {current_path}")
                return False

            if obj1.shape != obj2.shape:
                print(f"Shape mismatch at {current_path}")
                return False

            if obj1.dtype != obj2.dtype:
                print(f"Dtype mismatch at {current_path}")
                return False

            if not np.allclose(obj1[...], obj2[...], rtol=1e-1, atol=1e-1):
                print(f"Data mismatch at {current_path}")
                return False

        # Both groups
        elif isinstance(obj1, h5py.Group):
            if not isinstance(obj2, h5py.Group):
                print(f"Type mismatch at {current_path}")
                return False

            if not compare_groups(obj1, obj2, current_path):
                return False

    return True

# Usage
same = compare_h5_files("Plot_PCF/PCF2d_atom_5000_den_0.05_e_50_U_10_D_0.1_Dr_0_seed_2525_100%_False.h5", 
                        "Plot_PCF/PCF2d_atom_5000_den_0.05_e_50_U_10_D_0.1_Dr_0_seed_2525_100%_True.h5")
print("Files are identical:" if same else "Files differ")

pcf_True = amep.load.evaluation("Plot_PCF/PCF2d_atom_5000_den_0.05_e_50_U_10_D_0.1_Dr_0_seed_2525_100%_True.h5")
pcf_False = amep.load.evaluation("Plot_PCF/PCF2d_atom_5000_den_0.05_e_50_U_10_D_0.1_Dr_0_seed_2525_100%_False.h5")

npz = np.load("main_data/AVERAGE_g_xy_atom_5000_den_0.05_e_50_U_10_D_0.1_Dr_0_seed_2525_100%.npz")

avg_npz = npz["g_xy"]

new_array = avg_npz.T[3:-3, 3:-3]

diff_array = pcf_True.avg - pcf_False.avg
val = diff_array.flat[np.argmax(np.abs(diff_array))]

print(val)

diff_npz = new_array - pcf_False.avg

val2 = diff_npz.flat[np.argmax(np.abs(diff_npz))]
print(val2)
# print(diff_npz)

# PCF_plot = "Plot_PCF/PCF2d_atom_5000_den_0.05_e_50_U_10_D_0.1_Dr_0_seed_2525_100%_Difference.png"

# fig, axs = amep.plot.new(figsize=(3.6,3))

# mp = amep.plot.field(axs, pcf_True.avg - pcf_False.avg, pcf_True.x, pcf_True.y)

# cax = amep.plot.add_colorbar(

#     fig, axs, mp, label=r"$g(\Delta x, \Delta y)$"

# )

# # axs.set_xlim(-5,5)

# # axs.set_ylim(-5,5)

# axs.set_xlabel(r"$\Delta x$")

# axs.set_ylabel(r"$\Delta y$")   

# fig.savefig(PCF_plot)




if __name__ == "__main__":
    from dask_visualizer import ProgressMatrix
    from dask_visualizer.utils import generate_slow_dask_array

    da = generate_slow_dask_array((2, 128, 256), (1, 32, 32), randomize=True, delay=0.2)

    with ProgressMatrix(cmap="inferno", mode="index", scale=2) as pm:
        da.compute(num_workers=4)

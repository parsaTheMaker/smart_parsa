from data.shapenetcar_dataset import ShapeNetCarDataset
from data.ahmedml_dataset import AhmedMLDataset
from data.ahmedml_dataset_v2 import AhmedMLDatasetV2, DrivAerMLDataset
from data.shiftsuv_dataset import ShiftSUVDataset
from data.shiftwing_dataset import ShiftWingDataset
from data.naca4_dataset import NACA4Dataset


# Mapping of dataset names to their corresponding classes and properties
datasets = {"ShapeNetCar": {"dataset": ShapeNetCarDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "AhmedML": {"dataset": AhmedMLDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "AhmedMLV2": {"dataset": AhmedMLDatasetV2, "spatial_dim": 3, "surf_channels": 7, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}},
            "DrivAerML": {"dataset": DrivAerMLDataset, "spatial_dim": 3, "surf_channels": 7, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftSUV": {"dataset": ShiftSUVDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 0, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "ShiftWing": {"dataset": ShiftWingDataset, "spatial_dim": 3, "surf_channels": 1, "vol_channels": 3, "params_dim": 2, "fields": {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}},
            "NACA4": {"dataset": NACA4Dataset, "spatial_dim": 2, "surf_channels": 3, "vol_channels": 4, "params_dim": 0, "fields": {"surface": ["pressure", "normal_x", "normal_y"], "volume": ["pressure", "sdf", "velocity_x", "velocity_y"]}}
           }


def _uses_drivaerml_geometry_density(model_name):
    model_name = str(model_name)
    return model_name.startswith("SMART_SATLOSS") or "_SATLOSS" in model_name


def get_dataset(config):
    """Returns the dataset based on the provided configuration.

    Args:
        config: Configuration object containing dataset parameters.

    Returns:
        tuple: A tuple containing:
            - train_data: Training dataset.
            - test_data: Testing dataset.
            - stats: Stats for normalization.
            - spatial_dim: Spatial dimension of the dataset.
            - surf_channels: Number of surface channels.
            - vol_channels: Number of volume channels.
            - params_dim: Number of dimensions of simulation parameters.
    """
    
    dataset = config.dataset
    data_path = config.data_path
    print(f"Using dataset {dataset} stored at {data_path}")

    if dataset in datasets:
        spatial_dim = datasets[dataset]["spatial_dim"]
        surf_channels = datasets[dataset]["surf_channels"]
        vol_channels = datasets[dataset]["vol_channels"]
        params_dim = datasets[dataset]["params_dim"]
        fields = datasets[dataset]["fields"]
        dataset_kwargs = dict(geometry_points=config.num_body_points,
                              surface_points=config.num_surface_points,
                              volume_points=config.num_volume_points,
                              scale_positions=config.scale_positions)
        if dataset == "DrivAerML":
            dataset_kwargs["require_preprocessed"] = True
            dataset_kwargs["geometry_epoch_seeded_sampling"] = bool(getattr(config, "geometry_epoch_seeded_sampling", False))
            dataset_kwargs["return_sample_info"] = getattr(config, "model_name", "") == "DARM"
            dataset_kwargs["return_half_precision"] = getattr(config, "model_name", "") == "DARM" and getattr(config, "precision", "") == "float16"
            model_name = getattr(config, "model_name", "")
            if _uses_drivaerml_geometry_density(model_name):
                arch = getattr(config, "architecture", {})
                density_knn_k = int(getattr(config, "density_knn_k", getattr(arch, "density_knn_k", 8)))
                density_neighbor_hops = int(getattr(config, "density_neighbor_hops", getattr(arch, "density_neighbor_hops", 1)))
                density_estimator = getattr(config, "density_estimator", getattr(arch, "density_estimator", "rk2"))
                dataset_kwargs["geometry_density_knn_k"] = density_knn_k
                dataset_kwargs["geometry_density_neighbor_hops"] = density_neighbor_hops
                dataset_kwargs["geometry_density_estimator"] = density_estimator
                dataset_kwargs["geometry_density_cache_dtype"] = getattr(config, "geometry_density_cache_dtype", "float16")
                if _uses_drivaerml_geometry_density(model_name) and model_name != "SMART_SATLOSS":
                    dataset_kwargs["return_geometry_density"] = True
                if model_name == "SMART_SATLOSS":
                    dataset_kwargs["return_surface_density"] = True
        if dataset == "NACA4":
            dataset_kwargs["manifest_variant"] = getattr(config, "manifest_variant", "full")
        train_data = datasets[dataset]["dataset"](data_path,
                                                  if_test=False,
                                                  **dataset_kwargs)
        test_data = datasets[dataset]["dataset"](data_path,
                                                 if_test=True,
                                                 **dataset_kwargs)
        stats = [train_data.mean_surf_data, train_data.std_surf_data,
                train_data.mean_vol_data, train_data.std_vol_data]
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")
    
    return train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields


def prepare_dataset(config):
    """Prepare the dataset based on the provided configuration. Preparation means storing each sample in a 
    numpy array to speed up data loading during training and computing mean and std for normalization.

    Args:
        config: Configuration object containing dataset parameters.
    """
    
    dataset = config.dataset
    data_path = config.data_path
    print(f"Preparing dataset {dataset} stored at {data_path}")

    if dataset in datasets:
        dataset_kwargs = dict(if_test=False, prepare_data=True, copy_to_node=False)
        if dataset == "NACA4":
            dataset_kwargs["manifest_variant"] = getattr(config, "manifest_variant", "full")
        train_data = datasets[dataset]["dataset"](data_path, **dataset_kwargs)
        print(f"Dataset length: {len(train_data)}")
    else:
        raise ValueError(f"Unknown dataset ({config.dataset}) which is not supported!")

# hush_blog


第一步：内存池提取出来
第二步：提取参数（共享内存大小，线程块分块大小）
rocm_search_space = {
    "memory_management": {
        "pool_size": [4, 6, 8, 12, 16],
        "growth_factor": [1.1, 1.2, 1.5, 2.0],
        "double_buffer": [True, False],
        "param_preload": [True, False]
    },
    
    "hardware_optimization": {
        "launch_bounds": [(128,1), (256,2), (256,4), (512,1)],
        "block_dimensions": [(64,1), (128,1), (256,1), (64,2)],
        "cache_config": ["prefer_shared", "prefer_l1", "prefer_none"]
    },
    
    "memory_access": {
        "shared_mem_size": [4, 8, 16, 32],
        "vectorization": [1, 2, 4, 8],
        "bank_conflict_padding": [0, 1, 2, 4]
    },
    
    "operator_specific": {
        "conv2d": {
            "tile_size": [(8,16), (16,32), (32,64)],
            "unroll_factor": [1, 2, 4, 8]
        },
        "attention": {
            "tile_h_dim": [32, 64, 128],
            "tile_k_dim": [4, 8, 16],
            "pipeline_depth": [1, 2, 3, 4]
        }
    }
}

config.json

{
    {
        pool_size = 1
        growth_factor =
    }
}
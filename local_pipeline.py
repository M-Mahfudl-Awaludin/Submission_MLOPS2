import os
import sys
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from typing import Text

# Ensure directories exist
output_dirs = [
    "C:\\Users\\user\\Music\\submission\\submission-mlops2\\output\\happy-pipeline",
    "C:\\Users\\user\\Music\\submission\\submission-mlops2\\output\\happy-pipeline\\Transform",
    "C:\\Users\\user\\Music\\submission\\submission-mlops2\\output\\happy-pipeline\\Transform\\updated_analyzer_cache"
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Pipeline Configuration
PIPELINE_NAME = "happy-pipeline"
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/happy_transform.py"
TRAINER_MODULE_FILE = "modules/happy_trainer.py"
# requirement_file = os.path.join(root, "requirements.txt")

# pipeline outputs
OUTPUT_BASE = "C:\\Users\\user\\Music\\submission\\submission-mlops2\\output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")

def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
    "--direct_running_mode=multi_processing",
    "--direct_num_workers=0"
    ]

    
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    
    from modules.components import init_components
    
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )
    
    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
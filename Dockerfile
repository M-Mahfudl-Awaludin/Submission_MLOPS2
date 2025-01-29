FROM tensorflow/serving:latest
COPY ./output/serving_model_dir/happy-detection-model /models/cc-model
COPY ./config /model_config
# Pastikan struktur direktori model benar
RUN mv /models/cc-model/1738130904 /models/cc-model/1

ENV MODEL_NAME=cc-model

ENV MONITORING_CONFIG="/model_config/prometheus.config"

ENV PORT=8501
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh  

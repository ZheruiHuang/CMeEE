# Usage: bash run_cmeee.sh {task_id} {model_type} {model_path} {replace} {num_gpu}
# task_id:
#   0: linear
#   1: linear_nested
#   2: crf
#   3: crf_nested

SEED=2022
CBLUE_ROOT=../data/CBLUEDatasets

TASK_ID=${1:-0}
MODEL_TYPE=${2:-bert}
MODEL_PATH=${3:-../bert-base-chinese}
REPLACE=${4:-true}
NUM_GPU=${5:-2}

LABEL_NAMES=(labels)
echo "Task ID: ${TASK_ID}"

case ${TASK_ID} in
0)
  HEAD_TYPE=linear
  ;;
1)
  HEAD_TYPE=linear_nested
  LABEL_NAMES=(labels labels2)
  ;;
2)
  HEAD_TYPE=crf
  ;;
3)
  HEAD_TYPE=crf_nested
  LABEL_NAMES=(labels labels2)
  ;;
*)
  echo "Error ${TASK_ID}"
  exit -1
  ;;
esac

# if "large" in MODEL_PATH, then OUTPUT_DIR should be added _large
# OUTPUT_DIR=../ckpts/${MODEL_TYPE}_${HEAD_TYPE}_${SEED}
if [[ ${MODEL_PATH} == *"large"* ]]; then
  OUTPUT_DIR=../ckpts/${MODEL_TYPE}_large_${HEAD_TYPE}_${SEED}
else
  OUTPUT_DIR=../ckpts/${MODEL_TYPE}_${HEAD_TYPE}_${SEED}
fi

if [ ${REPLACE} = true ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_replace
fi

PYTHONPATH=../.. \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=$((RANDOM + 10000)) \
  run_cmeee.py \
  --output_dir                  ${OUTPUT_DIR} \
  --report_to                   none \
  --overwrite_output_dir        true \
  \
  --do_train                    false \
  --do_eval                     true \
  --do_predict                  true \
  \
  --dataloader_pin_memory       False \
  --per_device_eval_batch_size  16 \
  --gradient_accumulation_steps 4 \
  --eval_accumulation_steps     500 \
  \
  --logging_dir                 ${OUTPUT_DIR} \
  \
  --logging_strategy            steps \
  --logging_first_step          true \
  --logging_steps               200 \
  --save_strategy               steps \
  --save_steps                  1000 \
  --evaluation_strategy         steps \
  --eval_steps                  1000 \
  \
  --save_total_limit            1 \
  --no_cuda                     false \
  --seed                        ${SEED} \
  --dataloader_num_workers      8 \
  --disable_tqdm                true \
  --load_best_model_at_end      true \
  --metric_for_best_model       f1 \
  --greater_is_better           true \
  \
  --model_type                  ${MODEL_TYPE} \
  --model_path                  ${MODEL_PATH} \
  --head_type                   ${HEAD_TYPE} \
  \
  --cblue_root                  ${CBLUE_ROOT} \
  --max_length                  512 \
  --label_names                 ${LABEL_NAMES[@]} \
  \
  --synonyms_replacement        ${REPLACE}
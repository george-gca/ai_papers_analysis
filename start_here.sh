#!/bin/bash
# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run"
fi

# cluster_conferences=1
find_words_usage_over_conf=1
train_top2vec=1
# top2vec=1

searches=(
      # "bert"
      # "bert_based"
      "capsule"
      "catastrophic_forgetting"
      "continual_learning"
      "dall_e"
      "explainability"
      "explanatory_interactive_learning"
      "gpt"
      "incremental_learning"
      "interactive_learning"
      "interpretability"
      "large_scale pre_training"
      "llm"
      "midjourney"
      "model_editing"
      "multimodal_dataset"
      "multimodal_feature"
      "multimodal pre_training"
      "new_dataset"
      "new_multimodal_dataset"
      "pre_training"
      "rationale"
      "representation_learning"
      "scene_graph"
      "survey"
      "transformer"
      "transformer_based"
      "visual_question_answering"
      "visual_question_answering new_dataset"
      "visual_reasoning"
      "vqa"
    )


# https://github.com/awslabs/autogluon/issues/1020#issuecomment-926089808
export OPENBLAS_NUM_THREADS=15
export GOTO_NUM_THREADS=15
export OMP_NUM_THREADS=15

if [ -n "$cluster_conferences" ]; then
    echo -e "\nClustering conferences' words"
    $run_command python cluster_conference_words.py -l info

    echo -e "\nClustering conferences' papers"
    $run_command python cluster_conference_papers.py -l info

    echo -e "\nClustering search results"
    for search in "${searches[@]}"; do
        $run_command python cluster_filtered_papers.py "$search" -l info --suffix _50000w_150_clusters_pwc -y 2023
    	$run_command python cluster_filtered_papers.py "$search" -l info --suffix _50000w_150_clusters_pwc -y 2022
    	$run_command python cluster_filtered_papers.py "$search" -l info --suffix _50000w_150_clusters_pwc
    done
fi

if [ -n "$find_words_usage_over_conf" ]; then
    conferences=(
        "aaai"
        "acl"
        "coling"
        "cvpr"
        "eccv"
        "emnlp"
        "iccv"
        "iclr"
        "icml"
        "ijcai"
        "ijcnlp"
        "kdd"
        "naacl"
        "neurips"
        "sigchi"
        "sigdial"
        "tacl"
        "wacv"
    )
    for conference in "${conferences[@]}"; do
        $run_command python find_words_usage.py --suffix _50000w_150_clusters_pwc -c $conference
    done
fi

if [ -n "$train_top2vec" ]; then
    $run_command python top2vec_model.py -c -t --search ${searches[*]}
    $run_command python top2vec_model.py -c -t --year 2023
elif [ -n "$top2vec" ]; then
    $run_command python top2vec_model.py --search ${searches[*]}
    $run_command python top2vec_model.py --year 2023
fi
